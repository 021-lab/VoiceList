import {
  DEFAULT_BACKEND_MODEL, buildLiveSessionConfig, chunkDeltaLines, diffSnapshotRows,
  isLoggableEvent, readFunctionCall, snapshotRows, toTaskCommand
} from '../../src/v2/domain/live-session.js';
import { promptVersion } from '../../src/v2/domain/live-settings.js';
import { fail, safeError } from '../../src/v2/domain/contracts.js';

export const OPENAI_LIVE_SESSIONS_URL = 'https://api.openai.com/v1/live/sessions';
export const OPENAI_RESPONSES_URL = 'https://api.openai.com/v1/responses';

/** Two utterances by the same speaker are separated by a pause, not by an event. */
const SILENCE_GAP_MS = 1_500;

const PROMPT_TOOLS = new Set(['getVoicePrompt', 'getBackendPrompt', 'setVoicePrompt', 'setBackendPrompt']);

/** Owns one GPT-Live session for the document.
 *
 *  Audio stays between the browser and the model over WebRTC; this host attaches a second
 *  "sideband" WebSocket to the same session, which receives the same JSON events as the
 *  browser. That is what makes a server-side log possible at all — with the media path
 *  alone the worker never sees an event past the SDP exchange. */
export class LiveHost {
  constructor({ log, settings, services, apiKey, fetchImpl = (...args) => fetch(...args), sessionsUrl = OPENAI_LIVE_SESSIONS_URL, responsesUrl = OPENAI_RESPONSES_URL, store = true, now = () => new Date() }) {
    this.log = log;
    this.settings = settings;
    this.services = services;
    this.apiKey = apiKey;
    this.fetchImpl = fetchImpl;
    this.sessionsUrl = sessionsUrl;
    this.responsesUrl = responsesUrl;
    this.store = store;
    this.now = now;
    this.session = null;
  }

  get active() { return Boolean(this.session?.socket); }
  get sessionId() { return this.session?.id || ''; }

  stamp() { return this.now().toISOString(); }

  /** The session id is passed explicitly where the handler outlives the session, so a
   *  closing event is still attributed rather than landing under an empty id. */
  record(event, direction = 'in', sessionId = this.sessionId) {
    try { this.log.append(event, { sessionId, direction, at: this.stamp() }); } catch { /* the log must never break the call */ }
  }

  send(message) {
    const socket = this.session?.socket;
    if (!socket) return false;
    const event = { event_id: `vl_${++this.session.outgoing}`, ...message };
    try { socket.send(JSON.stringify(event)); } catch { return false; }
    this.record(event, 'out');
    return true;
  }

  async start({ sdp }) {
    if (!this.apiKey) fail('MODEL_UNAVAILABLE', 'Ключ OpenAI не настроен');
    if (!sdp || typeof sdp !== 'string' || sdp.length > 120_000) fail('INVALID_INPUT', 'Некорректное SDP-предложение');
    if (this.active) await this.stop('replaced');

    const items = await this.services.readItems();
    const [voicePrompt, backendPrompt, backendModel, reasoningEffort] = await Promise.all([
      this.settings.prompt('voice'), this.settings.prompt('backend'), this.settings.backendModel(), this.settings.reasoningEffort()
    ]);
    const config = buildLiveSessionConfig({ items, voicePrompt, backendPrompt, backendModel, reasoningEffort, store: this.store });

    // Every attempt leaves a trace, including one that never reaches OpenAI: an attempt
    // missing from the log would be indistinguishable from one that was never made.
    this.record({ type: 'vl.session.requested', backendModel: backendModel || DEFAULT_BACKEND_MODEL, store: this.store, sdpChars: sdp.length }, 'out');

    let created;
    try {
      created = await this.fetchImpl(this.sessionsUrl, {
        method: 'POST',
        headers: { Authorization: `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ session: config, transport: { type: 'webrtc', sdp } })
      });
    } catch (error) {
      this.record({ type: 'vl.session.unreachable', error: { name: error?.name, message: error?.message } }, 'out');
      fail('MODEL_UNAVAILABLE', `Не удалось обратиться к GPT-Live: ${error?.message || 'сеть недоступна'}`);
    }
    if (!created.ok) {
      const detail = await created.text().catch(() => '');
      this.record({ type: 'vl.session.create_failed', status: created.status, detail: detail.slice(0, 2_000) }, 'out');
      fail('MODEL_UNAVAILABLE', `GPT-Live отклонил сессию (${created.status}): ${detail.slice(0, 300)}`);
    }
    let payload;
    try { payload = await created.json(); }
    catch (error) {
      this.record({ type: 'vl.session.unreadable', error: { message: error?.message } }, 'out');
      fail('MODEL_UNAVAILABLE', 'GPT-Live вернул нечитаемый ответ');
    }
    const id = String(payload?.session?.id || '');
    const answer = String(payload?.transport?.sdp || '');
    if (!id || !answer) fail('MODEL_UNAVAILABLE', 'GPT-Live не вернул идентификатор сессии');

    this.session = { id, socket: null, outgoing: 0, seq: 0, rows: snapshotRows(items), speech: null, backendText: null, pending: [], closing: false };

    // The session record carries the prompt versions and the model name: an entry from last
    // week cannot be read without knowing which text drove the model then. The response is
    // logged whole, so an effective `store` that differs from the requested one shows up here.
    this.record({
      type: 'vl.session.created',
      session: payload.session,
      requested: { store: this.store, backendModel: backendModel || DEFAULT_BACKEND_MODEL, voicePromptVersion: promptVersion(voicePrompt), backendPromptVersion: promptVersion(backendPrompt) }
    }, 'out');

    await this.attach(id);
    return { sessionId: id, sdp: answer };
  }

  /** Ephemeral keys cannot look a session up server side, so the sideband uses the project
   *  key — which is why the session is created here rather than in the browser. */
  async attach(id) {
    let socket = null;
    try {
      const response = await this.fetchImpl(`${this.sessionsUrl}/${encodeURIComponent(id)}/attach`, {
        headers: { Authorization: `Bearer ${this.apiKey}`, Upgrade: 'websocket' }
      });
      socket = response.webSocket;
      if (!socket) fail('MODEL_UNAVAILABLE', `Sideband не открылся (${response.status})`);
      socket.accept();
    } catch (error) {
      this.record({ type: 'vl.sideband.failed', error: safeError(error) }, 'out');
      this.session = null;
      // Without the sideband there is no log and no tools, only a voice that silently cannot
      // act. Failing here is honest; the browser tears its peer connection down.
      fail('MODEL_UNAVAILABLE', 'Не удалось подключить наблюдение за сессией');
    }
    this.session.socket = socket;
    socket.addEventListener('message', event => { this.receive(event.data); });
    socket.addEventListener('close', () => { this.record({ type: 'vl.sideband.closed' }, 'out', id); this.session = null; });
    socket.addEventListener('error', () => { this.record({ type: 'vl.sideband.error' }, 'out', id); });
  }

  /** Transcript fragments follow audio cadence, so one row per fragment buries the log in
   *  syllables.
   *
   *  What ends a spoken turn is the speech itself: the other speaker starting, or a silence
   *  long enough to separate two utterances. The backend's own stream runs in parallel in a
   *  full-duplex model, so its events are not a boundary — treating them as one cut phrases
   *  wherever a response lifecycle event happened to land. */
  bufferSpeech(role, event) {
    const speech = this.session.speech;
    const startMs = Number.isFinite(event?.start_ms) ? event.start_ms : null;
    const endMs = Number.isFinite(event?.end_ms) ? event.end_ms : null;
    const silence = speech && startMs != null && speech.endMs != null && startMs - speech.endMs > SILENCE_GAP_MS;
    if (speech && (speech.role !== role || silence)) this.flushSpeech();
    if (!this.session.speech) this.session.speech = { role, text: '', startMs, endMs };
    this.session.speech.text += String(event?.delta ?? '');
    if (endMs != null) this.session.speech.endMs = endMs;
  }

  flushSpeech() {
    const speech = this.session?.speech;
    if (!speech) return;
    this.session.speech = null;
    const text = speech.text.trim();
    if (!text) return;
    // Turns since the last delegation, not a rolling window: a window mixes the request that
    // is being made now with one that was already answered.
    this.session.pending = [...(this.session.pending || []), { role: speech.role, text }].slice(-12);
    this.record({ type: 'vl.speech', role: speech.role, text, start_ms: speech.startMs, end_ms: speech.endMs });
  }

  /** The backend's own words arrive as a stream too, and they are what the log has to show
   *  beside the delegation that asked for them. Assembled the same way speech is. */
  bufferBackendText(event) {
    if (!this.session) return;
    const delegationId = event?.delegation_id || this.session.backendText?.delegationId || null;
    if (!this.session.backendText || this.session.backendText.delegationId !== delegationId) {
      this.flushBackendText();
      this.session.backendText = { delegationId, text: '' };
    }
    this.session.backendText.text += String(event?.event?.delta ?? '');
  }

  flushBackendText() {
    const pending = this.session?.backendText;
    if (!pending) return;
    this.session.backendText = null;
    const text = pending.text.trim();
    if (!text) return;
    this.record({ type: 'vl.backend_text', delegation_id: pending.delegationId, text });
  }

  /** What actually caused this delegation. The framework does not expose the input it sent to
   *  the backend — a response.created carries the model, tools and settings but neither the
   *  instructions nor the input — so the closest true thing is the exchange since the previous
   *  delegation. Assistant turns that open the window belong to the exchange that just ended,
   *  so they are dropped. */
  takeContext() {
    const turns = this.session?.pending || [];
    if (this.session) this.session.pending = [];
    let start = 0;
    while (start < turns.length && turns[start].role !== 'user') start += 1;
    return turns.slice(start);
  }

  /** The literal context the voice layer handed the backend.
   *
   *  It is not in any event: a response.created carries the model, tools and settings but
   *  neither instructions nor input. Since the responses are stored, the input can be read
   *  back by id, and that is the real thing rather than a reconstruction. */
  /** Reads the stored input of a past response, which is also how a delegation already in the
   *  log can be inspected after the fact. */
  async readBackendInput(responseId) {
    if (!this.apiKey) fail('MODEL_UNAVAILABLE', 'Ключ OpenAI не настроен');
    const reply = await this.fetchImpl(`${this.responsesUrl}/${encodeURIComponent(String(responseId))}/input_items?order=asc&limit=100`, {
      headers: { Authorization: `Bearer ${this.apiKey}` }
    });
    const body = await reply.text();
    if (!reply.ok) fail('MODEL_UNAVAILABLE', `Вход не прочитан (${reply.status}): ${body.slice(0, 300)}`);
    try { return JSON.parse(body); } catch { return { raw: body.slice(0, 2_000) }; }
  }

  async fetchBackendInput(delegationId, response) {
    const responseId = response?.id;
    if (!responseId || !this.apiKey) return;
    try {
      const reply = await this.fetchImpl(`${this.responsesUrl}/${encodeURIComponent(responseId)}/input_items?order=asc&limit=100`, {
        headers: { Authorization: `Bearer ${this.apiKey}` }
      });
      if (!reply.ok) {
        const detail = await reply.text().catch(() => '');
        this.record({ type: 'vl.backend_input.failed', delegation_id: delegationId, response_id: responseId, status: reply.status, detail: detail.slice(0, 800) }, 'out');
        return;
      }
      const payload = await reply.json();
      this.record({
        type: 'vl.backend_input', delegation_id: delegationId, response_id: responseId,
        previous_response_id: response?.previous_response_id || null,
        instructions: response?.instructions ?? null,
        items: payload?.data ?? payload
      });
    } catch (error) {
      this.record({ type: 'vl.backend_input.failed', delegation_id: delegationId, response_id: responseId, error: { message: error?.message } }, 'out');
    }
  }

  receive(raw) {
    let event;
    try { event = JSON.parse(typeof raw === 'string' ? raw : new TextDecoder().decode(raw)); }
    catch { this.flushSpeech(); this.record({ type: 'vl.event.unparsed', raw: String(raw).slice(0, 2_000) }); return; }

    const inner = event?.event;
    if (event?.type === 'session.input_transcript.delta') this.bufferSpeech('user', event);
    else if (event?.type === 'session.output_transcript.delta') this.bufferSpeech('assistant', event);
    else if (inner?.type === 'response.output_text.delta') this.bufferBackendText(event);
    else if (event?.type === 'session.delegation.created') {
      this.flushSpeech();
      this.record({ type: 'vl.delegation', delegation: event.delegation, offset_ms: event.offset_ms, context: this.takeContext(), raw: event });
    }
    else {
      // Delegation events arrive wrapped in response.event, so the skip list has to be applied
      // to the inner type as well — otherwise every streamed delta is kept after all.
      if (inner?.type === 'response.completed') this.flushBackendText();
      if (isLoggableEvent(event?.type) && isLoggableEvent(inner?.type ?? 'none')) this.record(event);
      if (inner?.type === 'response.created') {
        this.fetchBackendInput(event?.delegation_id || null, inner.response).catch(() => {});
      }
    }
    this.dispatch(event).catch(error => this.record({ type: 'vl.dispatch.failed', error: safeError(error) }, 'out'));
  }

  async dispatch(event) {
    if (event?.type === 'session.closed') { this.flushSpeech(); this.flushBackendText(); this.session = null; return; }
    if (event?.type !== 'response.event') return;
    const inner = event.event;
    if (inner?.type !== 'response.output_item.done') return;
    const call = readFunctionCall(inner);
    if (!call) return;

    let output;
    try { output = await this.invoke(call); }
    catch (error) { output = { status: 'rejected', reason: safeError(error).message }; }

    this.send({ type: 'response.item.create', item: { type: 'function_call_output', call_id: call.callId, output: JSON.stringify(output) } });
    // parallel_tool_calls is off, so a response carries one call at a time and resuming here
    // never strands a second pending result.
    this.send({ type: 'response.create' });
  }

  async invoke(call) {
    if (call.name === 'getFrontier') return { status: 'ok', frontier: await this.services.readFrontier() };
    if (PROMPT_TOOLS.has(call.name)) return this.invokePrompt(call);

    const command = toTaskCommand(call.name, call.arguments);
    const ack = await this.services.applyCommand(command, { clientKey: `gpt-live:${this.sessionId}`, seq: ++this.session.seq });
    if (ack?.status === 'rejected') return { status: 'rejected', reason: ack.reason || 'Операция отклонена' };
    await this.syncSnapshot();
    return { status: 'applied', operation: command.command, target: ack?.newTarget || command.actId };
  }

  async invokePrompt(call) {
    const target = call.name.includes('Voice') ? 'voice' : 'backend';
    if (call.name.startsWith('get')) return { status: 'ok', target, prompt: await this.settings.prompt(target) };
    const result = await this.settings.writePrompt(target, { mode: call.arguments.mode, text: call.arguments.text, source: 'model' });
    if (result.changed) await this.applyPromptChange(target, result.prompt, call.arguments);
    return { status: result.changed ? 'applied' : 'unchanged', target, chars: result.prompt.length };
  }

  /** Prompts are fixed at session creation, so a saved edit alone would look ignored in the
   *  conversation that asked for it. The voice layer's instructions are immutable after
   *  startup and can only be appended to; the backend's are updatable in place. */
  async applyPromptChange(target, prompt, args) {
    if (target === 'backend') {
      this.send({ type: 'session.update', session: { delegation: { type: 'responses', responses: { instructions: prompt } } } });
      return;
    }
    const note = args?.mode === 'append' ? String(args.text || '').trim() : prompt;
    for (const chunk of chunkDeltaLines(note.split('\n').filter(Boolean))) {
      this.send({ type: 'session.instructions.append', delegation_id: null, content: chunk });
    }
  }

  /** Keeps the voice layer's table current after any change, including one made by hand in
   *  the interface, by diffing the rendered rows rather than the graph's own change list. */
  async syncSnapshot() {
    if (!this.active) return [];
    const rows = snapshotRows(await this.services.readItems());
    const lines = diffSnapshotRows(this.session.rows, rows);
    this.session.rows = rows;
    for (const chunk of chunkDeltaLines(lines)) {
      this.send({ type: 'session.instructions.append', delegation_id: null, content: `Изменения в задачах:\n${chunk}` });
    }
    return lines;
  }

  async stop(reason = 'client') {
    if (!this.session) return false;
    this.flushSpeech();
    this.flushBackendText();
    this.send({ type: 'session.close' });
    this.record({ type: 'vl.session.stopped', reason }, 'out');
    try { this.session.socket?.close(); } catch { /* already gone */ }
    this.session = null;
    return true;
  }
}
