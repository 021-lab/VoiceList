import {
  DEFAULT_BACKEND_MODEL, LIVE_TOOLS, buildLiveSessionConfig, chunkDeltaLines, diffSnapshotRows,
  isLoggableEvent, readFunctionCall, snapshotRows, toTaskCommand, buildSimulatedInput,
  buildSimulatedVoiceInput, composeVoiceInstructions, composeBackendInstructions, formatTaskSnapshot, DEFAULT_VOICE_PROMPT
} from '../../src/v2/domain/live-session.js';
import { promptVersion } from '../../src/v2/domain/live-settings.js';
import { MAX_HELD, asideFor, buildSolInput, confirmText, deliveryFor, stillUnsaid } from '../../src/v2/domain/live-delegation.js';
import { parseCommand, toCommand } from '../../src/command-resolver.js';
import { fail, safeError } from '../../src/v2/domain/contracts.js';

export const OPENAI_LIVE_SESSIONS_URL = 'https://api.openai.com/v1/live/sessions';
export const OPENAI_RESPONSES_URL = 'https://api.openai.com/v1/responses';

/** Two utterances by the same speaker are separated by a pause, not by an event. */
const SILENCE_GAP_MS = 1_500;

/** How much of the conversation the backend reads, and how many passes it may take. */
const DIALOG_TURNS = 12;
const MAX_BACKEND_PASSES = 4;

/** A spoken update sent while the voice is still talking can be lost, so it waits for this
 *  much quiet — but not longer than the cap, after which it goes anyway. */
const QUIET_MS = 500;
const QUIET_WAIT_CAP_MS = 6_000;

/** How long after the person's turn a handover may still arrive before the voice is taken to
 *  be answering by itself. */
const ASIDE_GRACE_MS = 1_500;

/** The tail of a request can still be in transcription when the handover arrives. */
const TRANSCRIPT_GRACE_MS = 400;

/** The local parser reads «добавь X» and takes everything after the verb as the title, so it
 *  is trusted only with a plain title: a place, a date or a second request in the same breath
 *  would end up inside the title. Those go to the backend. */
const NOT_A_PLAIN_TITLE = /(^|\s)(в|во|на|под|к|ко|до|за|и|а|потом|ещё|еще|через|завтра|сегодня|послезавтра|срок|дедлайн)(\s|$)/i;


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
  today() { return this.stamp().slice(0, 10); }

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
    const [voicePrompt, backendPrompt, backendModel] = await Promise.all([
      this.settings.prompt('voice'), this.settings.prompt('backend'), this.settings.backendModel()
    ]);
    const config = buildLiveSessionConfig({ items, voicePrompt, store: this.store });

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

    this.session = {
      id, socket: null, outgoing: 0, seq: 0, rows: snapshotRows(items), speech: null, closing: false,
      dialog: [], handedOver: 0, lastUserStartMs: -1, delegations: 0, voiceAt: 0, userAt: 0, handoverMs: -1, spoken: Promise.resolve(), held: []
    };

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
    if (role === 'user' && startMs != null) this.session.lastUserStartMs = Math.max(this.session.lastUserStartMs, startMs);
    if (role === 'assistant') this.session.voiceAt = Date.now();
    // The request itself was speech too; only words after the last handover are someone
    // talking now.
    else if (startMs == null || startMs > this.session.handoverMs) this.session.userAt = Date.now();
    this.session.speech.text += String(event?.delta ?? '');
    if (endMs != null) this.session.speech.endMs = endMs;
  }

  flushSpeech() {
    const speech = this.session?.speech;
    if (!speech) return;
    this.session.speech = null;
    const text = speech.text.trim();
    if (!text) return;
    this.record({ type: 'vl.speech', role: speech.role, text, start_ms: speech.startMs, end_ms: speech.endMs });
    this.session.dialog.push({ role: speech.role, text });
    if (speech.role === 'user') this.offerAside(this.session);
    // The same turn goes on the bus, where the change it causes will point back at it. The
    // session log keeps its own copy with the timeline; the bus keeps the words. Turns are
    // written one after another: an answer can only be linked to the turn before it once
    // that turn has an id.
    const write = () => this.writeTurn(speech.role, text);
    this.turns = (this.turns || Promise.resolve()).then(write, write);
  }

  async writeTurn(role, text) {
    try {
      const entry = await this.services.recordSpeech?.({
        role, text, source: 'gpt-live', sessionId: this.sessionId,
        answers: role === 'assistant' ? this.session?.utterance || null : null
      });
      if (entry?.requestId && role === 'user' && this.session) this.session.utterance = entry.requestId;
    } catch (error) { this.record({ type: 'vl.speech.failed', error: safeError(error) }, 'out'); }
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

  async fetchBackendInput(delegationId, response, sessionId = this.sessionId) {
    const responseId = response?.id;
    if (!responseId || !this.apiKey) return;
    try {
      const reply = await this.fetchImpl(`${this.responsesUrl}/${encodeURIComponent(responseId)}/input_items?order=asc&limit=100`, {
        headers: { Authorization: `Bearer ${this.apiKey}` }
      });
      if (!reply.ok) {
        const detail = await reply.text().catch(() => '');
        this.record({ type: 'vl.backend_input.failed', delegation_id: delegationId, response_id: responseId, status: reply.status, detail: detail.slice(0, 800) }, 'out', sessionId);
        return;
      }
      const payload = await reply.json();
      this.record({
        type: 'vl.backend_input', delegation_id: delegationId, response_id: responseId,
        previous_response_id: response?.previous_response_id || null,
        instructions: response?.instructions ?? null,
        items: payload?.data ?? payload
      }, 'in', sessionId);
    } catch (error) {
      this.record({ type: 'vl.backend_input.failed', delegation_id: delegationId, response_id: responseId, error: { message: error?.message } }, 'out', sessionId);
    }
  }

  /** Re-reads the inputs of responses whose context never made it into the log, which is what
   *  the early-fetch race left behind. Safe to run again: a response already recorded is
   *  skipped. */
  async repairBackendInputs(rows) {
    const done = new Set(rows.filter(row => row.type === 'vl.backend_input').map(row => row.responseId || row.payload?.response_id));
    const wanted = new Map();
    for (const row of rows) {
      const responseId = row.responseId || row.payload?.response_id;
      if (!responseId || done.has(responseId) || wanted.has(responseId)) continue;
      const finished = row.type.endsWith('response.completed') || row.type === 'vl.backend_input.failed';
      if (finished) wanted.set(responseId, { sessionId: row.liveSessionId || '', delegationId: row.delegationId || row.payload?.delegation_id || null });
    }
    let repaired = 0;
    for (const [responseId, where] of wanted) {
      const before = repaired;
      await this.fetchBackendInput(where.delegationId, { id: responseId }, where.sessionId);
      repaired = before + 1;
    }
    return { requested: wanted.size, repaired };
  }

  /** Runs one delegation against the backend from a written transcript, so the half that
   *  picks a tool can be tested without speech. Nothing is executed and nothing is logged as
   *  a session event: the reply is returned for inspection. */
  async simulate(turns) {
    if (!this.apiKey) fail('MODEL_UNAVAILABLE', 'Ключ OpenAI не настроен');
    const [backendPrompt, backendModel, reasoningEffort, items] = await Promise.all([
      this.settings.prompt('backend'), this.settings.backendModel(), this.settings.reasoningEffort(), this.services.readItems()
    ]);
    const body = {
      model: backendModel || DEFAULT_BACKEND_MODEL,
      instructions: composeBackendInstructions(backendPrompt, formatTaskSnapshot(items), this.today()),
      input: buildSimulatedInput(turns),
      tools: LIVE_TOOLS,
      tool_choice: 'auto',
      store: false,
      ...(reasoningEffort ? { reasoning: { effort: reasoningEffort } } : {})
    };
    const reply = await this.fetchImpl(this.responsesUrl, {
      method: 'POST',
      headers: { Authorization: `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const text = await reply.text();
    if (!reply.ok) fail('MODEL_UNAVAILABLE', `Бэкенд отклонил запрос (${reply.status}): ${text.slice(0, 400)}`);
    const payload = JSON.parse(text);
    const output = payload.output || [];
    return {
      model: body.model,
      snapshotTasks: snapshotRows(items).length,
      calls: output.filter(item => item.type === 'function_call').map(item => ({ name: item.name, arguments: item.arguments })),
      text: output.filter(item => item.type === 'message').flatMap(item => (item.content || []).map(part => part.text)).filter(Boolean).join(' '),
      usage: payload.usage || null
    };
  }

  /** Replays a written dialogue against the voice layer's own instructions. The live model
   *  itself is not a Responses model, so the stand-in is the backend model reading the same
   *  instruction text; what it shows is whether the prompt leads to asking which of several
   *  tasks is meant and to saying the chosen identifier aloud. Nothing is changed and no
   *  tools are offered — the voice layer has none of its own. */
  async simulateVoice(turns, { prompt = 'current' } = {}) {
    if (!this.apiKey) fail('MODEL_UNAVAILABLE', 'Ключ OpenAI не настроен');
    // A saved edit hides the shipped text, so a change to the default cannot be judged
    // against the stored prompt. Asking for the default explicitly reads it without
    // touching what is saved — the alternative, resetting to test, destroys the edit.
    const [voicePrompt, backendModel, items] = await Promise.all([
      prompt === 'default' ? DEFAULT_VOICE_PROMPT : this.settings.prompt('voice'),
      this.settings.backendModel(), this.services.readItems()
    ]);
    const body = {
      model: backendModel || DEFAULT_BACKEND_MODEL,
      instructions: composeVoiceInstructions(voicePrompt, formatTaskSnapshot(items)),
      input: buildSimulatedVoiceInput(turns),
      store: false
    };
    const reply = await this.fetchImpl(this.responsesUrl, {
      method: 'POST',
      headers: { Authorization: `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const text = await reply.text();
    if (!reply.ok) fail('MODEL_UNAVAILABLE', `Модель отклонила запрос (${reply.status}): ${text.slice(0, 400)}`);
    const payload = JSON.parse(text);
    const output = payload.output || [];
    return {
      layer: 'voice',
      prompt: prompt === 'default' ? 'default' : 'current',
      model: body.model,
      snapshotTasks: snapshotRows(items).length,
      text: output.filter(item => item.type === 'message').flatMap(item => (item.content || []).map(part => part.text)).filter(Boolean).join(' '),
      usage: payload.usage || null
    };
  }

  receive(raw) {
    let event;
    try { event = JSON.parse(typeof raw === 'string' ? raw : new TextDecoder().decode(raw)); }
    catch { this.flushSpeech(); this.record({ type: 'vl.event.unparsed', raw: String(raw).slice(0, 2_000) }); return; }

    const inner = event?.event;
    if (event?.type === 'session.input_transcript.delta') this.bufferSpeech('user', event);
    else if (event?.type === 'session.output_transcript.delta') this.bufferSpeech('assistant', event);
    else if (event?.type === 'session.delegation.created') {
      this.flushSpeech();
      this.record({ type: 'vl.delegation', delegation: event.delegation, offset_ms: event.offset_ms, raw: event });
      if (event.delegation?.id) this.handleDelegation(event.delegation.id, event.offset_ms).catch(error => this.record({ type: 'vl.delegation.failed', error: safeError(error) }, 'out'));
    }
    else {
      // Delegation events arrive wrapped in response.event, so the skip list has to be applied
      // to the inner type as well — otherwise every streamed delta is kept after all.
      if (isLoggableEvent(event?.type) && isLoggableEvent(inner?.type ?? 'none')) this.record(event);
      // Read the input once the response is stored: at creation time it is not there yet and
      // the API answers "not found".
      if (inner?.type === 'response.completed') {
        this.fetchBackendInput(event?.delegation_id || null, inner.response, this.sessionId).catch(() => {});
      }
    }
    if (event?.type === 'session.closed') { this.flushSpeech(); this.session = null; }
  }

  /** A request the voice handed over. Answered by the local parser when it is sure, otherwise
   *  by the backend model; the result reaches the voice by the owner's timing rule. */
  async handleDelegation(delegationId, offsetMs) {
    const session = this.session;
    if (!session) return;
    const order = ++session.delegations;
    session.handoverMs = Math.max(session.handoverMs, offsetMs ?? -1);
    session.userAt = 0;
    const startedAt = Date.now();
    if (!session.dialog.slice(session.handedOver).some(turn => turn.role === 'user')) {
      await new Promise(resolve => setTimeout(resolve, TRANSCRIPT_GRACE_MS));
      if (this.session === session) this.flushSpeech();
    }
    const turns = session.dialog.slice(session.handedOver);
    session.handedOver = session.dialog.length;
    const movedOn = () => this.session !== session || session.delegations > order || session.lastUserStartMs > (offsetMs ?? Infinity);
    // Turns are written to the journal one after another; the change has to point at this one.
    await this.turns;

    let reported = false;
    // A request can yield more than one result: the backend's writes are confirmed at once
    // and it finishes the rest of the request afterwards. Each result meets the timing rule.
    const report = (result) => {
      reported = true;
      if (this.session !== session || !result?.text) return;
      const elapsedMs = Date.now() - startedAt;
      // The conversation going on means the person, not the voice saying our own aside.
      const active = movedOn();
      const delivery = deliveryFor({ elapsedMs, active, failed: result.failed, answer: result.answer, text: result.text });
      this.record({ type: 'vl.delegation.result', delegation_id: delegationId, by: result.by || null, elapsed_ms: elapsedMs, failed: Boolean(result.failed), delivery }, 'out');
      if (delivery.kind === 'speak') { this.speak(session, delegationId, delivery.content); return; }
      if (delivery.kind === 'quiet') { this.append('session.thinking.append', delegationId, `Ответ, не озвучен: ${delivery.content}`); return; }
      // No quiet note about it: the voice read such notes aloud by itself. It knows the new
      // state from the table deltas, and an aside it has already covered is dropped.
      session.held = [...session.held, { text: delivery.content, since: session.dialog.length }].slice(-MAX_HELD);
    };
    // The pause after a new question is where held confirmations of earlier requests go:
    // «Думаю, кстати, …» while the backend works, a plain «Кстати, …» when it will not.
    const fillPause = (thinking) => {
      const aside = asideFor(this.unsaid(session), { thinking });
      session.held = [];
      if (aside) this.speak(session, delegationId, aside);
    };
    try { await this.answer(turns, session, report, fillPause); }
    catch (error) { report({ failed: true, text: `Не вышло: ${safeError(error).message}` }); }
    // Nothing to say is not success: a made-up «готово» is worse than an honest failure.
    if (!reported) report({ failed: true, text: 'Не вышло: ответа нет' });
  }

  /** A question the voice answers by itself is never handed over, so held confirmations would
   *  wait for nothing. If no handover follows the person's turn, they go after the voice's own
   *  answer: speak() waits for it to fall quiet. */
  unsaid(session) {
    const since = Math.min(...session.held.map(item => item.since));
    const voice = session.dialog.slice(since).filter(turn => turn.role === 'assistant').map(turn => turn.text);
    if (session.speech?.role === 'assistant') voice.push(session.speech.text);
    return stillUnsaid(session.held.map(item => item.text), voice);
  }

  offerAside(session) {
    if (!session.held.length) return;
    const handovers = session.delegations;
    setTimeout(() => {
      if (this.session !== session || session.delegations !== handovers || !session.held.length) return;
      const aside = asideFor(this.unsaid(session), { thinking: false });
      session.held = [];
      if (!aside) return;
      this.speak(session, null, aside);
    }, ASIDE_GRACE_MS);
  }

  /** Spoken updates go one at a time and each waits for the voice to fall quiet: one sent
   *  over the voice's own speech was dropped in a live test (the first of two was said, the
   *  one right after it never was). */
  speak(session, delegationId, content) {
    const say = async () => {
      const until = Date.now() + QUIET_WAIT_CAP_MS;
      // Quiet means both: the voice has finished and the person is not mid-sentence.
      const busy = () => Date.now() - Math.max(session.voiceAt, session.userAt) < QUIET_MS;
      while (this.session === session && busy() && Date.now() < until) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      if (this.session === session) this.append('session.commentary.append', delegationId, content);
    };
    session.spoken = session.spoken.then(say, say);
    return session.spoken;
  }

  append(type, delegationId, content) {
    return this.send({ type, delegation_id: delegationId, content: String(content || '').slice(0, 1_500) });
  }

  async answer(turns, session, report, fillPause = () => {}) {
    if (this.session !== session) return;
    const said = turns.filter(turn => turn.role === 'user');
    const parsed = said.length === 1 ? parseCommand(said[0].text.trim(), null) : { kind: 'many' };
    const local = parsed.kind === 'one' ? toCommand(parsed.hypothesis, null) : null;
    if (local?.command === 'addChild' && !NOT_A_PLAIN_TITLE.test(local.payload?.title || '')) {
      fillPause(false);
      const outcome = await this.apply({ ...local, source: 'gpt-live' });
      report(outcome.status === 'rejected'
        ? { by: 'parser', failed: true, text: `Не вышло: ${outcome.reason}` }
        : { by: 'parser', failed: false, text: outcome.confirm });
      return;
    }
    fillPause(true);
    await this.askBackend(session, result => report({ by: 'backend', ...result }));
  }

  /** Writes run here and are confirmed by the server in a few words the moment they land; the
   *  backend's own pass after them only continues a request that holds more than one change.
   *  Its closing words after writes are dropped: the server has already said them. */
  async askBackend(session, report) {
    if (!this.apiKey) fail('MODEL_UNAVAILABLE', 'Ключ OpenAI не настроен');
    const [backendPrompt, backendModel, reasoningEffort, items] = await Promise.all([
      this.settings.prompt('backend'), this.settings.backendModel(), this.settings.reasoningEffort(), this.services.readItems()
    ]);
    const base = {
      model: backendModel || DEFAULT_BACKEND_MODEL,
      instructions: composeBackendInstructions(backendPrompt, formatTaskSnapshot(items), this.today()),
      tools: LIVE_TOOLS,
      tool_choice: 'auto',
      parallel_tool_calls: true,
      store: this.store,
      ...(this.store ? {} : { include: ['reasoning.encrypted_content'] }),
      ...(reasoningEffort ? { reasoning: { effort: reasoningEffort } } : {})
    };
    let input = buildSolInput(session.dialog.slice(-DIALOG_TURNS));
    let previous = null;
    let wrote = false;
    for (let pass = 0; pass < MAX_BACKEND_PASSES; pass++) {
      const body = { ...base, input, ...(previous ? { previous_response_id: previous } : {}) };
      const reply = await this.fetchImpl(this.responsesUrl, {
        method: 'POST',
        headers: { Authorization: `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const raw = await reply.text();
      if (!reply.ok) fail('MODEL_UNAVAILABLE', `Sol не ответил (${reply.status})`);
      const payload = JSON.parse(raw);
      const output = payload.output || [];
      this.record({ type: 'vl.backend.response', response_id: payload.id, model: payload.model, pass, output, usage: payload.usage || null }, 'in');
      const text = output.filter(item => item.type === 'message').flatMap(item => (item.content || []).map(part => part.text)).filter(Boolean).join(' ').trim();
      const calls = output.map(readFunctionCall).filter(Boolean);
      if (!calls.length) {
        // After writes the server has spoken for them; only a question still needs the voice.
        if (text && (!wrote || text.endsWith('?'))) report({ failed: false, answer: true, text });
        return;
      }
      if (this.session !== session) return;

      const results = [];
      const confirmations = [];
      const failures = [];
      for (const call of calls) {
        let outcome;
        try { outcome = await this.invoke(call); }
        catch (error) { outcome = { status: 'rejected', reason: safeError(error).message }; }
        if (outcome.status === 'rejected') failures.push(outcome.reason);
        else if (outcome.confirm) confirmations.push(outcome.confirm);
        const { confirm, ...visible } = outcome;
        results.push({ type: 'function_call_output', call_id: call.callId, output: JSON.stringify(visible) });
      }
      if (failures.length) {
        report({ failed: true, text: [...confirmations, `не вышло: ${failures.join('; ')}`].join(', ').replace(/^н/, 'Н') });
        return;
      }
      if (confirmations.length) { wrote = true; report({ failed: false, text: confirmations.join(', ') }); }
      previous = this.store ? payload.id : null;
      input = this.store ? results : [...input, ...output, ...results];
    }
  }

  async invoke(call) {
    if (call.name === 'getFrontier') return { status: 'ok', frontier: await this.services.readFrontier() };
    if (PROMPT_TOOLS.has(call.name)) return this.invokePrompt(call);
    return this.apply(toTaskCommand(call.name, call.arguments));
  }

  async apply(command) {
    const ack = await this.services.applyCommand(
      command, { clientKey: `gpt-live:${this.sessionId}`, seq: ++this.session.seq }, this.session?.utterance || null
    );
    if (ack?.status === 'rejected') return { status: 'rejected', reason: ack.reason || 'Операция отклонена' };
    // Open pages learn of a change only from this signal: without it a task added by voice
    // stayed off the list until the page was reloaded.
    this.services.changed?.();
    const items = await this.services.readItems();
    await this.syncSnapshot(items);
    return { status: 'applied', operation: command.command, target: ack?.newTarget || command.actId, confirm: confirmText(command, items) };
  }

  async invokePrompt(call) {
    const target = call.name.includes('Voice') ? 'voice' : 'backend';
    if (call.name.startsWith('get')) return { status: 'ok', target, prompt: await this.settings.prompt(target) };
    const result = await this.settings.writePrompt(target, { mode: call.arguments.mode, text: call.arguments.text, source: 'model' });
    if (result.changed) await this.applyPromptChange(target, result.prompt, call.arguments);
    return { status: result.changed ? 'applied' : 'unchanged', target, chars: result.prompt.length, confirm: result.changed ? 'Промпт изменён' : 'Промпт уже такой' };
  }

  /** Prompts are fixed at session creation, so a saved edit alone would look ignored in the
   *  conversation that asked for it. The voice layer's instructions are immutable after
   *  startup and can only be appended to; the backend's prompt is read on every request. */
  async applyPromptChange(target, prompt, args) {
    if (target === 'backend') return;
    const note = args?.mode === 'append' ? String(args.text || '').trim() : prompt;
    for (const chunk of chunkDeltaLines(note.split('\n').filter(Boolean))) {
      this.send({ type: 'session.instructions.append', delegation_id: null, content: chunk });
    }
  }

  /** Keeps the voice layer's table current after any change, including one made by hand in
   *  the interface, by diffing the rendered rows rather than the graph's own change list. */
  async syncSnapshot(items = null) {
    if (!this.active) return [];
    const rows = snapshotRows(items || await this.services.readItems());
    const lines = diffSnapshotRows(this.session.rows, rows);
    this.session.rows = rows;
    // Background context, not an instruction: an instruction append can cut the voice off
    // mid-sentence. The backend reads the whole table afresh on every request.
    for (const chunk of chunkDeltaLines(lines)) {
      this.send({ type: 'session.thinking.append', delegation_id: null, content: `Изменения в задачах:\n${chunk}` });
    }
    return lines;
  }

  async stop(reason = 'client') {
    if (!this.session) return false;
    this.flushSpeech();
    this.send({ type: 'session.close' });
    this.record({ type: 'vl.session.stopped', reason }, 'out');
    try { this.session.socket?.close(); } catch { /* already gone */ }
    this.session = null;
    return true;
  }
}
