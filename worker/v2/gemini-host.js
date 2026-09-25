/** Gemini Live, in the browser-connects-directly shape.
 *
 *  The audio path does not pass through the worker: the browser opens the socket to Google
 *  itself. What stays here is everything that must not be the browser's to decide — the API
 *  key, the session's instructions and tool list (both sealed into the ephemeral token), the
 *  execution of every tool call, and the log.
 *
 *  A tool call reaches this host over the ordinary document channel, so a change made by
 *  voice travels the same path as one made by hand and lands in the same journal. */
import {
  GEMINI_MODEL, GEMINI_MODELS_URL, GEMINI_TOKENS_URL, buildTokenRequest, readToolCalls, isLoggableFrame
} from '../../src/v2/domain/gemini-live.js';
import { toTaskCommand } from '../../src/v2/domain/live-session.js';
import { fail, safeError } from '../../src/v2/domain/contracts.js';

/** A client can report whatever it likes; the log takes frames in bounded batches so a
 *  chatty or hostile page cannot fill the database in one request. */
const MAX_FRAMES_PER_BATCH = 200;

export class GeminiHost {
  constructor({ log, settings, services, apiKey, fetchImpl = (...args) => fetch(...args), tokensUrl = GEMINI_TOKENS_URL, modelsUrl = GEMINI_MODELS_URL, now = () => new Date() }) {
    this.log = log;
    this.settings = settings;
    this.services = services;
    this.apiKey = apiKey;
    this.fetchImpl = fetchImpl;
    this.tokensUrl = tokensUrl;
    this.modelsUrl = modelsUrl;
    this.now = now;
    this.session = null;
    // The sequence outlives a session: a browser that reconnects mid-conversation must not
    // reuse a key the journal already settled.
    this.seq = 0;
  }

  get active() { return Boolean(this.session); }
  get sessionId() { return this.session?.id || ''; }

  stamp() { return this.now().toISOString(); }

  record(event, direction = 'in', sessionId = this.sessionId) {
    try { this.log.append(event, { sessionId, direction, at: this.stamp() }); } catch { /* the log must never break the call */ }
  }

  /** Asks Google whether the key works, by the cheapest read there is.
   *
   *  Storing an unverified key is how a one-time setup becomes a one-way door: a typo gets
   *  saved, the app reports "configured", and nothing short of redeploying the object can
   *  take it back. So a key proves itself before it is kept, and a stored key that stops
   *  proving itself can be replaced. */
  async verifyKey(apiKey) {
    try {
      const reply = await this.fetchImpl(`${this.modelsUrl}?pageSize=1`, {
        headers: { 'x-goog-api-key': apiKey }
      });
      if (reply.ok) return { ok: true };
      const detail = (await reply.text()).slice(0, 300);
      return { ok: false, status: reply.status, detail };
    } catch (error) {
      return { ok: false, detail: safeError(error).message };
    }
  }

  /** Mints a token that already carries the model, the prompt and the tools. The browser
   *  receives a credential it cannot widen: it may open this session and no other. */
  async mintToken() {
    if (!this.apiKey) fail('MODEL_UNAVAILABLE', 'Ключ Gemini не настроен');
    const [prompt, model, items] = await Promise.all([
      this.settings.prompt('gemini'), this.settings.geminiModel(), this.services.readItems()
    ]);
    const request = buildTokenRequest({ items, prompt, model: model || GEMINI_MODEL, now: this.now });

    const sessionId = `gm_${this.now().getTime().toString(36)}`;
    this.session = { id: sessionId, startedAt: this.stamp() };
    this.record({ type: 'gm.token.requested', model: model || GEMINI_MODEL, tasks: items.length }, 'out', sessionId);

    let reply;
    try {
      reply = await this.fetchImpl(this.tokensUrl, {
        method: 'POST',
        headers: { 'x-goog-api-key': this.apiKey, 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });
    } catch (error) {
      this.record({ type: 'gm.token.unreachable', error: safeError(error) }, 'in', sessionId);
      fail('MODEL_UNAVAILABLE', 'Не удалось получить токен Gemini');
    }

    const text = await reply.text();
    if (!reply.ok) {
      this.record({ type: 'gm.token.failed', status: reply.status, detail: text.slice(0, 1_000) }, 'in', sessionId);
      fail('MODEL_UNAVAILABLE', `Google отклонил запрос токена (${reply.status})`);
    }
    const payload = JSON.parse(text);
    // The token value is the resource name; nothing else in the reply is a credential.
    const token = payload?.name;
    if (!token) {
      this.record({ type: 'gm.token.failed', status: reply.status, detail: 'нет поля name' }, 'in', sessionId);
      fail('MODEL_UNAVAILABLE', 'Google вернул ответ без токена');
    }
    this.record({
      type: 'gm.session.started', model: model || GEMINI_MODEL,
      expireTime: request.expireTime, newSessionExpireTime: request.newSessionExpireTime,
      promptChars: request.liveConnectConstraints.config.systemInstruction.parts[0].text.length,
      tools: request.liveConnectConstraints.config.tools[0].functionDeclarations.map(tool => tool.name)
    }, 'in', sessionId);

    return { sessionId, token, model: model || GEMINI_MODEL, expireTime: request.expireTime };
  }

  /** Executes one tool call and returns what the page must send back as a functionResponse.
   *  The page relays; it does not decide. */
  async invoke(call, sessionId = this.sessionId) {
    if (!call?.name) fail('INVALID_INPUT', 'Пустой вызов инструмента');
    this.record({ type: 'gm.tool.call', name: call.name, arguments: call.arguments || {}, callId: call.id || '' }, 'in', sessionId);

    let response;
    try {
      if (call.name === 'getFrontier') {
        response = { status: 'ok', frontier: await this.services.readFrontier() };
      } else {
        const command = toTaskCommand(call.name, call.arguments || {}, 'gemini-live');
        const ack = await this.services.applyCommand(command, {
          clientKey: `gemini:${sessionId || 'session'}`, seq: (this.seq += 1)
        });
        response = ack?.status === 'rejected'
          ? { status: 'rejected', reason: ack.reason || 'Операция отклонена' }
          : { status: 'applied', operation: command.command, target: ack?.newTarget || command.actId };
      }
    } catch (error) {
      response = { status: 'rejected', reason: safeError(error).message };
    }

    this.record({ type: 'gm.tool.result', name: call.name, callId: call.id || '', response }, 'out', sessionId);
    return { id: call.id || '', name: call.name, response };
  }

  async invokeAll(frame, sessionId = this.sessionId) {
    const calls = readToolCalls(frame);
    const results = [];
    for (const call of calls) results.push(await this.invoke(call, sessionId));
    return results;
  }

  /** Frames the page saw. They are a report, not a source of truth: the page could omit or
   *  invent one, and only what passed through invoke() actually changed a task. */
  mirror(frames, sessionId = this.sessionId) {
    const batch = Array.isArray(frames) ? frames.slice(0, MAX_FRAMES_PER_BATCH) : [];
    let kept = 0;
    for (const entry of batch) {
      const frame = entry?.frame ?? entry;
      if (!isLoggableFrame(frame)) continue;
      this.record({ type: 'gm.frame', direction: entry?.direction || 'in', frame }, entry?.direction === 'out' ? 'out' : 'in', sessionId);
      kept += 1;
    }
    return { received: batch.length, kept };
  }

  stop(reason = 'client') {
    if (!this.session) return false;
    this.record({ type: 'gm.session.stopped', reason }, 'out');
    this.session = null;
    return true;
  }
}
