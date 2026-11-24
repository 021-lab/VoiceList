import { pageUrl } from './grok-pages.js';
/** Delegating a task to the Grok bot.
 *
 *  The bot is woken by its routine's webhook and answers with a POST of its own; both halves
 *  are a thread of messages per task, kept in the object's SQLite beside the document rather
 *  than in it. The bot never edits the tree. Its one effect on it, `done`, goes through the
 *  same command path as every other source, so it is journalled and can be rolled back.
 *  Contract: docs/Grok-Delegation.md. */

export const WEBHOOK_TIMEOUT_MS = 15_000;
export const MAX_TEXT = 8000;
export const THREAD_LIMIT = 30;
export const REPLY_LIMIT_BYTES = 32_000;
/** A message still «sending» this long after it left was lost with the call — the object
 *  restarted mid-call — and may be sent again. */
export const STALE_SENDING_MS = WEBHOOK_TIMEOUT_MS + 15_000;

/** What the person sees when the call did not start the routine. */
export const WEBHOOK_ERRORS = {
  401: 'Ключ вебхука неверный или перевыпущен',
  400: 'Рутина Грока выключена или упёрлась в лимит запусков',
  404: 'Вебхук Грока больше не существует',
  other: 'Грок не ответил на вызов'
};

export const grokConfigured = (env) => Boolean(env?.GROK_WEBHOOK_URL && env?.GROK_WEBHOOK_KEY && env?.GROK_REPLY_TOKEN);

const row = (value) => ({
  id: value.id, taskId: value.task_id, role: value.role, text: value.text, at: value.at,
  replyTo: value.reply_to ?? null, runUuid: value.run_uuid ?? null, state: value.state ?? null
});

export class GrokThreads {
  constructor(storage) {
    this.sql = storage.sql;
    this.sql.exec(`CREATE TABLE IF NOT EXISTS grok_messages (
      id TEXT PRIMARY KEY,
      task_id TEXT NOT NULL,
      role TEXT NOT NULL,
      text TEXT NOT NULL,
      at TEXT NOT NULL,
      reply_to TEXT,
      run_uuid TEXT,
      state TEXT)`);
    this.sql.exec('CREATE INDEX IF NOT EXISTS grok_messages_task ON grok_messages (task_id, at)');
    // Pages hold the document they last loaded and ask for a new one when this differs. A
    // counter alone would restart at zero with the object and could repeat a value a page
    // already holds; the start time makes every run's values its own.
    this.started = Date.now().toString(36);
    this.changes = 0;
  }
  mark() { return `${this.started}.${this.changes}`; }
  touch() { this.changes += 1; }

  get(id) { const found = this.sql.exec('SELECT * FROM grok_messages WHERE id = ?', id).toArray()[0]; return found ? row(found) : null; }
  insert(message) {
    this.sql.exec('INSERT INTO grok_messages (id,task_id,role,text,at,reply_to,run_uuid,state) VALUES (?,?,?,?,?,?,?,?)',
      message.id, message.taskId, message.role, message.text, message.at, message.replyTo ?? null, message.runUuid ?? null, message.state ?? null);
    this.touch();
    return this.get(message.id);
  }
  update(id, fields) {
    if ('state' in fields) this.sql.exec('UPDATE grok_messages SET state = ? WHERE id = ?', fields.state, id);
    if ('runUuid' in fields) this.sql.exec('UPDATE grok_messages SET run_uuid = ? WHERE id = ?', fields.runUuid, id);
    if ('at' in fields) this.sql.exec('UPDATE grok_messages SET at = ? WHERE id = ?', fields.at, id);
    this.touch();
    return this.get(id);
  }
  /** The thread of one task, oldest first. Deleting or archiving the task leaves it alone. */
  thread(taskId) { return this.sql.exec('SELECT * FROM grok_messages WHERE task_id = ? ORDER BY at, rowid', taskId).toArray().map(row); }
  /** Threads of several tasks at once, for the frontier. */
  threads(taskIds) {
    const wanted = [...new Set(taskIds)];
    const byTask = new Map();
    if (!wanted.length) return byTask;
    // SQLite in a Durable Object takes at most 100 bound parameters per statement.
    const rows = [];
    for (let at = 0; at < wanted.length; at += 90) {
      const part = wanted.slice(at, at + 90);
      rows.push(...this.sql.exec(`SELECT * FROM grok_messages WHERE task_id IN (${part.map(() => '?').join(',')}) ORDER BY at, rowid`, ...part).toArray());
    }
    for (const value of rows) {
      if (!byTask.has(value.task_id)) byTask.set(value.task_id, []);
      byTask.get(value.task_id).push(row(value));
    }
    return byTask;
  }
  /** Every row, for moving the document to another object. */
  exportAll() { return this.sql.exec('SELECT * FROM grok_messages ORDER BY at, rowid').toArray().map(row); }
  importAll(messages) { for (const message of messages || []) if (!this.get(message.id)) this.insert(message); }
}

/** What the frontier shows under a task: the last word in its thread. */
export const isStale = (message, now = Date.now()) => message.state === 'sending' && now - Date.parse(message.at) > STALE_SENDING_MS;

export function threadSummary(messages) {
  const spoken = messages.filter(message => message.role !== 'system');
  const last = spoken.at(-1);
  if (!last) return null;
  if (last.role === 'user' && (last.state === 'failed' || isStale(last))) {
    const error = messages.filter(message => message.role === 'system' && message.replyTo === last.id).at(-1);
    return { kind: 'error', text: error?.text || WEBHOOK_ERRORS.other, at: error?.at || last.at };
  }
  if (last.role === 'user') return { kind: 'working', text: 'Грок работает…', at: last.at };
  return { kind: 'grok', text: last.text, at: last.at };
}

/** The task as the bot reads it: where it sits and what is under it. */
export function taskForBot(items, taskId) {
  const byId = new Map(items.map(item => [item.id, item]));
  const task = byId.get(taskId);
  const path = [];
  for (let parent = byId.get(task.parentId); parent; parent = byId.get(parent.parentId)) path.unshift(parent.title);
  const children = items.filter(item => item.parentId === taskId && item.status !== 'Archive')
    .sort((a, b) => (a.order ?? 0) - (b.order ?? 0)).map(item => ({ title: item.title, status: item.status }));
  return { title: task.title, status: task.status, deadline: task.deadline ?? null, path, children };
}

/** The conversation as the bot reads it: no system lines, the last THREAD_LIMIT messages. */
export const threadForBot = (thread, exceptId = null) => thread.filter(item => item.role !== 'system' && item.id !== exceptId).slice(-THREAD_LIMIT)
  .map(item => ({ id: item.id, role: item.role, text: item.text, at: item.at }));

export function delegationBody({ taskId, message, task, thread, replyUrl }) {
  return {
    kind: 'voicelist.delegation',
    taskId,
    message: { id: message.id, text: message.text, at: message.at },
    task,
    thread: threadForBot(thread, message.id),
    reply: { url: replyUrl }
  };
}

/** «Открыть в Гроке»: the bot's chat with the task named in its field. The prefix is what the
 *  bot's profile recognises; the title is cut until the encoded text fits 1000 characters. */
export const DEEP_LINK_TEXT_LIMIT = 1000;
export function grokDeepLink({ botId = '', taskId, title = '' }) {
  const text = (name) => `VoiceList, задача ${taskId} «${name}». `;
  let name = String(title);
  if (encodeURIComponent(text(name)).length > DEEP_LINK_TEXT_LIMIT) {
    const chars = [...name];
    while (chars.length && encodeURIComponent(text(chars.join('') + '…')).length > DEEP_LINK_TEXT_LIMIT) chars.pop();
    name = chars.join('') + '…';
  }
  const query = [...(botId ? [`id=${encodeURIComponent(botId)}`] : []), `text=${encodeURIComponent(text(name))}`].join('&');
  return `grokbot://app/v1/prompt?${query}`;
}

const base64 = (bytes) => { let text = ''; for (const byte of new Uint8Array(bytes)) text += String.fromCharCode(byte); return btoa(text); };
const unbase64 = (text) => Uint8Array.from(atob(text), char => char.charCodeAt(0));

/** Headers that authenticate the call. The routine hands out either a bearer key or a
 *  Standard Webhooks signing secret (`whsec_…`); the prefix says which. */
export async function webhookHeaders(key, body, { id = crypto.randomUUID(), now = Date.now() } = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (!key.startsWith('whsec_')) return { ...headers, Authorization: `Bearer ${key}` };
  const timestamp = String(Math.floor(now / 1000));
  const secret = await crypto.subtle.importKey('raw', unbase64(key.slice('whsec_'.length)), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const signature = await crypto.subtle.sign('HMAC', secret, new TextEncoder().encode(`${id}.${timestamp}.${body}`));
  return { ...headers, 'webhook-id': id, 'webhook-timestamp': timestamp, 'webhook-signature': `v1,${base64(signature)}` };
}

/** One call of the webhook. It answers at once and confirms only that the run started. */
export async function callWebhook({ url, key, body, fetchImpl = fetch, timeoutMs = WEBHOOK_TIMEOUT_MS }) {
  const text = JSON.stringify(body);
  let status = 0, raw = '';
  try {
    const response = await fetchImpl(url, { method: 'POST', headers: await webhookHeaders(key, text), body: text, signal: AbortSignal.timeout(timeoutMs) });
    status = response.status;
    raw = (await response.text()).slice(0, 4000);
  } catch (error) {
    return { ok: false, status: 0, raw: '', error: WEBHOOK_ERRORS.other, detail: error?.name === 'TimeoutError' ? 'timeout' : 'network' };
  }
  let data = null;
  try { data = JSON.parse(raw); } catch { /* not JSON: judged by the status alone */ }
  if (status === 200 && data?.success === true) return { ok: true, status, raw, runUuid: typeof data.runUuid === 'string' ? data.runUuid : null };
  return { ok: false, status, raw, error: WEBHOOK_ERRORS[status] || WEBHOOK_ERRORS.other };
}

/** Equal tokens, compared without leaking where they differ: both sides are hashed first, so
 *  the comparison runs over the same length whatever was sent. */
export async function sameToken(given, expected) {
  if (typeof given !== 'string' || typeof expected !== 'string' || !expected) return false;
  const digest = async (value) => new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value)));
  const [a, b] = await Promise.all([digest(given), digest(expected)]);
  let difference = 0;
  for (let index = 0; index < a.length; index++) difference |= a[index] ^ b[index];
  return difference === 0;
}

export const bearer = (request) => /^Bearer (.+)$/.exec(request.headers.get('Authorization') || '')?.[1] ?? '';

/** The body as text, cut off at the limit: what the bot sent is logged as it came. */
export async function readBoundedText(request, limit = REPLY_LIMIT_BYTES) {
  const reader = request.body?.getReader();
  if (!reader) return { text: '', tooLarge: false };
  const chunks = []; let size = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    size += value.byteLength;
    if (size > limit) { await reader.cancel(); return { text: '', tooLarge: true }; }
    chunks.push(value);
  }
  const bytes = new Uint8Array(size); let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return { text: new TextDecoder().decode(bytes), tooLarge: false };
}

const id = (value) => typeof value === 'string' && value.length > 0 && value.length <= 160;

/** The bot's answer, checked. Returns the reply or the reason it is refused. */
export function parseReply(raw) {
  let body;
  try { body = JSON.parse(raw); } catch { return { error: 'Некорректный JSON' }; }
  if (!body || typeof body !== 'object' || Array.isArray(body)) return { error: 'Тело должно быть объектом' };
  if (!id(body.taskId)) return { error: 'taskId обязателен' };
  if (!id(body.replyId)) return { error: 'replyId обязателен' };
  if (body.replyTo != null && !id(body.replyTo)) return { error: 'replyTo некорректен' };
  if (typeof body.text !== 'string' || !body.text.trim()) return { error: 'text обязателен' };
  if (body.text.length > MAX_TEXT) return { error: `text длиннее ${MAX_TEXT} символов` };
  if (body.done != null && typeof body.done !== 'boolean') return { error: 'done должен быть true или false' };
  return { reply: { taskId: body.taskId, replyId: body.replyId, replyTo: body.replyTo ?? null, text: body.text, done: body.done === true } };
}

/** The person's message, checked. */
export function parseDelegation(body) {
  if (!body || typeof body !== 'object') return { error: 'Некорректный запрос' };
  if (!id(body.taskId)) return { error: 'taskId обязателен' };
  if (!id(body.messageId)) return { error: 'messageId обязателен' };
  const text = typeof body.text === 'string' ? body.text.trim() : '';
  if (!text) return { error: 'Пустое сообщение' };
  if (text.length > MAX_TEXT) return { error: `Сообщение длиннее ${MAX_TEXT} символов` };
  return { delegation: { taskId: body.taskId, messageId: body.messageId, text } };
}

/** Both requests, with everything they touch passed in: the object wires its own storage,
 *  document and log, tests wire a database in memory and a fake webhook. */
export class GrokDelegation {
  constructor({ threads, env, readItems, applyCommand, log = () => {}, changed = () => {}, fetchImpl = (...args) => fetch(...args), now = () => new Date().toISOString() }) {
    Object.assign(this, { threads, env, readItems, applyCommand, log, changed, fetchImpl, now });
  }
  configured() { return grokConfigured(this.env); }

  /** POST /api/grok/delegate. The same messageId never wakes the bot twice once a call was
   *  accepted; a failed one is sent again under its own id, which is what «Повторить» does. */
  async delegate(input, replyUrl) {
    if (!this.configured()) return { status: 503, body: { error: 'Грок не подключён' } };
    const { delegation, error } = parseDelegation(input);
    if (error) return { status: 400, body: { error } };
    const items = await this.readItems();
    if (!items.some(item => item.id === delegation.taskId)) return { status: 404, body: { error: 'Задача не найдена' } };

    let message = this.threads.get(delegation.messageId);
    if (message && (message.role !== 'user' || message.taskId !== delegation.taskId)) return { status: 400, body: { error: 'messageId уже занят' } };
    if (message && message.state !== 'failed' && !isStale(message)) return { status: 200, body: { message, duplicate: true } };
    // A resend is sent now: it moves after the error it answers.
    message = message
      ? this.threads.update(message.id, { state: 'sending', at: this.now() })
      : this.threads.insert({ id: delegation.messageId, taskId: delegation.taskId, role: 'user', text: delegation.text, at: this.now(), state: 'sending' });
    this.changed();

    const body = delegationBody({ taskId: message.taskId, message, task: taskForBot(items, message.taskId), thread: this.threads.thread(message.taskId), replyUrl });
    const result = await callWebhook({ url: this.env.GROK_WEBHOOK_URL, key: this.env.GROK_WEBHOOK_KEY, body, fetchImpl: this.fetchImpl });
    this.log({ type: 'grok.delegate', delegation_id: message.id, response_id: result.runUuid ?? null, task_id: message.taskId,
      webhook: { status: result.status, body: result.raw, ...(result.detail ? { detail: result.detail } : {}) }, request: body }, { sessionId: 'grok:' + message.taskId, direction: 'out' });

    if (result.ok) message = this.threads.update(message.id, { state: 'accepted', runUuid: result.runUuid });
    else {
      message = this.threads.update(message.id, { state: 'failed' });
      this.threads.insert({ id: crypto.randomUUID(), taskId: message.taskId, role: 'system', text: result.error, at: this.now(), replyTo: message.id });
    }
    this.changed();
    return { status: 200, body: { message, webhook: { status: result.status } } };
  }

  /** GET /api/grok/thread: what the webhook would carry, without `message`. The bot reads it
   *  when the conversation goes on in its own app and no call brought the context. */
  async readThread({ token, taskId, replyUrl }) {
    if (!await sameToken(token, this.env.GROK_REPLY_TOKEN)) return { status: 401, body: { error: 'Неверный токен' } };
    if (!id(taskId)) return { status: 400, body: { error: 'taskId обязателен' } };
    const items = await this.readItems();
    if (!items.some(item => item.id === taskId)) return { status: 404, body: { error: 'Задача не найдена' } };
    return { status: 200, body: { taskId, task: taskForBot(items, taskId), thread: threadForBot(this.threads.thread(taskId)), reply: { url: replyUrl } } };
  }

  /** POST /api/grok/reply. `raw` is the body as it arrived; `token` the bearer it carried. */
  async reply({ token, raw, tooLarge = false }) {
    if (!await sameToken(token, this.env.GROK_REPLY_TOKEN)) return { status: 401, body: { error: 'Неверный токен' } };
    if (tooLarge) return { status: 400, body: { error: 'Тело больше 32 КБ' } };
    const { reply, error } = parseReply(raw);
    const logged = (status, extra = {}) => this.log({ type: 'grok.reply', delegation_id: reply?.replyTo ?? null, response_id: reply?.replyId ?? null,
      task_id: reply?.taskId ?? null, status, raw, ...extra }, { sessionId: 'grok:' + (reply?.taskId || 'unknown'), direction: 'in' });
    if (error) { logged(400); return { status: 400, body: { error } }; }

    const items = await this.readItems();
    const task = items.find(item => item.id === reply.taskId);
    const existing = this.threads.get(reply.replyId);
    if (existing && (existing.role !== 'grok' || existing.taskId !== reply.taskId)) { logged(400); return { status: 400, body: { error: 'replyId уже занят' } }; }
    if (!task) { logged(404); return { status: 404, body: { error: 'Задача не найдена' } }; }
    const asked = !existing && reply.replyTo ? this.threads.get(reply.replyTo) : null;
    if (!existing && reply.replyTo && (!asked || asked.role !== 'user' || asked.taskId !== reply.taskId)) { logged(400); return { status: 400, body: { error: 'replyTo не из треда этой задачи' } }; }
    // The answer is proof the call got through, whatever the webhook said in time: a message
    // marked failed by a slow webhook must not offer to wake the bot again.
    if (asked && asked.state !== 'accepted') this.threads.update(asked.id, { state: 'accepted' });
    // Nothing is awaited between the check above and this write: a repeat arriving later is a
    // duplicate.
    if (!existing) this.threads.insert({ id: reply.replyId, taskId: reply.taskId, role: 'grok', text: reply.text, at: this.now(), replyTo: reply.replyTo });
    // A repeat still closes the task if the first attempt did not get that far. The journal key
    // is the replyId, so a close that did happen — or was rolled back since — is not redone.
    const closed = reply.done && !['Done', 'Archive'].includes(task.status) ? await this.close(task.id, reply.replyId) : null;
    logged(200, { ...(existing ? { duplicate: true } : {}), ...(closed ? { done: closed } : {}) });
    this.changed();
    return { status: 200, body: existing ? { stored: true, duplicate: true } : { stored: true } };
  }
  async close(taskId, replyId) {
    try {
      const ack = await this.applyCommand({ command: 'setStatus', actId: taskId, actType: 'task', payload: { status: 'Done' }, source: 'grok' },
        { clientKey: 'grok:' + replyId, seq: 1 });
      return ack?.status === 'applied' ? 'applied' : `rejected: ${ack?.reason || 'unknown'}`;
    } catch (error) { return `failed: ${error?.message || error}`; }
  }
}

/** The document with the threads put where the page draws them: the whole thread on the task
 *  card, the last word under each frontier task. */
export function withGrok(document, threads, configured, { pages = null, botId = '' } = {}) {
  const body = document.root?.children?.[1];
  if (body?.type === 'task-editor' && body.props.taskId) body.props.grok = {
    // Staleness is judged by the server's clock, not the phone's.
    configured, messages: threads.thread(body.props.taskId).map(message => (isStale(message) ? { ...message, stale: true } : message)),
    pages: pages ? pages.forTask(body.props.taskId).map(page => ({ ...page, url: pageUrl(pages.env, page.id) })) : [],
    link: configured ? grokDeepLink({ botId, taskId: body.props.taskId, title: body.props.title }) : null
  };
  if (body?.type === 'task-list' && document.view === 'frontier') {
    const byTask = threads.threads((body.children || []).map(child => child.props?.taskId).filter(Boolean));
    for (const child of body.children || []) {
      const summary = threadSummary(byTask.get(child.props?.taskId) || []);
      if (summary) child.props.grok = summary;
    }
  }
  return { ...document, grok: threads.mark() };
}
