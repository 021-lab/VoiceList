import { describe, expect, it, vi } from 'vitest';
import { DatabaseSync } from 'node:sqlite';
import { createHmac } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { DocumentRuntime } from '../../src/v2/domain/document-runtime.js';
import { CompatibilityPort } from '../../worker/v2/compatibility.js';
import { GrokThreads, GrokDelegation, webhookHeaders, withGrok, threadSummary, grokDeepLink, WEBHOOK_ERRORS } from '../../worker/v2/grok.js';

/** The object's SQLite, played by SQLite. */
function storage() {
  const db = new DatabaseSync(':memory:');
  return { sql: { exec(query, ...params) {
    const prepared = db.prepare(query);
    if (/^\s*select/i.test(query)) { const rows = prepared.all(...params); return { toArray: () => rows }; }
    prepared.run(...params); return { toArray: () => [] };
  } } };
}

const env = { GROK_WEBHOOK_URL: 'https://grok.example/hook', GROK_WEBHOOK_KEY: 'hook-key', GROK_REPLY_TOKEN: 'reply-token' };
const REPLY_URL = 'https://vl.example/api/grok/reply';

function setup({ webhook = () => new Response(JSON.stringify({ success: true, runUuid: 'run-1' }), { status: 200 }), environment = env, applyCommand = null } = {}) {
  const runtime = new DocumentRuntime();
  const port = new CompatibilityPort(runtime);
  const threads = new GrokThreads(storage());
  const fetchImpl = vi.fn(async (url, options) => webhook(url, options));
  const log = vi.fn();
  const changed = vi.fn();
  const grok = new GrokDelegation({
    threads, env: environment, fetchImpl, log, changed,
    readItems: async () => runtime.graph.read().items,
    applyCommand: applyCommand ? applyCommand(port) : (command, message) => port.applyCommand(command, { message })
  });
  return { runtime, port, threads, grok, fetchImpl, log, changed };
}
const delegate = (grok, extra = {}) => grok.delegate({ taskId: 'milk1', messageId: 'm1', text: 'Найди три варианта', ...extra }, REPLY_URL);
const reply = (grok, body, token = 'reply-token') => grok.reply({ token, raw: JSON.stringify({ taskId: 'milk1', replyTo: null, replyId: 'r1', text: 'Нашёл', ...body }) });

describe('POST /api/grok/delegate', () => {
  it('sends the task, the thread and where to answer; accepted keeps runUuid', async () => {
    const { grok, threads, fetchImpl, log } = setup();
    const result = await delegate(grok);
    expect(result.status).toBe(200);
    expect(result.body.message).toMatchObject({ id: 'm1', role: 'user', state: 'accepted', runUuid: 'run-1' });
    const [url, options] = fetchImpl.mock.calls[0];
    expect(url).toBe(env.GROK_WEBHOOK_URL);
    expect(options.headers.Authorization).toBe('Bearer hook-key');
    const body = JSON.parse(options.body);
    expect(body).toMatchObject({ kind: 'voicelist.delegation', taskId: 'milk1', message: { id: 'm1', text: 'Найди три варианта' }, reply: { url: REPLY_URL }, thread: [] });
    expect(body.task).toMatchObject({ title: expect.any(String), status: 'Open', deadline: null, path: expect.any(Array), children: expect.any(Array) });
    expect(threads.thread('milk1')).toHaveLength(1);
    // Logged beside the message, with the webhook's answer, and without the key.
    const [event] = log.mock.calls[0];
    expect(event).toMatchObject({ type: 'grok.delegate', delegation_id: 'm1', response_id: 'run-1', webhook: { status: 200 } });
    expect(JSON.stringify(log.mock.calls)).not.toContain('hook-key');
  });

  it.each([
    [401, WEBHOOK_ERRORS[401]], [400, WEBHOOK_ERRORS[400]], [404, WEBHOOK_ERRORS[404]], [500, WEBHOOK_ERRORS.other], [503, WEBHOOK_ERRORS.other]
  ])('webhook %i → failed with «%s»', async (status, text) => {
    const { grok, threads } = setup({ webhook: () => new Response('{}', { status }) });
    const result = await delegate(grok);
    expect(result.body.message.state).toBe('failed');
    const system = threads.thread('milk1').find(message => message.role === 'system');
    expect(system).toMatchObject({ text, replyTo: 'm1' });
  });

  it('network failure and timeout → «Грок не ответил на вызов»', async () => {
    for (const failure of [new TypeError('fetch failed'), Object.assign(new Error('t'), { name: 'TimeoutError' })]) {
      const { grok, threads } = setup({ webhook: () => { throw failure; } });
      expect((await delegate(grok)).body.message.state).toBe('failed');
      expect(threads.thread('milk1').at(-1).text).toBe(WEBHOOK_ERRORS.other);
    }
  });

  it('the same messageId after acceptance neither stores a second message nor wakes the bot', async () => {
    const { grok, threads, fetchImpl } = setup();
    await delegate(grok);
    const again = await delegate(grok);
    expect(again.body.duplicate).toBe(true);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(threads.thread('milk1')).toHaveLength(1);
  });

  it('a repeat while the first call is still out does not call twice', async () => {
    let release;
    const { grok, fetchImpl } = setup({ webhook: () => new Promise(resolve => { release = () => resolve(new Response('{"success":true,"runUuid":"r"}')); }) });
    const first = delegate(grok);
    await vi.waitFor(() => expect(fetchImpl).toHaveBeenCalledTimes(1));
    expect((await delegate(grok)).body.duplicate).toBe(true);
    release(); await first;
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it('a failed message is sent again under the same id («Повторить»)', async () => {
    let status = 500;
    const { grok, threads, fetchImpl } = setup({ webhook: () => new Response(status === 200 ? '{"success":true,"runUuid":"run-2"}' : '{}', { status }) });
    await delegate(grok);
    status = 200;
    const retried = await delegate(grok);
    expect(retried.body.message).toMatchObject({ id: 'm1', state: 'accepted', runUuid: 'run-2' });
    expect(fetchImpl).toHaveBeenCalledTimes(2);
    expect(threads.thread('milk1').filter(message => message.role === 'user')).toHaveLength(1);
  });

  it('a message left «sending» by a call that never came back may be sent again', async () => {
    const { grok, threads, fetchImpl } = setup();
    threads.insert({ id: 'm1', taskId: 'milk1', role: 'user', text: 'Найди три варианта', at: new Date(Date.now() - 60_000).toISOString(), state: 'sending' });
    expect(threadSummary(threads.thread('milk1'))).toMatchObject({ kind: 'error' });
    expect((await delegate(grok)).body.message.state).toBe('accepted');
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it('the thread goes along without the current message and without system lines', async () => {
    const { grok, fetchImpl } = setup();
    await delegate(grok);
    await reply(grok, {});
    await delegate(grok, { messageId: 'm2', text: 'А дешевле?' });
    const body = JSON.parse(fetchImpl.mock.calls[1][1].body);
    expect(body.thread.map(message => [message.id, message.role])).toEqual([['m1', 'user'], ['r1', 'grok']]);
  });

  it('refuses an empty text, an unknown task, and a missing setup', async () => {
    expect((await delegate(setup().grok, { text: '  ' })).status).toBe(400);
    expect((await delegate(setup().grok, { taskId: 'nope' })).status).toBe(404);
    const unconfigured = setup({ environment: {} });
    expect((await delegate(unconfigured.grok)).status).toBe(503);
    expect(unconfigured.fetchImpl).not.toHaveBeenCalled();
  });

  it('a whsec_ key signs the call by Standard Webhooks instead of sending a bearer', async () => {
    const secret = Buffer.from('0123456789abcdef0123456789abcdef').toString('base64');
    const body = '{"a":1}';
    const headers = await webhookHeaders('whsec_' + secret, body, { id: 'msg_1', now: 1_700_000_000_000 });
    expect(headers.Authorization).toBeUndefined();
    const expected = createHmac('sha256', Buffer.from(secret, 'base64')).update(`msg_1.1700000000.${body}`).digest('base64');
    expect(headers).toMatchObject({ 'webhook-id': 'msg_1', 'webhook-timestamp': '1700000000', 'webhook-signature': `v1,${expected}` });
  });
});

describe('POST /api/grok/reply', () => {
  it('a wrong token is 401 and stores nothing', async () => {
    const { grok, threads, log } = setup();
    expect((await reply(grok, {}, 'wrong')).status).toBe(401);
    expect((await reply(grok, {}, '')).status).toBe(401);
    expect(threads.thread('milk1')).toHaveLength(0);
    expect(log).not.toHaveBeenCalled();
  });

  it('an unknown task is 404, a malformed body is 400', async () => {
    const { grok } = setup();
    expect((await reply(grok, { taskId: 'nope' })).status).toBe(404);
    expect((await grok.reply({ token: 'reply-token', raw: '{' })).status).toBe(400);
    expect((await reply(grok, { text: '' })).status).toBe(400);
    expect((await reply(grok, { text: 'x'.repeat(8001) })).status).toBe(400);
    expect((await reply(grok, { done: 'yes' })).status).toBe(400);
    expect((await grok.reply({ token: 'reply-token', raw: '', tooLarge: true })).status).toBe(400);
  });

  it('stores the reply in the thread and logs the raw body', async () => {
    const { grok, threads, log, changed } = setup();
    await delegate(grok);
    const result = await reply(grok, { replyTo: 'm1' });
    expect(result).toEqual({ status: 200, body: { stored: true } });
    expect(threads.get('r1')).toMatchObject({ role: 'grok', text: 'Нашёл', replyTo: 'm1', taskId: 'milk1' });
    const event = log.mock.calls.at(-1)[0];
    expect(event).toMatchObject({ type: 'grok.reply', delegation_id: 'm1', response_id: 'r1', status: 200 });
    expect(JSON.parse(event.raw).text).toBe('Нашёл');
    expect(JSON.stringify(event)).not.toContain('reply-token');
    expect(changed).toHaveBeenCalled();
  });

  it('a repeated replyId is a duplicate: no second row, no second done', async () => {
    const { grok, threads, runtime } = setup();
    await reply(grok, { done: true });
    const actions = runtime.journal.actions().length;
    expect(await reply(grok, { done: true })).toEqual({ status: 200, body: { stored: true, duplicate: true } });
    expect(threads.thread('milk1').filter(message => message.role === 'grok')).toHaveLength(1);
    expect(runtime.journal.actions()).toHaveLength(actions);
  });

  it('done on an open task closes it through the journal, as Grok, and can be rolled back', async () => {
    const { grok, runtime, port } = setup();
    await reply(grok, { done: true });
    expect(runtime.graph.read({ id: 'milk1' }).status).toBe('Done');
    const action = runtime.journal.actions().at(-1);
    expect(action).toMatchObject({ source: 'grok', canRollback: true });
    await port.applyCommand({ command: 'rollbackAction', actId: action.id, actType: 'action', payload: { actionId: action.id } }, { message: { clientKey: 'test-ui', seq: 1 } });
    expect(runtime.graph.read({ id: 'milk1' }).status).toBe('Open');
  });

  it('a repeat of a reply whose close failed closes the task; a repeat after a rollback does not reopen it', async () => {
    let broken = true;
    const { grok, runtime, port } = setup({ applyCommand: (port) => async (command, message) => {
      if (broken) throw new Error('runtime down');
      return port.applyCommand(command, { message });
    } });
    expect((await reply(grok, { done: true })).status).toBe(200);
    expect(runtime.graph.read({ id: 'milk1' }).status).toBe('Open');
    broken = false;
    expect((await reply(grok, { done: true })).body).toEqual({ stored: true, duplicate: true });
    expect(runtime.graph.read({ id: 'milk1' }).status).toBe('Done');
    const action = runtime.journal.actions().at(-1);
    await port.applyCommand({ command: 'rollbackAction', actId: action.id, actType: 'action', payload: { actionId: action.id } }, { message: { clientKey: 'test-ui', seq: 1 } });
    await reply(grok, { done: true });
    expect(runtime.graph.read({ id: 'milk1' }).status).toBe('Open');
  });

  it('a reply to a message a slow webhook marked failed makes it accepted: no «Повторить» beside the answer', async () => {
    const { grok, threads } = setup({ webhook: () => new Response('{}', { status: 504 }) });
    await delegate(grok);
    expect(threads.get('m1').state).toBe('failed');
    await reply(grok, { replyTo: 'm1' });
    expect(threads.get('m1').state).toBe('accepted');
    expect(threadSummary(threads.thread('milk1'))).toMatchObject({ kind: 'grok' });
  });

  it('replyTo must be a message of the same task', async () => {
    const { grok } = setup();
    await delegate(grok, { taskId: 'cofee', messageId: 'other' });
    expect((await reply(grok, { replyTo: 'other' })).status).toBe(400);
    expect((await reply(grok, { replyTo: 'nope' })).status).toBe(400);
  });

  it('done on a task already Done or archived leaves the status alone', async () => {
    for (const status of ['Done', 'Archive']) {
      const { grok, runtime, port } = setup();
      await port.applyCommand({ command: 'setStatus', actId: 'milk1', actType: 'task', payload: { status } }, { message: { clientKey: 'test-ui', seq: 1 } });
      const before = runtime.journal.entries.length;
      expect((await reply(grok, { done: true })).status).toBe(200);
      expect(runtime.graph.read({ id: 'milk1' }).status).toBe(status);
      expect(runtime.journal.entries.length).toBe(before);
    }
  });

  it('done: false or absent changes nothing in the tree', async () => {
    const { grok, runtime } = setup();
    const before = runtime.journal.entries.length;
    await reply(grok, { done: false });
    await reply(grok, { replyId: 'r2', done: undefined });
    expect(runtime.graph.read({ id: 'milk1' }).status).toBe('Open');
    expect(runtime.journal.entries.length).toBe(before);
  });
});

describe('the thread in the document', () => {
  it('threads move with the document to another object', async () => {
    const { grok, threads } = setup();
    await delegate(grok); await reply(grok, {});
    const target = new GrokThreads(storage());
    target.importAll(threads.exportAll());
    expect(target.thread('milk1')).toEqual(threads.thread('milk1'));
  });

  it('the task card carries the thread; the frontier carries the last word', async () => {
    const { grok, threads, runtime } = setup();
    // goldn is on the frontier (a leaf of the focused «Яблоки») and sits under it.
    await delegate(grok, { taskId: 'goldn' });
    const card = withGrok(runtime.getDocument({ view: 'edit', taskId: 'goldn' }), threads, true);
    expect(card.root.children[1].props.grok).toMatchObject({ configured: true, messages: [{ id: 'm1', state: 'accepted' }] });
    const row = (document) => document.root.children[1].children.find(node => node.props.taskId === 'goldn');
    const working = withGrok(runtime.getDocument({ view: 'frontier' }), threads, true);
    expect(row(working).props.grok).toMatchObject({ kind: 'working' });
    expect(working.root.children[1].children.filter(node => node.props.grok)).toHaveLength(1);
    const mark = working.grok;
    await reply(grok, { taskId: 'goldn' });
    const answered = withGrok(runtime.getDocument({ view: 'frontier' }), threads, true);
    expect(row(answered).props.grok).toMatchObject({ kind: 'grok', text: 'Нашёл' });
    expect(answered.grok).not.toBe(mark);
  });

  it('the bot sees where the task sits and what is under it', async () => {
    const { grok, fetchImpl } = setup();
    await delegate(grok, { taskId: 'fudji' });
    expect(JSON.parse(fetchImpl.mock.calls[0][1].body).task).toEqual({ title: 'Фуджи', status: 'Pause', deadline: null, path: ['Яблоки'], children: [{ title: 'Позззд', status: 'Done' }] });
  });

  it('the frontier reads threads of more than a hundred tasks (SQLite parameter limit)', async () => {
    const threads = new GrokThreads(storage());
    const ids = Array.from({ length: 250 }, (_, index) => 't' + index);
    for (const id of ids) threads.insert({ id: 'm-' + id, taskId: id, role: 'user', text: 'x', at: '2026-11-24T10:00:00Z', state: 'accepted' });
    const byTask = threads.threads(ids);
    expect(byTask.size).toBe(250);
  });

  it('the card marks a lost call stale by the server\'s clock', () => {
    const { threads, runtime } = setup();
    threads.insert({ id: 'm1', taskId: 'milk1', role: 'user', text: 'x', at: new Date(Date.now() - 60_000).toISOString(), state: 'sending' });
    threads.insert({ id: 'm2', taskId: 'milk1', role: 'user', text: 'y', at: new Date().toISOString(), state: 'sending' });
    const messages = withGrok(runtime.getDocument({ view: 'edit', taskId: 'milk1' }), threads, true).root.children[1].props.grok.messages;
    expect(messages.map(message => Boolean(message.stale))).toEqual([true, false]);
  });

  it('a failed call shows its error as the last word', () => {
    expect(threadSummary([
      { id: 'm1', role: 'user', state: 'failed', at: '1' },
      { id: 's1', role: 'system', text: WEBHOOK_ERRORS[404], replyTo: 'm1', at: '2' }
    ])).toMatchObject({ kind: 'error', text: WEBHOOK_ERRORS[404] });
    expect(threadSummary([])).toBeNull();
  });
});

describe('the routine doc and the code agree', () => {
  const doc = readFileSync('docs/Grok-Bot-Routine.md', 'utf8');
  it('the reply body the routine tells the bot to send is the one /api/grok/reply accepts', async () => {
    const body = doc.match(/Тело:\s*\{([\s\S]*?)\n\s*\}/)[1];
    const fields = [...body.matchAll(/"(\w+)":/g)].map(match => match[1]);
    expect(fields).toEqual(['taskId', 'replyTo', 'replyId', 'text', 'done']);
    const { grok } = setup();
    await delegate(grok);
    const sent = { taskId: 'milk1', replyTo: 'm1', replyId: 'r-doc', text: 'x'.repeat(8000), done: false };
    expect(Object.keys(sent)).toEqual(fields);
    expect((await grok.reply({ token: 'reply-token', raw: JSON.stringify(sent) })).status).toBe(200);
    expect(await grok.reply({ token: 'reply-token', raw: JSON.stringify(sent) })).toEqual({ status: 200, body: { stored: true, duplicate: true } });
    expect(doc).toContain('до 8000 символов');
    expect(doc).toContain('Authorization: Bearer');
  });
  it('the page tool and the chat profile name the endpoints the worker serves', () => {
    expect(doc).toContain('$HOST/api/grok/pages?taskId=$TASK_ID&title=$ENC_TITLE');
    expect(doc).toContain('-X PUT "$HOST/api/grok/pages/$ID"');
    expect(doc).toContain('-X DELETE "$HOST/api/grok/pages/$ID"');
    expect(doc).toContain('{"id": "...", "url": "https://vlist-pages.smileme.ai/p/..."}');
    expect(doc).toContain('401 — токен, 404 — нет задачи или страницы, 413 — больше 10 МБ');
    expect(doc).toContain('/api/grok/thread?taskId=<id>');
    expect(doc).toContain('{taskId, task, thread, reply}');
    // The profile's opening words are the ones the deep link puts in the bot's field.
    expect(doc).toContain('начинается с «VoiceList, задача <id>»');
    expect(new URL(grokDeepLink({ taskId: 't1', title: 'x' })).searchParams.get('text').startsWith('VoiceList, задача t1')).toBe(true);
  });

  it('the call the routine reads carries every field the routine names', async () => {
    const { grok, fetchImpl } = setup();
    await delegate(grok);
    const call = JSON.parse(fetchImpl.mock.calls[0][1].body);
    for (const field of ['kind', 'taskId', 'message', 'task', 'thread', 'reply']) {
      expect(doc).toContain(`- ${field}`);
      expect(call).toHaveProperty(field);
    }
    expect(call.kind).toBe('voicelist.delegation');
    expect(Object.keys(call.task)).toEqual(['title', 'status', 'deadline', 'path', 'children']);
    expect(call.reply).toHaveProperty('url');
  });
});

describe('GET /api/grok/thread', () => {
  it('gives the bot the task, the thread and where to answer — the webhook body without message', async () => {
    const { grok, fetchImpl } = setup();
    await delegate(grok); await reply(grok, { replyTo: 'm1' });
    const result = await grok.readThread({ token: 'reply-token', taskId: 'milk1', replyUrl: REPLY_URL });
    expect(result.status).toBe(200);
    const call = JSON.parse(fetchImpl.mock.calls[0][1].body);
    expect(Object.keys(result.body)).toEqual(['taskId', 'task', 'thread', 'reply']);
    expect(result.body.task).toEqual(call.task);
    expect(result.body.reply).toEqual({ url: REPLY_URL });
    expect(result.body.thread.map(message => [message.id, message.role])).toEqual([['m1', 'user'], ['r1', 'grok']]);
  });
  it('a wrong token is 401, an unknown task 404, no taskId 400', async () => {
    const { grok } = setup();
    expect((await grok.readThread({ token: 'wrong', taskId: 'milk1', replyUrl: REPLY_URL })).status).toBe(401);
    expect((await grok.readThread({ token: 'reply-token', taskId: 'nope', replyUrl: REPLY_URL })).status).toBe(404);
    expect((await grok.readThread({ token: 'reply-token', taskId: '', replyUrl: REPLY_URL })).status).toBe(400);
  });
  it('a reply without replyTo — written from the bot\'s own app — lands in the thread', async () => {
    const { grok, threads } = setup();
    expect((await reply(grok, { replyTo: undefined, replyId: 'chat-1' })).status).toBe(200);
    expect(threads.get('chat-1')).toMatchObject({ role: 'grok', replyTo: null });
  });
});

describe('«Открыть в Гроке»', () => {
  const textOf = (link) => decodeURIComponent(new URL(link).searchParams.get('text'));
  it('names the task in the bot\'s field and the bot by its id', () => {
    const link = grokDeepLink({ botId: 'bot 42', taskId: 't42', title: 'Купить кондиционер & пульт' });
    expect(link.startsWith('grokbot://app/v1/prompt?id=bot%2042&text=')).toBe(true);
    expect(new URL(link).searchParams.get('text')).toBe('VoiceList, задача t42 «Купить кондиционер & пульт». ');
  });
  it('without a bot id opens the main bot', () => {
    const link = grokDeepLink({ botId: '', taskId: 't42', title: 'x' });
    expect(link).toBe(`grokbot://app/v1/prompt?text=${encodeURIComponent('VoiceList, задача t42 «x». ')}`);
  });
  it('cuts the title so the encoded text fits 1000 characters', () => {
    const link = grokDeepLink({ botId: 'b', taskId: 't42', title: 'Очень длинное название '.repeat(40) });
    const encoded = link.split('text=')[1];
    expect(encoded.length).toBeLessThanOrEqual(1000);
    expect(encoded.length).toBeGreaterThan(980);
    const text = new URL(link).searchParams.get('text');
    expect(text.startsWith('VoiceList, задача t42 «Очень длинное')).toBe(true);
    expect(text.endsWith('…». ')).toBe(true);
  });
  it('the card carries the link only when Grok is set up', async () => {
    const { threads, runtime } = setup();
    const on = withGrok(runtime.getDocument({ view: 'edit', taskId: 'milk1' }), threads, true, { botId: 'b' });
    expect(on.root.children[1].props.grok.link).toBe(grokDeepLink({ botId: 'b', taskId: 'milk1', title: 'Молоко 3.2%' }));
    expect(withGrok(runtime.getDocument({ view: 'edit', taskId: 'milk1' }), threads, false).root.children[1].props.grok.link).toBeNull();
  });
});
