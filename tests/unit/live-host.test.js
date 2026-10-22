import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LiveHost } from '../../worker/v2/live-host.js';
import { LiveSettings } from '../../src/v2/domain/live-settings.js';
import { DEFAULT_VOICE_PROMPT } from '../../src/v2/domain/live-session.js';

class FakeSocket {
  constructor() { this.sent = []; this.listeners = {}; this.accepted = false; this.closed = false; }
  accept() { this.accepted = true; }
  send(raw) { this.sent.push(JSON.parse(raw)); }
  close() { this.closed = true; }
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
  emit(type, event) { for (const handler of this.listeners[type] || []) handler(event); }
  deliver(event) { this.emit('message', { data: JSON.stringify(event) }); }
}

const memory = () => { const map = new Map(); return { map, async get(k) { return map.get(k); }, async put(k, v) { map.set(k, v); } }; };

const task = (id, parentId, status, title) => ({ id, parentId, order: 10, status, title, collapsed: false, tags: [] });

function harness({ createOk = true, attachOk = true, applyAck = { status: 'applied', newTarget: 'rs' }, backend = [], backendDelayMs = 0 } = {}) {
  const socket = new FakeSocket();
  const rows = [];
  let items = [task('rs', null, 'Open', 'Яблоки'), task('rt', 'rs', 'Open', 'Голден')];
  const calls = [];
  const replies = [...backend];
  const fetchImpl = async (url, options) => {
    if (String(url).endsWith('/attach')) return attachOk ? { status: 101, webSocket: socket } : { status: 403, webSocket: null };
    calls.push({ url, body: JSON.parse(options.body) });
    if (String(url).endsWith('/responses')) {
      if (backendDelayMs) await new Promise(resolve => setTimeout(resolve, backendDelayMs));
      const output = replies.shift() || [];
      const id = `resp_${calls.filter(call => String(call.url).endsWith('/responses')).length}`;
      return { ok: true, status: 200, text: async () => JSON.stringify({ id, model: 'gpt-6.1-sol', output }) };
    }
    if (!createOk) return { ok: false, status: 400, text: async () => 'nope' };
    return { ok: true, status: 201, json: async () => ({ session: { id: 'live_1' }, transport: { type: 'webrtc', sdp: 'answer-sdp' } }) };
  };
  const applied = [];
  const host = new LiveHost({
    log: { append: (event, options) => { rows.push({ ...options, event }); } },
    settings: new LiveSettings(memory()),
    apiKey: 'sk-test',
    fetchImpl,
    services: {
      readItems: async () => items,
      readFrontier: async () => [{ taskId: 'rt', taskTitle: 'Голден' }],
      applyCommand: async (command, meta) => { applied.push({ command, meta }); return applyAck; }
    }
  });
  return { host, socket, rows, applied, calls, setItems: next => { items = next; } };
}

describe('starting a session', () => {
  it('sends the session config with the SDP offer and returns the answer', async () => {
    const { host, calls, socket } = harness();
    const result = await host.start({ sdp: 'offer-sdp' });
    expect(result).toEqual({ sessionId: 'live_1', sdp: 'answer-sdp' });
    expect(calls[0].body.transport).toEqual({ type: 'webrtc', sdp: 'offer-sdp' });
    expect(calls[0].body.session.model).toBe('gpt-live-1');
    expect(calls[0].body.session.instructions).toContain('rt\trs\tO\tГолден');
    expect(socket.accepted).toBe(true);
  });

  it('records the prompt versions and the requested store with the session', async () => {
    const { host, rows } = harness();
    await host.start({ sdp: 'offer-sdp' });
    const created = rows.find(row => row.event.type === 'vl.session.created');
    expect(created.event.requested.store).toBe(true);
    expect(created.event.requested.voicePromptVersion).toBeTruthy();
    expect(created.event.requested.backendModel).toBe('gpt-6.1-sol');
  });

  it('fails when the sideband cannot attach, rather than leaving a voice that cannot act', async () => {
    const { host } = harness({ attachOk: false });
    await expect(host.start({ sdp: 'offer-sdp' })).rejects.toThrow();
    expect(host.active).toBe(false);
  });

  it('reports a rejected session without pretending it started', async () => {
    const { host } = harness({ createOk: false });
    await expect(host.start({ sdp: 'offer-sdp' })).rejects.toThrow();
  });
});

describe('logging', () => {
  let context;
  beforeEach(async () => { context = harness(); await context.host.start({ sdp: 'offer-sdp' }); });

  it('keeps every framework event, incoming and outgoing', async () => {
    context.socket.deliver({ type: 'session.started', session: { id: 'live_1' } });
    const types = context.rows.map(row => row.event.type);
    expect(types).toContain('session.started');
    // Everything from the session itself is keyed by its id; only the record of the attempt
    // predates the id, which is the point of writing it before the call goes out.
    expect(context.rows.filter(row => row.event.type !== 'vl.session.requested').every(row => row.sessionId === 'live_1')).toBe(true);
    expect(context.rows[0].event.type).toBe('vl.session.requested');
  });

  it('records an attempt that never reaches OpenAI', async () => {
    const failing = harness();
    failing.host.fetchImpl = async () => { throw new Error('network down'); };
    await expect(failing.host.start({ sdp: 'offer-sdp' })).rejects.toThrow(/GPT-Live/);
    const types = failing.rows.map(row => row.event.type);
    expect(types).toEqual(['vl.session.requested', 'vl.session.unreachable']);
    expect(failing.rows.at(-1).event.error.message).toBe('network down');
  });

  it('records the detail when GPT-Live rejects the session', async () => {
    const rejected = harness({ createOk: false });
    await expect(rejected.host.start({ sdp: 'offer-sdp' })).rejects.toThrow();
    const failure = rejected.rows.find(row => row.event.type === 'vl.session.create_failed');
    expect(failure.event).toMatchObject({ status: 400, detail: 'nope' });
  });

  it('drops audio bytes and streamed text deltas', async () => {
    const before = context.rows.length;
    context.socket.deliver({ type: 'session.output_audio.delta', audio: 'AAAA' });
    context.socket.deliver({ type: 'response.output_text.delta', delta: 'при' });
    expect(context.rows).toHaveLength(before);
  });

  it('writes speech as one assembled record rather than a row per fragment', async () => {
    for (const [delta, from, to] of [['по', 10, 200], ['ставь ', 200, 400], ['в фокус', 400, 900]]) {
      context.socket.deliver({ type: 'session.input_transcript.delta', delta, start_ms: from, end_ms: to });
    }
    expect(context.rows.some(row => row.event.type === 'vl.speech')).toBe(false);

    // A delegation is a boundary: it is the moment the turn was acted on.
    context.socket.deliver({ type: 'session.delegation.created', delegation: { id: 'item_1', target: 'responses' }, offset_ms: 950 });
    const speech = context.rows.find(row => row.event.type === 'vl.speech');
    expect(speech.event).toMatchObject({ role: 'user', text: 'поставь в фокус', start_ms: 10, end_ms: 900 });
    expect(context.rows.filter(row => row.event.type === 'vl.speech')).toHaveLength(1);
  });

  it('is not cut by the backend stream running alongside it', async () => {
    context.socket.deliver({ type: 'session.input_transcript.delta', delta: 'Сек', start_ms: 0, end_ms: 200 });
    context.socket.deliver({ type: 'response.event', delegation_id: 'item_1', event: { type: 'response.created', response: { id: 'r1' } } });
    context.socket.deliver({ type: 'session.usage.updated', usage: {} });
    context.socket.deliver({ type: 'session.input_transcript.delta', delta: 'унду.', start_ms: 200, end_ms: 400 });
    context.socket.deliver({ type: 'session.closed', reason: 'client' });
    expect(context.rows.filter(row => row.event.type === 'vl.speech').map(row => row.event.text)).toEqual(['Секунду.']);
  });

  it('separates two utterances of the same speaker by the silence between them', async () => {
    context.socket.deliver({ type: 'session.input_transcript.delta', delta: 'переименуй первый', start_ms: 0, end_ms: 900 });
    context.socket.deliver({ type: 'session.input_transcript.delta', delta: 'а ещё добавь второй', start_ms: 4000, end_ms: 5000 });
    context.socket.deliver({ type: 'session.closed', reason: 'client' });
    expect(context.rows.filter(row => row.event.type === 'vl.speech').map(row => row.event.text))
      .toEqual(['переименуй первый', 'а ещё добавь второй']);
  });

  it('starts a new record when the other speaker begins', async () => {
    context.socket.deliver({ type: 'session.input_transcript.delta', delta: 'привет', start_ms: 0, end_ms: 100 });
    context.socket.deliver({ type: 'session.output_transcript.delta', delta: 'слушаю', start_ms: 120, end_ms: 300 });
    context.socket.deliver({ type: 'session.closed', reason: 'client' });
    expect(context.rows.filter(row => row.event.type === 'vl.speech').map(row => [row.event.role, row.event.text]))
      .toEqual([['user', 'привет'], ['assistant', 'слушаю']]);
  });

  it('records an unparsable frame instead of dropping it silently', async () => {
    context.socket.emit('message', { data: '{not json' });
    expect(context.rows.at(-1).event.type).toBe('vl.event.unparsed');
  });
});

const settle = () => new Promise(resolve => setTimeout(resolve, 0));
const fn = (name, args, callId = 'call_1') => ({ type: 'function_call', call_id: callId, name, arguments: JSON.stringify(args) });
const say = (text) => ({ type: 'message', content: [{ type: 'output_text', text }] });
const backendCalls = (context) => context.calls.filter(call => String(call.url).endsWith('/responses'));
const spoken = (context) => context.socket.sent.filter(message => message.type === 'session.commentary.append');

/** The voice hands the request over in silence; what comes back is decided here. */
async function request(context, text, { offset = 1_000 } = {}) {
  context.socket.deliver({ type: 'session.input_transcript.delta', delta: text, start_ms: offset - 900, end_ms: offset - 100 });
  context.socket.deliver({ type: 'session.delegation.created', delegation: { id: 'item_1', type: 'delegation', target: 'client' }, offset_ms: offset });
}

describe('delegated requests', () => {
  afterEach(() => { vi.useRealTimers(); });

  it('runs on client delegation, so the server answers every handover', async () => {
    const context = harness();
    await context.host.start({ sdp: 'offer-sdp' });
    expect(context.calls[0].body.session.delegation).toEqual({ type: 'client' });
  });

  it('answers «добавь X» by the local parser, without the backend', async () => {
    const context = harness();
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'добавь купить молоко');
    await settle();

    expect(backendCalls(context)).toHaveLength(0);
    expect(context.applied[0].command).toMatchObject({ command: 'addChild', actId: 'inbox', payload: { title: 'купить молоко' }, source: 'gpt-live' });
    expect(spoken(context)).toEqual([expect.objectContaining({ delegation_id: 'item_1', content: 'Добавил: купить молоко' })]);
  });

  it('asks the backend, applies its writes and confirms them in a few words at once', async () => {
    const context = harness({ backend: [[fn('setStatus', { taskId: 'rt', status: 'Pause' })], [say('Готово, на паузе.')]] });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'поставь голден на паузу');
    await settle();

    const [call] = backendCalls(context);
    expect(call.body.model).toBe('gpt-6.1-sol');
    expect(call.body.instructions).toContain('rt\trs\tO\tГолден');
    expect(call.body.input).toEqual([{ role: 'user', content: 'поставь голден на паузу' }]);
    expect(context.applied[0].command).toMatchObject({ command: 'setStatus', actId: 'rt', payload: { status: 'Pause' } });
    // The backend's own closing words after a write are not spoken: the server already said it.
    expect(spoken(context).map(message => message.content)).toEqual(['Голден на паузе']);
  });

  it('lets the backend finish a request of several changes after the first is confirmed', async () => {
    const context = harness({ backend: [
      [fn('setStatus', { taskId: 'rt', status: 'Focus' })],
      [fn('setStatus', { taskId: 'rs', status: 'Pause' }, 'call_2')],
      []
    ] });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'голден в фокус, а яблоки на паузу');
    await settle();

    expect(context.applied.map(entry => entry.command.actId)).toEqual(['rt', 'rs']);
    expect(spoken(context).map(message => message.content)).toEqual(['Голден в фокусе', 'Яблоки на паузе']);
    expect(backendCalls(context)[1].body.previous_response_id).toBe('resp_1');
  });

  it('takes another pass after a read, because the answer depends on it', async () => {
    const context = harness({ backend: [[fn('getFrontier', {})], [fn('setStatus', { taskId: 'rt', status: 'Focus' }, 'call_2')]] });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'возьми в фокус первое из фронтира');
    await settle();

    const second = backendCalls(context)[1].body;
    expect(second.previous_response_id).toBe('resp_1');
    expect(second.input[0]).toMatchObject({ type: 'function_call_output', call_id: 'call_1' });
    expect(JSON.parse(second.input[0].output).frontier[0].taskTitle).toBe('Голден');
    expect(spoken(context).map(message => message.content)).toEqual(['Голден в фокусе']);
  });

  it('leaves «добавь X куда-то» to the backend, which knows where that is', async () => {
    const context = harness({ backend: [[fn('addChild', { parentId: 'rs', title: 'молоко' })], []] });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'добавь молоко в яблоки');
    await settle();

    expect(backendCalls(context).length).toBeGreaterThan(0);
    expect(context.applied[0].command).toMatchObject({ command: 'addChild', actId: 'rs' });
  });

  it('still speaks a question the backend asks after a write', async () => {
    const context = harness({ backend: [[fn('setStatus', { taskId: 'rt', status: 'Pause' })], [say('Клубнику в Хлебе или на Даче?')]] });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'голден на паузу и перенеси клубнику');
    await settle();

    expect(spoken(context).map(message => message.content)).toEqual(['Голден на паузе', 'Клубнику в Хлебе или на Даче?']);
  });

  it('names what landed alongside what failed', async () => {
    let calls = 0;
    const context = harness({ backend: [[fn('setStatus', { taskId: 'rt', status: 'Pause' }), fn('setParent', { taskId: 'rs', parentId: 'нет' }, 'call_2')]] });
    const apply = context.host.services.applyCommand;
    context.host.services.applyCommand = async (command, meta) => (++calls === 2 ? { status: 'rejected', reason: 'нет такой задачи' } : apply(command, meta));
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'голден на паузу, яблоки в никуда');
    await settle();

    expect(spoken(context).map(message => message.content)).toEqual(['Голден на паузе, не вышло: нет такой задачи']);
  });

  it('does not claim success when nothing came back', async () => {
    const context = harness({ backend: [[]] });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'сделай что-нибудь');
    await settle();

    expect(spoken(context).map(message => message.content)).toEqual(['Не вышло: ответа нет']);
  });

  it('passes the backend\'s own question to the voice', async () => {
    const context = harness({ backend: [[say('Голден в Яблоках или в Саду?')]] });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'голден в фокус');
    await settle();

    expect(context.applied).toHaveLength(0);
    expect(spoken(context).map(message => message.content)).toEqual(['Голден в Яблоках или в Саду?']);
  });

  it('always speaks a failure', async () => {
    const context = harness({ applyAck: { status: 'rejected', reason: 'Задача не найдена' }, backend: [[fn('editItem', { taskId: 'нет', title: 'Новое' })]] });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'переименуй её');
    await settle();

    expect(spoken(context).map(message => message.content)).toEqual(['Не вышло: Задача не найдена']);
  });

  it('refuses an unsupported operation without touching the document', async () => {
    const context = harness({ backend: [[fn('deleteItem', { taskId: 'rt' })]] });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'удали голден');
    await settle();

    expect(context.applied).toHaveLength(0);
    expect(spoken(context)[0].content).toMatch(/^Не вышло/);
  });

  it('says «Думаю» after three seconds and the late result as an aside', async () => {
    vi.useFakeTimers();
    const context = harness({ backend: [[fn('setStatus', { taskId: 'rt', status: 'Pause' })]], backendDelayMs: 4_000 });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'поставь голден на паузу');
    await vi.advanceTimersByTimeAsync(5_000);

    expect(spoken(context).map(message => message.content)).toEqual(['Думаю.', 'Кстати, голден на паузе']);
  });

  it('waits for the voice to fall quiet before the next spoken update', async () => {
    vi.useFakeTimers();
    const context = harness({ backend: [[fn('setStatus', { taskId: 'rt', status: 'Pause' })], []], backendDelayMs: 3_300 });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'поставь голден на паузу');
    await vi.advanceTimersByTimeAsync(3_100);
    context.socket.deliver({ type: 'session.output_transcript.delta', delta: 'Думаю.', start_ms: 4_100, end_ms: 4_500 });
    await vi.advanceTimersByTimeAsync(400);
    expect(spoken(context).map(message => message.content)).toEqual(['Думаю.']);
    await vi.advanceTimersByTimeAsync(400);
    expect(spoken(context).map(message => message.content)).toEqual(['Думаю.', 'Кстати, голден на паузе']);
  });

  it('keeps a late result silent when the conversation has moved on', async () => {
    vi.useFakeTimers();
    const context = harness({ backend: [[fn('setStatus', { taskId: 'rt', status: 'Pause' })]], backendDelayMs: 4_000 });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'поставь голден на паузу');
    context.socket.deliver({ type: 'session.input_transcript.delta', delta: 'а погода завтра', start_ms: 1_500, end_ms: 2_500 });
    await vi.advanceTimersByTimeAsync(5_000);

    expect(spoken(context)).toHaveLength(0);
    const quiet = context.socket.sent.find(message => message.type === 'session.thinking.append' && message.delegation_id === 'item_1');
    expect(quiet.content).toContain('Голден на паузе');
  });

  it('keeps the voice\'s table current as quiet context', async () => {
    const context = harness({ backend: [[fn('setStatus', { taskId: 'rt', status: 'Focus' })]] });
    await context.host.start({ sdp: 'offer-sdp' });
    context.setItems([task('rs', null, 'Open', 'Яблоки'), task('rt', 'rs', 'Focus', 'Голден')]);
    await request(context, 'голден в фокус');
    await settle();

    const delta = context.socket.sent.find(message => message.type === 'session.thinking.append' && message.delegation_id === null);
    expect(delta.content).toContain('* rt F');
  });
});

describe('prompt edits by Sol', () => {
  it('appends a rule to the voice prompt and applies it in the running session', async () => {
    const context = harness({ backend: [[fn('setVoicePrompt', { mode: 'append', text: 'Отвечай короче.' })]] });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'да, меняй');
    await settle();

    expect(await context.host.settings.prompt('voice')).toContain('Отвечай короче.');
    const append = context.socket.sent.find(message => message.type === 'session.instructions.append' && message.content.includes('Отвечай короче.'));
    expect(append).toBeTruthy();
    expect(spoken(context).map(message => message.content)).toEqual(['Промпт изменён']);
  });

  it('uses its own new prompt from the very next request', async () => {
    const context = harness({ backend: [[fn('setBackendPrompt', { mode: 'replace', text: 'Новый промпт Sol.' })], [], [say('ок')]] });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'да, меняй');
    await settle();
    await request(context, 'а теперь проверь', { offset: 5_000 });
    await settle();

    expect(backendCalls(context)[2].body.instructions.startsWith('Новый промпт Sol.')).toBe(true);
  });

  it('reads a prompt before replacing it', async () => {
    const context = harness({ backend: [[fn('getVoicePrompt', {})], [say('Сейчас там правила голоса.')]] });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'что в промпте голоса');
    await settle();

    expect(JSON.parse(backendCalls(context)[1].body.input[0].output).prompt).toBe(DEFAULT_VOICE_PROMPT);
  });

  it('keeps both texts of the edit so it can be read back and undone', async () => {
    const context = harness({ backend: [[fn('setVoicePrompt', { mode: 'replace', text: 'Переписанный промпт.' })]] });
    await context.host.start({ sdp: 'offer-sdp' });
    await request(context, 'да, меняй');
    await settle();

    const [version] = await context.host.settings.history();
    expect(version).toMatchObject({ target: 'voice', source: 'model', after: 'Переписанный промпт.' });
    expect(version.before).toBe(DEFAULT_VOICE_PROMPT);
  });
});

describe('stopping', () => {
  it('asks the session to close and lets go of the socket', async () => {
    const context = harness();
    await context.host.start({ sdp: 'offer-sdp' });
    expect(await context.host.stop('client')).toBe(true);
    expect(context.socket.sent.some(message => message.type === 'session.close')).toBe(true);
    expect(context.socket.closed).toBe(true);
    expect(context.host.active).toBe(false);
  });

  it('still attributes the sideband close to its session', async () => {
    const context = harness();
    await context.host.start({ sdp: 'offer-sdp' });
    await context.host.stop('client');
    context.socket.emit('close', {});
    const closed = context.rows.find(row => row.event.type === 'vl.sideband.closed');
    expect(closed.sessionId).toBe('live_1');
  });

  it('lets go when the session reports itself closed', async () => {
    const context = harness();
    await context.host.start({ sdp: 'offer-sdp' });
    context.socket.deliver({ type: 'session.closed', reason: 'duration_limit', usage: { seconds: 42 } });
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(context.host.active).toBe(false);
    expect(context.rows.some(row => row.event.type === 'session.closed')).toBe(true);
  });
});

describe('replaying a dialogue against the voice layer', () => {
  it('sends the voice instructions with the task table and returns what it would say', async () => {
    const sent = [];
    const host = new LiveHost({
      log: { append: () => {} },
      settings: new LiveSettings(memory()),
      apiKey: 'sk-test',
      fetchImpl: async (url, options) => {
        sent.push({ url: String(url), body: JSON.parse(options.body) });
        return {
          ok: true,
          status: 200,
          text: async () => JSON.stringify({
            output: [{ type: 'message', content: [{ text: 'Голден на Яблоках или Голден на Даче?' }] }],
            usage: { input_tokens: 700 }
          })
        };
      },
      services: {
        readItems: async () => [task('rs', null, 'Open', 'Яблоки'), task('rt', 'rs', 'Open', 'Голден')],
        readFrontier: async () => [],
        applyCommand: async () => ({ status: 'applied' })
      }
    });

    const result = await host.simulateVoice([{ role: 'user', text: 'переименуй голден' }]);

    expect(sent[0].url).toContain('/responses');
    expect(sent[0].body.instructions).toContain('rt\trs\tO\tГолден');
    expect(sent[0].body.instructions).toContain('SpawnThinking');
    // The voice layer holds no tools of its own, so none are offered here either.
    expect(sent[0].body.tools).toBeUndefined();
    expect(sent[0].body.store).toBe(false);
    expect(sent[0].body.input[0]).toEqual({ role: 'user', content: 'переименуй голден' });
    // The stand-in is told what the live transport enforces on its own: no tools, speech only.
    expect(sent[0].body.input[1].role).toBe('developer');
    expect(sent[0].body.input[1].content).toContain('Инструментов у тебя нет');
    expect(result.layer).toBe('voice');
    expect(result.snapshotTasks).toBe(2);
    expect(result.text).toContain('Голден на Яблоках');
  });
});
