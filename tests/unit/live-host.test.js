import { beforeEach, describe, expect, it } from 'vitest';
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

const task = (id, parentId, status, line1) => ({ id, parentId, order: 10, status, line1, line2: '', collapsed: false, tags: [] });

function harness({ createOk = true, attachOk = true, applyAck = { status: 'applied', newTarget: 'rs' } } = {}) {
  const socket = new FakeSocket();
  const rows = [];
  let items = [task('rs', null, 'Open', 'Яблоки'), task('rt', 'rs', 'Open', 'Голден')];
  const calls = [];
  const fetchImpl = async (url, options) => {
    if (String(url).endsWith('/attach')) return attachOk ? { status: 101, webSocket: socket } : { status: 403, webSocket: null };
    calls.push({ url, body: JSON.parse(options.body) });
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
    expect(created.event.requested.backendModel).toBe('gpt-5.6-luna');
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

describe('tool calls', () => {
  const functionCall = (name, args, callId = 'call_1') => ({
    type: 'response.event', delegation_id: 'item_1',
    event: { type: 'response.output_item.done', item: { type: 'function_call', call_id: callId, name, arguments: JSON.stringify(args) } }
  });
  const settle = () => new Promise(resolve => setTimeout(resolve, 0));

  it('runs a task change through the ordinary command path and answers the call', async () => {
    const context = harness();
    await context.host.start({ sdp: 'offer-sdp' });
    context.socket.deliver(functionCall('setStatus', { taskId: 'rt', status: 'Focus' }));
    await settle();

    expect(context.applied[0].command).toMatchObject({ command: 'setStatus', actId: 'rt', payload: { status: 'Focus' }, source: 'gpt-live' });
    const result = context.socket.sent.find(message => message.type === 'response.item.create');
    expect(result.item).toMatchObject({ type: 'function_call_output', call_id: 'call_1' });
    expect(JSON.parse(result.item.output)).toMatchObject({ status: 'applied', operation: 'setStatus' });
    expect(context.socket.sent.some(message => message.type === 'response.create')).toBe(true);
  });

  it('pushes the resulting change to the voice layer as a delta', async () => {
    const context = harness();
    await context.host.start({ sdp: 'offer-sdp' });
    context.setItems([task('rs', null, 'Open', 'Яблоки'), task('rt', 'rs', 'Focus', 'Голден')]);
    context.socket.deliver(functionCall('setStatus', { taskId: 'rt', status: 'Focus' }));
    await settle();

    const append = context.socket.sent.find(message => message.type === 'session.instructions.append');
    expect(append.content).toContain('* rt F');
    expect(append.delegation_id).toBeNull();
  });

  it('reports a rejected command back to the model instead of claiming success', async () => {
    const context = harness({ applyAck: { status: 'rejected', reason: 'Задача не найдена' } });
    await context.host.start({ sdp: 'offer-sdp' });
    context.socket.deliver(functionCall('editItem', { taskId: 'нет', line1: 'Новое' }));
    await settle();

    const result = context.socket.sent.find(message => message.type === 'response.item.create');
    expect(JSON.parse(result.item.output)).toEqual({ status: 'rejected', reason: 'Задача не найдена' });
  });

  it('answers an unsupported operation without touching the document', async () => {
    const context = harness();
    await context.host.start({ sdp: 'offer-sdp' });
    context.socket.deliver(functionCall('deleteItem', { taskId: 'rt' }));
    await settle();

    expect(context.applied).toHaveLength(0);
    const result = context.socket.sent.find(message => message.type === 'response.item.create');
    expect(JSON.parse(result.item.output).status).toBe('rejected');
  });

  it('serves the frontier without a document change', async () => {
    const context = harness();
    await context.host.start({ sdp: 'offer-sdp' });
    context.socket.deliver(functionCall('getFrontier', {}));
    await settle();

    expect(context.applied).toHaveLength(0);
    const result = context.socket.sent.find(message => message.type === 'response.item.create');
    expect(JSON.parse(result.item.output).frontier[0].taskTitle).toBe('Голден');
  });
});

describe('prompt edits by the model', () => {
  const settle = () => new Promise(resolve => setTimeout(resolve, 0));
  const promptCall = (name, args) => ({
    type: 'response.event', delegation_id: 'item_2',
    event: { type: 'response.output_item.done', item: { type: 'function_call', call_id: 'call_9', name, arguments: JSON.stringify(args) } }
  });

  it('appends a rule to the voice prompt and applies it in the running session', async () => {
    const context = harness();
    await context.host.start({ sdp: 'offer-sdp' });
    context.socket.deliver(promptCall('setVoicePrompt', { mode: 'append', text: 'Отвечай короче.' }));
    await settle();

    expect(await context.host.settings.prompt('voice')).toContain('Отвечай короче.');
    const append = context.socket.sent.find(message => message.type === 'session.instructions.append' && message.content.includes('Отвечай короче.'));
    expect(append).toBeTruthy();
  });

  it('updates the backend prompt in place, which the session allows', async () => {
    const context = harness();
    await context.host.start({ sdp: 'offer-sdp' });
    context.socket.deliver(promptCall('setBackendPrompt', { mode: 'replace', text: 'Новый бэкенд-промпт.' }));
    await settle();

    const update = context.socket.sent.find(message => message.type === 'session.update');
    expect(update.session.delegation.responses.instructions).toBe('Новый бэкенд-промпт.');
  });

  it('lets the model read a prompt before replacing it', async () => {
    const context = harness();
    await context.host.start({ sdp: 'offer-sdp' });
    context.socket.deliver(promptCall('getVoicePrompt', {}));
    await settle();

    const result = context.socket.sent.find(message => message.type === 'response.item.create');
    expect(JSON.parse(result.item.output).prompt).toBe(DEFAULT_VOICE_PROMPT);
  });

  it('keeps both texts of the edit so it can be read back and undone', async () => {
    const context = harness();
    await context.host.start({ sdp: 'offer-sdp' });
    context.socket.deliver(promptCall('setVoicePrompt', { mode: 'replace', text: 'Переписанный промпт.' }));
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
    expect(sent[0].body.instructions).toContain('Выбор задачи');
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
