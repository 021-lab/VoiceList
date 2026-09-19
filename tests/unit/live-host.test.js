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
    context.socket.deliver({ type: 'session.input_transcript.delta', delta: 'поставь', start_ms: 10, end_ms: 400 });
    const types = context.rows.map(row => row.event.type);
    expect(types).toContain('session.started');
    expect(types).toContain('session.input_transcript.delta');
    expect(context.rows.every(row => row.sessionId === 'live_1')).toBe(true);
  });

  it('drops audio bytes and streamed text deltas', async () => {
    const before = context.rows.length;
    context.socket.deliver({ type: 'session.output_audio.delta', audio: 'AAAA' });
    context.socket.deliver({ type: 'response.output_text.delta', delta: 'при' });
    expect(context.rows).toHaveLength(before);
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

  it('lets go when the session reports itself closed', async () => {
    const context = harness();
    await context.host.start({ sdp: 'offer-sdp' });
    context.socket.deliver({ type: 'session.closed', reason: 'duration_limit', usage: { seconds: 42 } });
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(context.host.active).toBe(false);
    expect(context.rows.some(row => row.event.type === 'session.closed')).toBe(true);
  });
});
