import { describe, expect, it } from 'vitest';
import { GeminiHost } from '../../worker/v2/gemini-host.js';
import { LiveSettings } from '../../src/v2/domain/live-settings.js';

const memory = () => { const map = new Map(); return { map, async get(k) { return map.get(k); }, async put(k, v) { map.set(k, v); } }; };
const task = (id, parentId, status, line1) => ({ id, parentId, order: 10, status, line1, line2: '', collapsed: false, tags: [] });

function harness({ tokenOk = true, ack = { status: 'applied', newTarget: 'rt' } } = {}) {
  const rows = [];
  const calls = [];
  const applied = [];
  const host = new GeminiHost({
    log: { append: (event, options) => rows.push({ ...options, event }) },
    settings: new LiveSettings(memory()),
    apiKey: 'AIza-test-key-value-000000',
    now: () => new Date('2026-09-24T10:00:00.000Z'),
    fetchImpl: async (url, options) => {
      calls.push({ url: String(url), headers: options.headers, body: JSON.parse(options.body) });
      return tokenOk
        ? { ok: true, status: 200, text: async () => JSON.stringify({ name: 'auth_tokens/abc123' }) }
        : { ok: false, status: 403, text: async () => 'no' };
    },
    services: {
      readItems: async () => [task('rs', null, 'Open', 'Яблоки'), task('rt', 'rs', 'Open', 'Голден')],
      readFrontier: async () => [{ taskId: 'rt', taskTitle: 'Голден' }],
      applyCommand: async (command, message) => { applied.push({ command, message }); return ack; }
    }
  });
  return { host, rows, calls, applied };
}

describe('minting the session token', () => {
  it('asks Google with the key in a header and returns the token to the page', async () => {
    const { host, calls, rows } = harness();
    const result = await host.mintToken();

    expect(calls[0].url).toContain('/v1beta/auth_tokens');
    expect(calls[0].headers['x-goog-api-key']).toBe('AIza-test-key-value-000000');
    expect(result.token).toBe('auth_tokens/abc123');
    expect(host.active).toBe(true);

    const started = rows.find(row => row.event.type === 'gm.session.started');
    expect(started.event.tools).toContain('setStatus');
    expect(started.event.tools).not.toContain('setVoicePrompt');
    // The credential itself never reaches the log.
    expect(JSON.stringify(rows)).not.toContain('auth_tokens/abc123');
  });

  it('sends the task table inside the token constraints, not to the page', async () => {
    const { host, calls } = harness();
    await host.mintToken();
    const instruction = calls[0].body.liveConnectConstraints.config.systemInstruction.parts[0].text;
    expect(instruction).toContain('rt\trs\tO\tГолден');
  });

  it('records a refusal and fails loudly', async () => {
    const { host, rows } = harness({ tokenOk: false });
    await expect(host.mintToken()).rejects.toThrow(/Google отклонил/);
    expect(rows.some(row => row.event.type === 'gm.token.failed')).toBe(true);
  });

  it('refuses without a key', async () => {
    const { host } = harness();
    host.apiKey = '';
    await expect(host.mintToken()).rejects.toThrow(/Ключ Gemini/);
  });
});

describe('running a tool call', () => {
  it('maps the call onto a document command and answers with the result', async () => {
    const { host, applied, rows } = harness();
    await host.mintToken();
    const results = await host.invokeAll({
      toolCall: { functionCalls: [{ id: 'c1', name: 'setStatus', args: { taskId: 'rt', status: 'Focus' } }] }
    });

    expect(applied[0].command).toMatchObject({ actId: 'rt', command: 'setStatus', payload: { status: 'Focus' }, source: 'gemini-live' });
    expect(results[0]).toEqual({ id: 'c1', name: 'setStatus', response: { status: 'applied', operation: 'setStatus', target: 'rt' } });
    expect(rows.some(row => row.event.type === 'gm.tool.call')).toBe(true);
    expect(rows.some(row => row.event.type === 'gm.tool.result')).toBe(true);
  });

  it('passes a rejection back instead of throwing at the page', async () => {
    const { host } = harness({ ack: { status: 'rejected', reason: 'Задача не найдена' } });
    await host.mintToken();
    const [result] = await host.invokeAll({ toolCall: { functionCalls: [{ id: 'c2', name: 'editItem', args: { taskId: 'zz', line1: 'Нет' } }] } });
    expect(result.response).toEqual({ status: 'rejected', reason: 'Задача не найдена' });
  });

  it('turns a malformed call into a rejection, not an exception', async () => {
    const { host } = harness();
    await host.mintToken();
    const [result] = await host.invokeAll({ toolCall: { functionCalls: [{ id: 'c3', name: 'editItem', args: {} }] } });
    expect(result.response.status).toBe('rejected');
  });

  it('reads the frontier without touching the document', async () => {
    const { host, applied } = harness();
    await host.mintToken();
    const [result] = await host.invokeAll({ toolCall: { functionCalls: [{ id: 'c4', name: 'getFrontier', args: {} }] } });
    expect(result.response.frontier).toHaveLength(1);
    expect(applied).toHaveLength(0);
  });

  it('gives each call its own journal key, so two changes are two entries', async () => {
    const { host, applied } = harness();
    await host.mintToken();
    await host.invokeAll({ toolCall: { functionCalls: [
      { id: 'a', name: 'editItem', args: { taskId: 'rt', line1: 'Голден новый' } },
      { id: 'b', name: 'setStatus', args: { taskId: 'rt', status: 'Pause' } }
    ] } });
    expect(applied.map(entry => entry.message.seq)).toEqual([1, 2]);
  });
});

describe('mirroring what the page saw', () => {
  it('keeps readable frames, drops audio and caps the batch', async () => {
    const { host, rows } = harness();
    await host.mintToken();
    const audio = { serverContent: { modelTurn: { parts: [{ inlineData: { data: 'AAA' } }] } } };
    const text = { serverContent: { outputTranscription: { text: 'Готово' } } };
    const result = host.mirror([{ direction: 'in', frame: audio }, { direction: 'in', frame: text }]);

    expect(result).toEqual({ received: 2, kept: 1 });
    expect(rows.filter(row => row.event.type === 'gm.frame')).toHaveLength(1);

    const flood = Array.from({ length: 500 }, () => ({ direction: 'in', frame: text }));
    expect(host.mirror(flood).received).toBe(200);
  });
});

describe('proving a key before it is kept', () => {
  function keyHarness(replies) {
    const seen = [];
    const host = new GeminiHost({
      log: { append: () => {} },
      settings: new LiveSettings(memory()),
      apiKey: '',
      fetchImpl: async (url, options) => {
        seen.push({ url: String(url), key: options.headers['x-goog-api-key'] });
        const reply = replies[seen.length - 1] ?? replies.at(-1);
        return reply.ok
          ? { ok: true, status: 200, text: async () => '{"models":[]}' }
          : { ok: false, status: reply.status || 400, text: async () => reply.detail || 'API key not valid' };
      },
      services: { readItems: async () => [], readFrontier: async () => [], applyCommand: async () => ({ status: 'applied' }) }
    });
    return { host, seen };
  }

  it('reports a working key', async () => {
    const { host, seen } = keyHarness([{ ok: true }]);
    await expect(host.verifyKey('AIzaGood')).resolves.toEqual({ ok: true });
    expect(seen[0].url).toContain('/v1beta/models');
    expect(seen[0].key).toBe('AIzaGood');
  });

  it('reports why Google refused, so a typo is not stored as configured', async () => {
    const { host } = keyHarness([{ ok: false, status: 400, detail: 'API key not valid' }]);
    const result = await host.verifyKey('AIzaBad');
    expect(result).toEqual({ ok: false, status: 400, detail: 'API key not valid' });
  });

  it('treats an unreachable Google as a failed check rather than a pass', async () => {
    const host = new GeminiHost({
      log: { append: () => {} },
      settings: new LiveSettings(memory()),
      apiKey: '',
      fetchImpl: async () => { throw new Error('network down'); },
      services: { readItems: async () => [], readFrontier: async () => [], applyCommand: async () => ({ status: 'applied' }) }
    });
    expect((await host.verifyKey('AIzaAny')).ok).toBe(false);
  });
});
