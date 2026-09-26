import { describe, expect, it } from 'vitest';
import { DocumentRuntime } from '../../src/v2/domain/document-runtime.js';
import { CompatibilityPort } from '../../worker/v2/compatibility.js';
import { GeminiHost } from '../../worker/v2/gemini-host.js';
import { LiveSettings } from '../../src/v2/domain/live-settings.js';

const memory = () => { const map = new Map(); return { async get(key) { return map.get(key); }, async put(key, value) { map.set(key, value); } }; };

/** A voice session wired the way the durable object wires it: the host talks to the runtime
 *  through the same port the interface and MCP use, and nothing else. */
function session() {
  const runtime = new DocumentRuntime({ seed: { items: [], revision: 0, nextId: 1 } });
  const port = new CompatibilityPort(runtime);
  const host = new GeminiHost({
    log: { append: () => {} }, settings: new LiveSettings(memory()), apiKey: 'k',
    services: {
      readItems: async () => runtime.graph.read().items,
      readFrontier: async () => [],
      applyCommand: (command, message, corrects) => port.applyCommand(command, { message, corrects }),
      recordSpeech: (speech) => runtime.recordSpeech(speech)
    }
  });
  host.session = { id: 'gm_test' };
  const heard = (role, text) => ({ direction: 'in', frame: { serverContent: role === 'user'
    ? { inputTranscription: { text } } : { outputTranscription: { text } } } });
  return { runtime, port, host, heard };
}

const call = (id, name, args) => ({ toolCall: { functionCalls: [{ id, name, args }] } });
const turnEnd = { direction: 'in', frame: { vlTurnEnd: true } };

describe('a change made by voice', () => {
  it('reaches the bus as one action: the phrase, the change and the answer', async () => {
    const { runtime, host, heard } = session();
    host.mirror([heard('user', 'Добавь'), heard('user', 'купить молоко')]);
    await host.invokeAll(call('c1', 'addItem', { line1: 'Купить молоко' }));
    host.mirror([heard('assistant', 'Готово.')]);
    await host.stop('client');

    const [action] = runtime.journal.actions();
    expect(runtime.journal.actions()).toHaveLength(1);
    expect(action).toMatchObject({ label: 'Добавь купить молоко', status: 'applied', source: 'gemini-live', canRollback: true });
    expect(action.target).toBeTruthy();
    expect(runtime.journal.dialogue(action.id)).toEqual([
      { role: 'user', text: 'Добавь купить молоко', id: action.id, kind: 'speech' },
      { role: 'assistant', text: 'Готово.', id: expect.any(String), kind: 'speech' }
    ]);
  });

  it('is rolled back with the phrase that caused it', async () => {
    const { runtime, port, host, heard } = session();
    host.mirror([heard('user', 'Добавь хлеб')]);
    await host.invokeAll(call('c1', 'addItem', { line1: 'Хлеб' }));
    await host.flushSpeech();
    expect(runtime.graph.read().items.some(item => item.line1 === 'Хлеб')).toBe(true);

    const [action] = runtime.journal.actions();
    await port.applyCommand({ command: 'rollbackAction', actId: action.id, actType: 'action', payload: {} }, { message: { clientKey: 'ui', seq: 1 } });
    expect(runtime.graph.read().items.some(item => item.line1 === 'Хлеб')).toBe(false);
  });

  it('closes a turn when the page says it ended, without waiting for the session to end', async () => {
    const { runtime, host, heard } = session();
    host.mirror([heard('user', 'Добавь соль'), turnEnd]);
    await host.turns;
    // The object can restart at any moment; a turn still open at that point is lost.
    expect(runtime.journal.entries.map(entry => entry.text)).toEqual(['Добавь соль']);
  });

  it('keeps a turn that changed nothing, as a turn that changed nothing', async () => {
    const { runtime, host, heard } = session();
    host.mirror([heard('user', 'Что во фронтире?'), heard('assistant', 'Пока ничего срочного.')]);
    await host.stop('client');

    const actions = runtime.journal.actions();
    expect(actions).toHaveLength(1);
    expect(actions[0]).toMatchObject({ label: 'Что во фронтире?', status: 'needs-input', canRollback: false });
    expect(runtime.journal.dialogue(actions[0].id).map(message => message.role)).toEqual(['user', 'assistant']);
  });

  it('survives the page reporting the same frames twice', async () => {
    const { runtime, host, heard } = session();
    host.mirror([heard('user', 'Добавь хлеб')]);
    await host.flushSpeech();
    host.mirror([heard('user', 'Добавь хлеб')]);
    await host.flushSpeech();
    expect(runtime.journal.entries.filter(entry => entry.kind === 'speech')).toHaveLength(1);
  });

  it('records a change whose phrase never arrived, rather than dropping it', async () => {
    const { runtime, host } = session();
    await host.invokeAll(call('c1', 'addItem', { line1: 'Без слов' }));
    const [action] = runtime.journal.actions();
    expect(action).toMatchObject({ status: 'applied', source: 'gemini-live' });
    expect(action.label).toBe('addItem');
  });
});

describe('what the journal screen shows', () => {
  it('hides a button press and keeps what was said to an agent', async () => {
    const { runtime, port, host, heard } = session();
    await port.applyCommand({ command: 'addItem', actId: 'list', actType: 'list', payload: { line1: 'Рукой' } }, { message: { clientKey: 'ui', seq: 1 } });
    host.mirror([heard('user', 'Голосом')]);
    await host.invokeAll(call('c1', 'addItem', { line1: 'Голосом' }));
    await host.stop('client');

    expect(runtime.journal.actions().map(action => action.label)).toEqual(['Голосом']);
    expect(runtime.graph.read().items.map(item => item.line1)).toContain('Рукой');
  });
});
