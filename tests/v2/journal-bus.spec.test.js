import { describe, it, expect } from 'vitest';
import { JSDOM } from 'jsdom';
import { DocumentRuntime } from '../../src/v2/domain/document-runtime.js';
import { TaskAgent } from '../../src/v2/domain/task-agent.js';
import { InteractionJournal } from '../../src/v2/domain/interaction-journal.js';
import { Client } from '../../src/v2/client/client.js';

const context = (runtime, extra = {}) => ({ elementId: 'task:milk1', view: 'list', revision: runtime.graph.revision, ...extra });
const request = (runtime, seq, value, extra = {}) => ({ key: { clientKey: 'journal-spec', seq }, context: context(runtime, extra), ...value });

describe('v0.2 InteractionJournal bus comprehensive contract', () => {
  it('covers model, UI, corrections, rollback, idempotency, recovery, validation, parser and context reveal', async () => {
    const modelCalls = [];
    const model = async ({ modelContext }) => {
      modelCalls.push(structuredClone(modelContext));
      if (modelContext.text.toLowerCase().includes('исправь')) return JSON.stringify({ answer: 'Статус исправлен', commands: [{ command: 'setStatus', actId: 'milk1', payload: { status: 'Done' } }] });
      return JSON.stringify({ answer: 'Перевёл в фокус', commands: [{ command: 'setStatus', actId: 'milk1', payload: { status: 'Focus' } }] });
    };
    const runtime = new DocumentRuntime({ resolveModel: model });

    const rootInput = request(runtime, 1, { text: 'Сделай молоко фокусом' });
    const rootReceipt = await runtime.executeAndWait(rootInput);
    const rootId = rootReceipt.requestId;
    const root = runtime.journal.get(rootId);
    expect(root).toMatchObject({ type: 'interaction', kind: 'text', text: rootInput.text, answer: 'Перевёл в фокус' });
    expect(root.modelContext).toEqual(modelCalls[0]);
    expect(root.rawModelResponse).toBe(JSON.stringify({ answer: 'Перевёл в фокус', commands: [{ command: 'setStatus', actId: 'milk1', payload: { status: 'Focus' } }] }));
    expect(root.commands[0].command).toBe('setStatus');
    expect(Object.keys(root.versions).sort()).toEqual(['contextBuilder', 'parser', 'prompt']);
    expect(runtime.graph.read({ id: 'milk1' }).status).toBe('Focus');
    expect(runtime.journal.entries).toHaveLength(1);
    expect(runtime.journal.entries.some(entry => ['result', 'settled', 'command'].includes(entry.type))).toBe(false);
    expect(runtime.state.technical.commandKeys[`${rootId}:0`]).toBeTruthy();

    const uiReceipt = await runtime.executeAndWait(request(runtime, 2, { command: { command: 'setDeadline', actId: 'milk1', payload: { deadline: '2026-10-20' } } }));
    expect(uiReceipt.actions).toEqual([]);
    expect(modelCalls).toHaveLength(1);
    expect(runtime.graph.read({ id: 'milk1' }).deadline).toBe('2026-10-20');
    expect(runtime.journal.actions().map(action => action.id)).toEqual([rootId]);

    const correctionReceipt = await runtime.executeAndWait(request(runtime, 3, { text: 'Исправь: статус готово' }, { elementId: `action:${rootId}`, view: 'action', actionId: rootId }));
    const correction = runtime.journal.get(correctionReceipt.requestId);
    expect(correction.corrects).toBe(rootId);
    expect(modelCalls).toHaveLength(2);
    expect(runtime.graph.read({ id: 'milk1' }).status).toBe('Done');
    expect(runtime.journal.actions().map(action => action.id)).toEqual([rootId]);

    const rollbackReceipt = await runtime.executeAndWait(request(runtime, 4, { command: { command: 'rollbackAction', actId: rootId, payload: {} } }, { elementId: `action:${rootId}`, view: 'action', actionId: rootId }));
    const rollback = runtime.journal.get(rollbackReceipt.requestId);
    expect(rollback).toMatchObject({ kind: 'ui', corrects: rootId, command: { command: 'rollbackAction' } });
    expect(modelCalls).toHaveLength(2);
    expect(rollbackReceipt.actions).toEqual([]);
    expect(runtime.graph.read({ id: 'milk1' })).toMatchObject({ status: 'Open', deadline: '2026-10-20' });
    expect(runtime.journal.actions()).toHaveLength(1);
    expect(runtime.journal.actions()[0]).toMatchObject({ id: rootId, rolledBack: true, canRollback: false });

    const actionDocument = runtime.getDocument({ view: 'action', actionId: rootId });
    const actionPage = actionDocument.root.children[1];
    expect(actionPage.props.records.map(record => record.id)).toEqual([rootId, correction.id, rollback.id]);
    expect(actionPage.props.records[0].modelContext).toEqual(root.modelContext);
    expect(actionPage.props.records[2]).not.toHaveProperty('modelContext');

    const dom = new JSDOM('<!doctype html><div id="app-root"></div>', { url: 'https://example.test/' });
    const client = new Client({ document: dom.window.document, storage: dom.window.sessionStorage, fetch: async () => { throw new Error('unused'); }, pollMs: 0 });
    client.renderActionPage(actionPage);
    const answer = dom.window.document.querySelector('.v02-model-answer');
    const savedContext = answer.previousElementSibling;
    expect(savedContext.classList.contains('v02-model-context')).toBe(true);
    expect(savedContext.hidden).toBe(true);
    answer.click();
    expect(savedContext.hidden).toBe(false);
    expect(answer.getAttribute('aria-expanded')).toBe('true');

    const duplicate = request(runtime, 5, { command: { command: 'setTags', actId: 'milk1', payload: { tag: 'важно' } } });
    const accepted = await Promise.all([runtime.submit(duplicate), runtime.submit(duplicate)]);
    expect(accepted[0].requestId).toBe(accepted[1].requestId);
    await runtime.processPending();
    expect(runtime.graph.read({ id: 'milk1' }).tags).toEqual(['важно']);
    await expect(runtime.submit({ ...duplicate, command: { command: 'setStatus', actId: 'milk1', payload: { status: 'Pause' } } })).rejects.toMatchObject({ code: 'REQUEST_KEY_REUSED' });

    const pendingModel = async () => JSON.stringify({ answer: '', commands: [{ command: 'setStatus', actId: 'milk1', payload: { status: 'Pause' } }] });
    const beforeRestart = new DocumentRuntime({ resolveModel: pendingModel });
    const pendingReceipt = await beforeRestart.submit(request(beforeRestart, 1, { text: 'Поставь на паузу' }));
    await beforeRestart.processHarness(beforeRestart.journal.get(pendingReceipt.requestId));
    expect(beforeRestart.state.technical.executor[pendingReceipt.requestId].status).toBe('pending');
    const restarted = new DocumentRuntime({ initialState: beforeRestart.exportState(), resolveModel: pendingModel });
    await restarted.processPending();
    await restarted.processPending();
    expect(restarted.graph.read({ id: 'milk1' }).status).toBe('Pause');
    expect(restarted.state.technical.executor[pendingReceipt.requestId].outcomes).toHaveLength(1);

    const checkpoint = runtime.exportState();
    await expect(runtime.submit({ key: { clientKey: 'bad', seq: 1 }, context: context(runtime), text: '' })).rejects.toMatchObject({ code: 'INVALID_INPUT' });
    await expect(runtime.submit({ key: { clientKey: 'bad', seq: 2 }, context: context(runtime), text: 'x', command: { command: 'setStatus' } })).rejects.toMatchObject({ code: 'INVALID_INPUT' });
    await expect(runtime.submit(request(runtime, 6, { text: 'коррекция' }, { elementId: 'action:missing', view: 'action', actionId: 'missing' }))).rejects.toMatchObject({ code: 'NOT_FOUND' });
    await expect(runtime.submit({ ...request(runtime, 7, { command: { command: 'setStatus', actId: 'milk1', payload: { status: 'Info' } } }), context: { ...context(runtime), revision: runtime.graph.revision - 1 } })).rejects.toMatchObject({ code: 'CONFLICT' });
    expect(runtime.exportState()).toEqual(checkpoint);

    const parser = new TaskAgent();
    expect(parser.parse('{"answer":"ok"}')).toEqual({ answer: 'ok', commands: [] });
    expect(parser.parse('{"commands":[{"command":"showList"}]}')).toEqual({ answer: '', commands: [{ command: 'showList' }] });
    expect(parser.parse('{"answer":"ok","commands":[{"command":"showList"}]}')).toEqual({ answer: 'ok', commands: [{ command: 'showList' }] });

    const answerOnlyRuntime = new DocumentRuntime({ resolveModel: async ({ text }) => text === 'root answer'
      ? '{"answer":"only an answer"}'
      : '{"commands":[{"command":"setStatus","actId":"milk1","payload":{"status":"Info"}}]}' });
    const answerRoot = await answerOnlyRuntime.executeAndWait(request(answerOnlyRuntime, 1, { text: 'root answer' }));
    expect(answerOnlyRuntime.journal.actions()[0].canRollback).toBe(false);
    await answerOnlyRuntime.executeAndWait(request(answerOnlyRuntime, 2, { text: 'correction mutates' }, { elementId: `action:${answerRoot.requestId}`, view: 'action', actionId: answerRoot.requestId }));
    expect(answerOnlyRuntime.journal.actions()[0].canRollback).toBe(true);

    const legacyRows = [
      { id: 'e1', cursor: 1, at: '2026-01-01', type: 'input', input: { key: { clientKey: 'old', seq: 1 }, context: { elementId: 'task:milk1', view: 'list', revision: 0 }, text: 'root' } },
      { id: 'e2', cursor: 2, at: '2026-01-01', type: 'command', requestId: 'e1', rootActionId: 'e2', command: { command: 'setStatus', actId: 'milk1', payload: { status: 'Focus' } } },
      { id: 'e3', cursor: 3, at: '2026-01-01', type: 'result', requestId: 'e1', actionId: 'e2', rootActionId: 'e2', status: 'applied', changes: [] },
      { id: 'e4', cursor: 4, at: '2026-01-01', type: 'settled', requestId: 'e1' },
      { id: 'e5', cursor: 5, at: '2026-01-02', type: 'input', input: { key: { clientKey: 'old', seq: 2 }, context: { elementId: 'action:e2', view: 'action', revision: 1, actionId: 'e2' }, text: 'fix' } }
    ];
    const legacy = new InteractionJournal(legacyRows);
    expect(legacy.entries.map(entry => entry.type)).toEqual(['interaction', 'interaction']);
    expect(legacy.get('e5').corrects).toBe('e1');
    const mixed = new InteractionJournal([legacy.get('e1'), ...legacyRows.slice(1)]);
    expect(mixed.entries.map(entry => entry.id)).toEqual(['e1', 'e5']);
    expect(mixed.get('e5').corrects).toBe('e1');
    const migratedRuntime = new DocumentRuntime({ initialState: { graph: runtime.graph.read(), entries: legacyRows } });
    const reloadedMigration = new DocumentRuntime({ initialState: migratedRuntime.exportState() });
    expect(reloadedMigration.journal.entries.map(entry => entry.type)).toEqual(['interaction', 'interaction']);
    expect(reloadedMigration.journal.get('e5').corrects).toBe('e1');
  });
});
