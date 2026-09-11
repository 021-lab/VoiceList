import { describe, expect, test } from 'vitest';

import { seedState } from '../../list-data.js';
import { createDocumentCore } from '../../worker/list-document-core.js';

function workflowyResponse(body, headers = {}) {
  return {
    ok: true,
    status: 200,
    headers: {
      get(name) {
        return headers[name.toLowerCase()] || null;
      }
    },
    async text() {
      return typeof body === 'string' ? body : JSON.stringify(body);
    },
    async json() {
      return typeof body === 'string' ? JSON.parse(body) : body;
    }
  };
}

describe('Cloudflare list document core', () => {
  test('starts from seed and applies commands through the server interpreter log', async () => {
    const core = createDocumentCore({ seedState, openRouterApiKey: '' });
    await core.init();

    const initial = core.getSnapshot();
    expect(initial.rev).toBe(0);
    expect(initial.content.snapshot.items[0]).toMatchObject({
      id: 'inbox',
      line1: 'Входящие'
    });
    expect(initial.content.snapshot.items.some((item) => item.id === 'milk1')).toBe(true);

    const applied = await core.handleClientMessage({
      type: 'command',
      clientKey: 'tab-a',
      seq: 1,
      input: {
        actId: 'list',
        actType: 'list',
        command: 'addItem',
        payload: { line1: 'Server task', line2: '' },
        source: 'unit-test'
      }
    });

    expect(applied.ack).toMatchObject({
      seq: 1,
      status: 'applied',
      newTarget: 'rs'
    });
    expect(applied.state.rev).toBe(1);
    expect(applied.state.content.snapshot.items.find((item) => item.id === 'rs')).toMatchObject({
      id: 'rs',
      parentId: null,
      status: 'Open',
      line1: 'Server task'
    });
    expect(core.listLog()).toHaveLength(1);
    expect(core.listLog()[0]).toMatchObject({
      id: '1',
      rev: 1,
      clientKey: 'tab-a',
      seq: 1,
      op: 'addItem',
      target: 'rs'
    });

    const duplicate = await core.handleClientMessage({
      type: 'command',
      clientKey: 'tab-a',
      seq: 1,
      input: {
        actId: 'list',
        actType: 'list',
        command: 'addItem',
        payload: { line1: 'Server task again', line2: '' },
        source: 'unit-test'
      }
    });

    expect(duplicate.ack).toEqual(applied.ack);
    expect(duplicate.state.rev).toBe(1);
    expect(core.listLog()).toHaveLength(1);
  });

  test('persists a deadline through the Durable Object command path', async () => {
    const core = createDocumentCore({ seedState, openRouterApiKey: '' });
    await core.init();

    const applied = await core.handleClientMessage({
      type: 'command',
      clientKey: 'tab-a',
      seq: 1,
      input: {
        actId: 'milk1',
        actType: 'task',
        command: 'setDeadline',
        payload: { deadline: '2026-09-01' },
        source: 'left-swipe-panel'
      }
    });

    expect(applied.ack).toMatchObject({ status: 'applied', newTarget: 'milk1' });
    expect(applied.state.content.snapshot.items.find((item) => item.id === 'milk1')).toMatchObject({ deadline: '2026-09-01' });
    expect(core.listLog().at(-1)).toMatchObject({ op: 'setDeadline', label: 'Дедлайн: 2026-09-01' });
  });

  test('stores voice comments on the addressed log entry', async () => {
    const core = createDocumentCore({ seedState, openRouterApiKey: '' });
    await core.init();

    await core.handleClientMessage({
      type: 'command',
      clientKey: 'tab-a',
      seq: 1,
      input: {
        actId: 'list',
        actType: 'list',
        command: 'addItem',
        payload: { line1: 'Commented task', line2: '' },
        source: 'unit-test'
      }
    });

    const commented = await core.handleClientMessage({
      type: 'command',
      clientKey: 'tab-a',
      seq: 2,
      input: {
        actId: '1',
        actType: 'log-entry',
        command: 'commentLogEntry',
        payload: { text: 'купить сегодня' },
        source: 'unit-test'
      }
    });

    expect(commented.ack).toMatchObject({
      seq: 2,
      status: 'applied',
      newTarget: '1'
    });
    expect(commented.state.content.actionLog.find((entry) => entry.id === '1').comments).toEqual([
      expect.objectContaining({ text: 'купить сегодня' })
    ]);
  });

  test('applies collapse without adding an action-log entry', async () => {
    const core = createDocumentCore({ seedState, openRouterApiKey: '' });
    await core.init();

    const result = await core.handleClientMessage({
      type: 'command',
      clientKey: 'tab-a',
      seq: 1,
      input: {
        actId: 'bread',
        actType: 'task',
        command: 'toggleCollapse',
        payload: {},
        source: 'unit-test'
      }
    });

    expect(result.ack).toMatchObject({
      seq: 1,
      status: 'applied',
      newTarget: 'bread'
    });
    expect(result.state.content.snapshot.items.find((item) => item.id === 'bread').collapsed).toBe(true);
    expect(core.listLog()).toHaveLength(0);
  });

  test('logs unrecognized fallback utterances without changing the snapshot', async () => {
    const core = createDocumentCore({ seedState, openRouterApiKey: '' });
    await core.init();

    const result = await core.handleClientMessage({
      type: 'command',
      clientKey: 'tab-a',
      seq: 1,
      input: {
        actId: 'milk1',
        actType: 'task',
        command: 'logFallbackUtterance',
        payload: { text: 'позвонить Ване' },
        source: 'voice-fallback',
        transcript: 'позвонить Ване'
      }
    });

    expect(result.ack).toMatchObject({
      seq: 1,
      status: 'applied',
      newTarget: 'milk1'
    });
    expect(result.state.rev).toBe(1);
    expect(result.state.content.snapshot.items).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: 'milk1', line1: 'Молоко 3.2%' })
    ]));
    expect(result.state.content.actionLog).toHaveLength(1);
    expect(result.state.content.actionLog[0]).toMatchObject({
      command: expect.objectContaining({ command: 'logFallbackUtterance' }),
      label: 'Нераспознано: позвонить Ване',
      transcript: 'позвонить Ване'
    });
  });

  test('logs LLM fallback failures instead of dropping unrecognized utterances', async () => {
    const core = createDocumentCore({ seedState, openRouterApiKey: '' });
    await core.init();

    const result = await core.handleClientMessage({
      type: 'utterance',
      clientKey: 'tab-a',
      seq: 2,
      target: 'milk1',
      transcript: 'сделай что-нибудь очень странное'
    });

    expect(result.ack).toMatchObject({
      seq: 2,
      status: 'applied',
      newTarget: 'milk1'
    });
    expect(result.state.rev).toBe(1);
    expect(core.listLog()).toHaveLength(1);
    expect(core.listLog()[0]).toMatchObject({
      op: 'logFallbackUtterance',
      transcript: 'сделай что-нибудь очень странное',
      label: 'Нераспознано: сделай что-нибудь очень странное'
    });
  });

  test('exports the full task title tree as text', async () => {
    const core = createDocumentCore({ seedState, openRouterApiKey: '' });
    await core.init();

    const lines = core.getTaskTitleTreeText().trimEnd().split('\n');

    expect(lines[0]).toBe('inbox >> Входящие >> root');
    expect(lines).toContain('bread >> Хлеб ржаной >> root');
    expect(lines).toContain('borod >>   Бородинский >> Хлеб ржаной');
    expect(lines).toContain('first >>         Первый позад >> Воовоага');
  });

  test('returns one task with all attributes and recursive children by id', async () => {
    const core = createDocumentCore({ seedState, openRouterApiKey: '' });
    await core.init();

    const byId = core.getTaskTree({ id: 'bread' });
    expect(byId.status).toBe('found');
    expect(byId.task).toMatchObject({
      id: 'bread',
      parentId: null,
      order: 20,
      status: 'Open',
      line1: 'Хлеб ржаной',
      line2: '',
      collapsed: false,
      tags: [],
      children: [
        expect.objectContaining({ id: 'borod', line1: 'Бородинский', children: [] }),
        expect.objectContaining({ id: 'stoli', line1: 'Столичный', children: [] })
      ]
    });
    expect(core.getTaskTree({ title: 'Хлеб ржаной' })).toEqual({ status: 'missing-query' });
    expect(core.getTaskTree({ q: 'bread' })).toEqual({ status: 'missing-query' });
  });

  test('returns MCP active task tree without archived branches', async () => {
    const initialState = {
      content: {
        snapshot: {
          items: [
            ...seedState.snapshot.items,
            { id: 'hidden', parentId: 'shamp', order: 10, status: 'Open', line1: 'Hidden child', line2: '', collapsed: false, tags: [] }
          ]
        },
        actionLog: []
      },
      log: [],
      rev: 0,
      nextId: 1000,
      clients: {}
    };
    const core = createDocumentCore({ seedState, initialState, openRouterApiKey: '' });
    await core.init();

    const treeText = JSON.stringify(core.getActiveTaskTree());

    expect(treeText).toContain('"id":"bread"');
    expect(treeText).not.toContain('"id":"shamp"');
    expect(treeText).not.toContain('"id":"hidden"');
  });

  test('returns MCP task subgraph path, full task, and direct non-archived children', async () => {
    const initialState = {
      content: {
        snapshot: {
          items: [
            ...seedState.snapshot.items,
            { id: 'oldbread', parentId: 'bread', order: 30, status: 'Archive', line1: 'Old bread', line2: '', collapsed: false, tags: [] }
          ]
        },
        actionLog: []
      },
      log: [],
      rev: 0,
      nextId: 1000,
      clients: {}
    };
    const core = createDocumentCore({ seedState, initialState, openRouterApiKey: '' });
    await core.init();

    const result = core.getTaskSubgraph('bread');

    expect(result.status).toBe('found');
    expect(result.subgraph).toMatchObject({
      path: [],
      task: { id: 'bread', line1: 'Хлеб ржаной', line2: '', parentId: null },
      children: [
        { id: 'borod', title: 'Бородинский', status: 'Open' },
        { id: 'stoli', title: 'Столичный', status: 'Open' }
      ]
    });
    expect(JSON.stringify(result.subgraph.children)).not.toContain('oldbread');
    expect(core.getTaskSubgraph('missing')).toEqual({ status: 'not-found' });
  });

  test('applies server commands directly and supports command-log undo', async () => {
    const core = createDocumentCore({ seedState, openRouterApiKey: '' });
    await core.init();

    const added = await core.applyCommand({
      actId: 'list',
      actType: 'list',
      command: 'addItem',
      payload: { line1: 'MCP task', line2: '' },
      source: 'unit-test'
    }, { message: { clientKey: 'mcp-test', seq: 1 } });
    expect(added.status).toBe('applied');
    expect(core.getTaskById(added.newTarget)).toMatchObject({ line1: 'MCP task', status: 'Open' });

    await core.applyCommand({
      actId: added.newTarget,
      actType: 'task',
      command: 'setStatus',
      payload: { status: 'Archive' },
      source: 'unit-test'
    }, { message: { clientKey: 'mcp-test', seq: 2 } });
    expect(core.getTaskById(added.newTarget).status).toBe('Archive');

    const statusUndo = await core.undoLastAction({ clientKey: 'mcp-test', seq: 3, source: 'unit-test' });
    expect(statusUndo).toMatchObject({
      status: 'applied',
      undone: { command: 'setStatus', id: added.newTarget },
      node: { id: added.newTarget, status: 'Open' }
    });

    const addUndo = await core.undoLastAction({ clientKey: 'mcp-test', seq: 4, source: 'unit-test' });
    expect(addUndo).toMatchObject({
      status: 'applied',
      undone: { command: 'addItem', id: added.newTarget },
      node: null
    });
    expect(core.getTaskById(added.newTarget)).toBeNull();
    expect(await core.undoLastAction({ clientKey: 'mcp-test', seq: 5, source: 'unit-test' })).toEqual({
      status: 'error',
      error: 'нечего откатывать'
    });
  });

  test('compacts stored action-log patches and undo payload snapshots on init', async () => {
    const largePatch = [{ op: 'replace', path: '/snapshot/items', value: seedState.snapshot.items }];
    const initialState = {
      content: {
        snapshot: { items: seedState.snapshot.items },
        actionLog: []
      },
      log: [{
        id: '1',
        rev: 1,
        clientKey: 'tab-a',
        seq: 1,
        op: 'undo',
        target: 'list',
        value: null,
        undo: null,
        undoes: null,
        transcript: null,
        llm_raw: null,
        command: {
          actId: 'list',
          actType: 'list',
          command: 'undo',
          payload: { snapshot: seedState.snapshot },
          source: 'unit-test'
        },
        patch: largePatch,
        label: 'Выполнен undo',
        comments: [],
        at: '2026-08-06T00:00:00.000Z'
      }],
      rev: 1,
      nextId: 1000,
      clients: {}
    };
    const core = createDocumentCore({ seedState, initialState, openRouterApiKey: '' });
    await core.init();

    const [entry] = core.listLog();
    expect(entry.patch).toEqual([]);
    expect(entry.command.payload).toEqual({ snapshot: '[omitted]' });
  });

  test('imports a Workflowy shared tree through a server command', async () => {
    const fetchImpl = async (url, init = {}) => {
      const textUrl = String(url);
      if (textUrl.includes('/s/task-tree/')) {
        return workflowyResponse(
          '<script>var PROJECT_TREE_DATA_URL_PARAMS = {"share_id":"Share.123"};</script>',
          { 'set-cookie': 'sessionid=abc; Path=/; HttpOnly' }
        );
      }
      if (textUrl.includes('/get_initialization_data')) {
        expect(init.headers.Cookie).toContain('sessionid=abc');
        return workflowyResponse({
          projectTreeData: {
            auxiliaryProjectTreeInfos: [{
              rootProject: { id: 'root', nm: 'task tree' }
            }],
            initialMostRecentOperationTransactionId: '42'
          }
        });
      }
      if (textUrl.includes('/get_tree_data/')) {
        expect(textUrl).toContain('ot=42');
        return workflowyResponse({
          items: [
            { id: 'child', prnt: 'root', pr: 10, nm: 'Child task' },
            { id: 'nested', prnt: 'child', pr: 10, nm: 'Nested task' }
          ]
        });
      }
      throw new Error(`Unexpected fetch ${textUrl}`);
    };
    const core = createDocumentCore({ seedState, openRouterApiKey: '', fetchImpl });
    await core.init();

    const result = await core.handleClientMessage({
      type: 'command',
      clientKey: 'tab-a',
      seq: 1,
      input: {
        actId: 'list',
        actType: 'list',
        command: 'importWorkflowy',
        payload: { url: 'https://workflowy.com/s/task-tree/iq43ak7FYqEEO1uO' },
        source: 'settings-import'
      }
    });

    expect(result.ack).toMatchObject({
      seq: 1,
      status: 'applied',
      newTarget: 'rs'
    });
    const importedRoot = result.state.content.snapshot.items.find((item) => item.id === 'rs');
    const importedChild = result.state.content.snapshot.items.find((item) => item.parentId === 'rs');
    const importedNested = result.state.content.snapshot.items.find((item) => item.parentId === importedChild.id);
    expect(importedRoot).toMatchObject({ line1: 'task tree', status: 'Open', parentId: null });
    expect(importedChild).toMatchObject({ line1: 'Child task', status: 'Open' });
    expect(importedNested).toMatchObject({ line1: 'Nested task', status: 'Open' });
    expect(core.listLog()[0]).toMatchObject({
      op: 'importWorkflowyTree',
      label: 'Импортировано дерево Workflowy: task tree'
    });
  });
});
