import { describe, expect, test, vi } from 'vitest';

import { createApp } from '../../src/list-app.js';

describe('list app', () => {
  test('creates the log entry before applying the task snapshot patch', async () => {
    const order = [];
    let dispatch = null;
    const renderer = { render: vi.fn() };
    const taskStore = {
      load: vi.fn(() => ({ snapshot: { items: [] } })),
      replaceState: vi.fn((state) => state),
      applyMutation: vi.fn((result) => {
        order.push('apply-mutation');
        expect(result.patch).toHaveLength(1);
        return {
          snapshot: {
            items: [
              { id: 'new01', parentId: null, order: 10, status: 'Open', line1: 'Task from app', line2: '', collapsed: false, tags: [] }
            ]
          }
        };
      }),
      takeLegacyActionLog: vi.fn(() => [])
    };
    const logStore = {
      listEntries: vi.fn(() => []),
      createEntry: vi.fn((entry) => {
        order.push('create-log-entry');
        return entry;
      })
    };
    const sync = {
      enqueueCreate: vi.fn(),
      start: vi.fn()
    };
    const ui = {
      setDispatch: vi.fn((fn) => { dispatch = fn; }),
      setGetState: vi.fn(),
      bindGlobal: vi.fn(),
      openModal: vi.fn()
    };
    const interpreter = {
      execute: vi.fn(() => ({
        patch: [{ op: 'replace', path: '/snapshot/items', value: [] }],
        logEntryDraft: {
          id: 'log-1',
          createdAt: '2026-08-02T10:00:00.000Z',
          transcript: 'добавь задачу',
          command: { command: 'addItem', payload: { line1: 'Task from app', line2: '' } },
          patch: [{ op: 'replace', path: '/snapshot/items', value: [] }],
          syncStatus: 'pending',
          comments: []
        }
      }))
    };

    const app = createApp({
      adapter: { load: vi.fn(async () => null) },
      interpreter,
      renderer,
      store: taskStore,
      logStore,
      sync,
      ui
    });

    await app.init();
    await dispatch({
      actId: 'list',
      actType: 'list',
      command: 'addItem',
      payload: { line1: 'Task from app', line2: '' },
      transcript: 'добавь задачу',
      source: 'voice'
    });

    expect(order).toEqual(['create-log-entry', 'apply-mutation']);
    expect(sync.enqueueCreate).toHaveBeenCalledWith({
      id: 'log-1',
      createdAt: '2026-08-02T10:00:00.000Z',
      transcript: 'добавь задачу',
      command: { command: 'addItem', payload: { line1: 'Task from app', line2: '' } },
      patch: [{ op: 'replace', path: '/snapshot/items', value: [] }],
      syncStatus: 'pending',
      comments: []
    });
  });

  test('sends document commands to the Cloudflare backend without local interpretation', async () => {
    let dispatch = null;
    let stateHandler = null;
    const renderer = { render: vi.fn() };
    const taskStore = {
      load: vi.fn(() => ({ snapshot: { items: [] } })),
      replaceState: vi.fn((state) => state),
      applyMutation: vi.fn(),
      takeLegacyActionLog: vi.fn(() => [])
    };
    const logStore = {
      importLegacyEntries: vi.fn(),
      listEntries: vi.fn(() => []),
      createEntry: vi.fn(),
      updateEntry: vi.fn()
    };
    const documentClient = {
      connect: vi.fn(async () => ({
        rev: 0,
        content: { snapshot: { items: [] }, actionLog: [] }
      })),
      onState: vi.fn((handler) => { stateHandler = handler; }),
      sendCommand: vi.fn(async () => {})
    };
    const ui = {
      setDispatch: vi.fn((fn) => { dispatch = fn; }),
      setGetState: vi.fn(),
      bindGlobal: vi.fn(),
      openModal: vi.fn()
    };
    const interpreter = {
      execute: vi.fn()
    };

    const app = createApp({
      adapter: { load: vi.fn(async () => null) },
      documentClient,
      interpreter,
      renderer,
      store: taskStore,
      logStore,
      sync: {},
      ui
    });

    await app.init();
    expect(stateHandler).toBeTypeOf('function');

    await dispatch({
      actId: 'list',
      actType: 'list',
      command: 'addItem',
      payload: { line1: 'Backend task', line2: '' },
      source: 'unit-test'
    });

    expect(documentClient.sendCommand).toHaveBeenCalledWith({
      actId: 'list',
      actType: 'list',
      command: 'addItem',
      payload: { line1: 'Backend task', line2: '' },
      source: 'unit-test'
    });
    expect(interpreter.execute).not.toHaveBeenCalled();
    expect(taskStore.applyMutation).not.toHaveBeenCalled();

    stateHandler({
      rev: 1,
      content: {
        snapshot: {
          items: [
            { id: 'rs', parentId: null, order: 10, status: 'Open', line1: 'Backend task', line2: '', collapsed: false, tags: [] }
          ]
        },
        actionLog: []
      }
    });
    expect(renderer.render).toHaveBeenLastCalledWith({
      snapshot: {
        items: [
          { id: 'rs', parentId: null, order: 10, status: 'Open', line1: 'Backend task', line2: '', collapsed: false, tags: [] }
        ]
      },
      actionLog: []
    }, 'list', {});
  });
});
