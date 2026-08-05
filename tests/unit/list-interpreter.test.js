import { describe, expect, test } from 'vitest';
import { createInterpreter } from '../../src/list-interpreter.js';

describe('list interpreter', () => {
  test('creates a top-level task with default Open status and log entry draft', () => {
    const interpreter = createInterpreter();
    const state = { snapshot: { items: [] }, actionLog: [] };

    const result = interpreter.execute(state, {
      actId: 'list',
      actType: 'list',
      command: 'addItem',
      payload: { line1: 'Task from test', line2: 'detail' },
      transcript: 'добавь task from test',
      source: 'unit-test'
    });

    expect(result.patch.length).toBeGreaterThan(0);
    expect(result.logEntryDraft.command.command).toBe('addItem');
    expect(result.logEntryDraft.command.payload.line1).toBe('Task from test');
    expect(result.logEntryDraft.transcript).toBe('добавь task from test');
    expect(result.logEntryDraft.comments).toEqual([]);
  });

  test('switches to frontier view without mutating state', () => {
    const interpreter = createInterpreter();
    const state = { snapshot: { items: [] }, actionLog: [] };

    const result = interpreter.execute(state, {
      actId: 'frontier',
      actType: 'tab',
      command: 'showFrontier',
      payload: {},
      source: 'unit-test'
    });

    expect(result).toEqual({ patch: [], logEntryDraft: null, viewMode: 'frontier' });
  });

  test('moves an item to a new parent without creating a cycle', () => {
    const interpreter = createInterpreter();
    const state = {
      snapshot: {
        items: [
          { id: 'a', parentId: null, order: 10, status: 'Open', line1: 'A' },
          { id: 'b', parentId: 'a', order: 10, status: 'Open', line1: 'B' },
          { id: 'c', parentId: null, order: 20, status: 'Open', line1: 'C' }
        ]
      },
      actionLog: []
    };

    const moved = interpreter.execute(state, {
      actId: 'b',
      actType: 'task',
      command: 'setParent',
      payload: { parentId: 'c' },
      source: 'unit-test'
    });
    expect(moved.patch[0].value.find((item) => item.id === 'b').parentId).toBe('c');
    expect(moved.logEntryDraft.command.command).toBe('setParent');

    const rejected = interpreter.execute(state, {
      actId: 'a',
      actType: 'task',
      command: 'setParent',
      payload: { parentId: 'b' },
      source: 'unit-test'
    });
    expect(rejected).toEqual({ patch: [], logEntryDraft: null });
  });

  test('protects inbox and returns search view without action log entry', () => {
    const interpreter = createInterpreter();
    const state = {
      snapshot: {
        items: [
          { id: 'inbox', parentId: null, order: 0, status: 'Open', line1: 'Входящие' },
          { id: 'milk', parentId: 'inbox', order: 10, status: 'Open', line1: 'Купить молоко' },
          { id: 'old', parentId: null, order: 20, status: 'Archive', line1: 'Старое молоко' }
        ]
      },
      actionLog: []
    };

    expect(interpreter.execute(state, {
      actId: 'inbox',
      actType: 'task',
      command: 'editItem',
      payload: { line1: 'Other' },
      source: 'unit-test'
    })).toEqual({ patch: [], logEntryDraft: null });

    const search = interpreter.execute(state, {
      actId: null,
      actType: 'list',
      command: 'showSearch',
      payload: { query: 'молоко' },
      source: 'unit-test'
    });
    expect(search.viewMode).toBe('search');
    expect(search.effect.itemIds).toEqual(['milk']);
    expect(search.logEntryDraft).toBeNull();
  });

  test('does not create a log entry draft for collapse toggles', () => {
    const interpreter = createInterpreter();
    const state = {
      snapshot: {
        items: [
          { id: 'bread', parentId: null, order: 10, status: 'Open', line1: 'Хлеб', collapsed: false, tags: [] }
        ]
      },
      actionLog: []
    };

    const result = interpreter.execute(state, {
      actId: 'bread',
      actType: 'task',
      command: 'toggleCollapse',
      payload: {},
      source: 'unit-test'
    });

    expect(result.patch).toHaveLength(1);
    expect(result.logEntryDraft).toBeNull();
  });

  test('accepts Info as a task status', () => {
    const interpreter = createInterpreter();
    const state = {
      snapshot: {
        items: [
          { id: 'note', parentId: null, order: 10, status: 'Open', line1: 'Reference', collapsed: false, tags: [] }
        ]
      },
      actionLog: []
    };

    const result = interpreter.execute(state, {
      actId: 'note',
      actType: 'task',
      command: 'setStatus',
      payload: { status: 'Info' },
      source: 'unit-test'
    });

    expect(result.patch[0].value.find((item) => item.id === 'note').status).toBe('Info');
    expect(result.logEntryDraft.label).toBe('Статус изменён: Info');
  });

  test('imports a Workflowy tree as Open tasks appended to the list', () => {
    let nextId = 0;
    const interpreter = createInterpreter({
      createItemId(existingIds) {
        let id;
        do {
          id = `wf${nextId}`;
          nextId += 1;
        } while (existingIds.has(id));
        existingIds.add(id);
        return id;
      },
      createLogId: () => 'log-import',
      now: () => new Date('2026-08-05T00:00:00.000Z')
    });
    const state = {
      snapshot: {
        items: [
          { id: 'milk1', parentId: null, order: 10, status: 'Open', line1: 'Молоко', line2: '', collapsed: false, tags: [] }
        ]
      },
      actionLog: []
    };

    const result = interpreter.execute(state, {
      actId: 'list',
      actType: 'list',
      command: 'importWorkflowyTree',
      payload: {
        sourceUrl: 'https://workflowy.com/s/task-tree/iq43ak7FYqEEO1uO',
        tree: {
          title: 'task tree',
          children: [
            { title: 'First', children: [{ title: 'Nested', children: [] }] },
            { title: 'Second', children: [] }
          ]
        }
      },
      source: 'settings-import'
    });

    expect(result.patch).toHaveLength(1);
    expect(result.patch[0].value.slice(1)).toEqual([
      { id: 'wf0', parentId: null, order: 20, status: 'Open', line1: 'task tree', line2: '', collapsed: false, tags: [] },
      { id: 'wf1', parentId: 'wf0', order: 10, status: 'Open', line1: 'First', line2: '', collapsed: false, tags: [] },
      { id: 'wf2', parentId: 'wf1', order: 10, status: 'Open', line1: 'Nested', line2: '', collapsed: false, tags: [] },
      { id: 'wf3', parentId: 'wf0', order: 20, status: 'Open', line1: 'Second', line2: '', collapsed: false, tags: [] }
    ]);
    expect(result.logEntryDraft).toMatchObject({
      id: 'log-import',
      label: 'Импортировано дерево Workflowy: task tree'
    });
  });
});
