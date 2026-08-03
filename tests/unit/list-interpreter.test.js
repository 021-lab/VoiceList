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
});
