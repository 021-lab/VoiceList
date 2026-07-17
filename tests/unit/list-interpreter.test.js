import { describe, expect, test } from 'vitest';
import { createInterpreter } from '../../src/list-interpreter.js';

describe('list interpreter', () => {
  test('creates a top-level task with default Open status and action log entry', () => {
    const interpreter = createInterpreter();
    const state = { snapshot: { items: [] }, actionLog: [] };

    const result = interpreter.execute(state, {
      actId: 'list',
      actType: 'list',
      command: 'addItem',
      payload: { line1: 'Task from test', line2: 'detail' },
      source: 'unit-test'
    });

    expect(result.patch.length).toBeGreaterThan(0);
    expect(result.actionLogEntry.command.command).toBe('addItem');
    expect(result.actionLogEntry.command.payload.line1).toBe('Task from test');
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

    expect(result).toEqual({ patch: [], actionLogEntry: null, viewMode: 'frontier' });
  });
});
