import { describe, expect, test } from 'vitest';
import { compareByDeadline, deadlineDaysFromToday, deadlineFromToday, isDeadline } from '../../src/task-deadline.js';
import { createInterpreter } from '../../src/list-interpreter.js';

describe('task deadlines', () => {
  const now = new Date(2026, 7, 25, 12);

  test('uses local calendar days for the postpone menu', () => {
    expect(deadlineFromToday(0, now)).toBe('2026-08-25');
    expect(deadlineFromToday(7, now)).toBe('2026-09-01');
    expect(deadlineDaysFromToday('2026-08-28', now)).toBe(3);
    expect(isDeadline('2026-02-29')).toBe(false);
  });

  test('keeps scheduled tasks before tasks without a deadline', () => {
    const items = [
      { id: 'none', order: 10 },
      { id: 'week', order: 20, deadline: '2026-09-01' },
      { id: 'tomorrow', order: 30, deadline: '2026-08-26' }
    ];
    expect(items.sort((left, right) => compareByDeadline(left, right, now)).map((item) => item.id)).toEqual(['tomorrow', 'week', 'none']);
  });

  test('stores only a valid deadline in the task document', () => {
    const interpreter = createInterpreter();
    const state = {
      snapshot: { items: [{ id: 'task', parentId: null, order: 10, status: 'Open', line1: 'Task' }] },
      actionLog: []
    };

    const result = interpreter.execute(state, {
      actId: 'task',
      actType: 'task',
      command: 'setDeadline',
      payload: { deadline: '2026-09-01' },
      source: 'unit-test'
    });
    expect(result.patch[0].value[0].deadline).toBe('2026-09-01');

    expect(interpreter.execute(state, {
      actId: 'task',
      actType: 'task',
      command: 'setDeadline',
      payload: { deadline: 'not-a-date' },
      source: 'unit-test'
    })).toEqual({ patch: [], logEntryDraft: null });
  });
});
