import { describe, expect, it } from 'vitest';
import { TaskGraph, createIdAllocator, FIRST_TASK_ID_SEQUENCE } from '../../src/v2/domain/task-graph.js';

const emptySeed = { snapshot: { items: [] } };
const addItem = (line1) => ({ command: 'addItem', actId: 'list', payload: { line1 } });

describe('base36 task ids', () => {
  it('starts the sequence at 1000 rendered in base36', () => {
    const graph = new TaskGraph({}, emptySeed);
    graph.apply([addItem('Молоко')]);
    const created = graph.read().items.find(item => item.line1 === 'Молоко');
    expect(created.id).toBe(FIRST_TASK_ID_SEQUENCE.toString(36));
    expect(created.id).toBe('rs');
  });

  it('keeps ids short and increasing across commands', () => {
    const graph = new TaskGraph({}, emptySeed);
    graph.apply([addItem('Первая')]);
    graph.apply([addItem('Вторая')]);
    const ids = graph.read().items.filter(item => item.id !== 'inbox').map(item => item.id);
    expect(ids).toEqual(['rs', 'rt']);
    expect(ids.every(id => id.length <= 4)).toBe(true);
  });

  it('carries the counter through a persistence round trip', () => {
    const first = new TaskGraph({}, emptySeed);
    first.apply([addItem('Первая')]);
    const stored = first.read();
    expect(stored.nextId).toBe(FIRST_TASK_ID_SEQUENCE + 1);

    const restored = new TaskGraph(stored, emptySeed);
    restored.apply([addItem('Вторая')]);
    expect(restored.read().items.find(item => item.line1 === 'Вторая').id).toBe('rt');
  });

  it('skips a candidate already taken by a legacy id', () => {
    const graph = new TaskGraph({}, {
      snapshot: { items: [{ id: 'rs', parentId: null, order: 10, status: 'Open', line1: 'Наследная', line2: '', collapsed: false, tags: [] }] }
    });
    graph.apply([addItem('Новая')]);
    expect(graph.read().items.find(item => item.line1 === 'Новая').id).toBe('rt');
  });

  it('never rewinds the counter when an action is rolled back', () => {
    const graph = new TaskGraph({}, emptySeed);
    const outcome = graph.apply([addItem('Ошибочная')]);
    graph.rollback([{ changes: outcome.changes }]);
    expect(graph.read().items.some(item => item.line1 === 'Ошибочная')).toBe(false);
    graph.apply([addItem('Следующая')]);
    expect(graph.read().items.find(item => item.line1 === 'Следующая').id).toBe('rt');
  });

  it('never hands out the reserved inbox id', () => {
    // 27_436 is "inbox" in base36; the allocator must step over it.
    const allocator = createIdAllocator(Number.parseInt('inbox', 36));
    expect(allocator.allocate(new Set())).not.toBe('inbox');
  });
});
