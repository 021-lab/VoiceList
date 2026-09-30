import { describe, expect, test } from 'vitest';
import { TaskGraph } from '../../src/v2/domain/task-graph.js';

/** Сказанная вслух задача должна быть видна сразу: список отсортирован по order, значит
 *  новая задача встаёт над соседями, а не в конец, куда уже не долистать. */
const ordered = (graph, parentId = null) =>
  graph.read().items.filter(item => item.parentId === parentId).sort((a, b) => a.order - b.order).map(item => item.title);

describe('новая задача появляется вверху', () => {
  test('задача в корне встаёт над всеми, включая созданную только что', () => {
    const graph = new TaskGraph();
    graph.apply([{ command: 'addItem', actId: 'list', payload: { title: 'Первая' } }], 0);
    graph.apply([{ command: 'addItem', actId: 'list', payload: { title: 'Вторая' } }], graph.revision);
    expect(ordered(graph)).toEqual(['Вторая', 'Первая', 'Входящие']);
  });

  test('подзадача встаёт над остальными подзадачами своего родителя', () => {
    const graph = new TaskGraph();
    const parent = graph.apply([{ command: 'addItem', actId: 'list', payload: { title: 'Проект' } }], 0).target;
    graph.apply([{ command: 'addChild', actId: parent, payload: { title: 'Старая' } }], graph.revision);
    graph.apply([{ command: 'addChild', actId: parent, payload: { title: 'Новая' } }], graph.revision);
    expect(ordered(graph, parent)).toEqual(['Новая', 'Старая']);
  });

  test('перетаскивание перенумеровывает список, и следующая задача снова оказывается сверху', () => {
    const graph = new TaskGraph();
    graph.apply([{ command: 'addItem', actId: 'list', payload: { title: 'Одна' } }], 0);
    const arranged = graph.read().items.map((item, index) => ({ id: item.id, parentId: item.parentId, order: (index + 1) * 10 }));
    graph.apply([{ command: 'reorderItems', actId: 'list', payload: { arranged } }], graph.revision);
    graph.apply([{ command: 'addItem', actId: 'list', payload: { title: 'Свежая' } }], graph.revision);
    expect(ordered(graph)[0]).toBe('Свежая');
  });
});
