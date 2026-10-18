import { expect, it } from 'vitest';
import { TaskGraph } from '../../src/v2/domain/task-graph.js';

/** Случай с прода: модель прислала editItem без нового названия, и пользователь увидел
 *  «Некорректная задача» — ни он, ни агент исправлений не могли понять, что не так. */
it('переименование без нового названия отказывает понятным сообщением', () => {
  const graph = new TaskGraph({}, { snapshot: { items: [] } });
  graph.apply([{ command: 'addItem', actId: 'list', payload: { title: 'Молоко' } }]);
  const id = graph.read().items.find(item => item.title === 'Молоко').id;
  for (const command of [{ command: 'editItem', actId: id, title: 'Кефир' }, { command: 'editItem', actId: id, payload: {} }]) {
    expect(() => graph.apply([command])).toThrow('В команде переименования нет нового названия');
  }
  expect(graph.read().items.find(item => item.id === id).title).toBe('Молоко');
});
