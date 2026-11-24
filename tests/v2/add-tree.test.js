import { describe, expect, test } from 'vitest';
import { TaskGraph } from '../../src/v2/domain/task-graph.js';
import { toTaskCommand, LIVE_TOOL_NAMES } from '../../src/v2/domain/live-session.js';
import { confirmText } from '../../src/v2/domain/live-delegation.js';

const children = (graph, parentId) => graph.read().items.filter(item => item.parentId === parentId).sort((a, b) => a.order - b.order);

describe('задача сразу с подзадачами', () => {
  test('один вызов создаёт цель во «Входящих» и шаги внутри неё, без ожидания id', () => {
    const graph = new TaskGraph();
    const command = toTaskCommand('addTaskWithSubtasks', { parentId: 'inbox', title: 'Послать деньги Маше', subtasks: ['Разблокировать аккаунт', 'Проверить лимит'] });
    const result = graph.apply([command], 0);

    const goal = graph.read().items.find(item => item.id === result.target);
    expect(goal).toMatchObject({ title: 'Послать деньги Маше', parentId: 'inbox', status: 'Open' });
    expect(children(graph, goal.id).map(item => item.title)).toEqual(['Разблокировать аккаунт', 'Проверить лимит']);
  });

  test('без родителя цель встаёт в корень наверх — и при null, и при "list"', () => {
    const graph = new TaskGraph();
    graph.apply([toTaskCommand('addTaskWithSubtasks', { parentId: null, title: 'Заказать шлем', subtasks: ['Измерить голову'] })], 0);
    expect(children(graph, null)[0].title).toBe('Заказать шлем');
    graph.apply([{ command: 'addTree', actId: 'list', actType: 'list', payload: { title: 'Корневая', subtasks: ['Шаг'] } }], graph.revision);
    expect(children(graph, null)[0].title).toBe('Корневая');
  });

  test('одна отмена убирает и цель, и шаги', () => {
    const graph = new TaskGraph();
    const before = graph.read().items.length;
    const result = graph.apply([toTaskCommand('addTaskWithSubtasks', { parentId: 'inbox', title: 'Цель', subtasks: ['Шаг'] })], 0);
    expect(graph.read().items.length).toBe(before + 2);
    graph.rollback([result]);
    expect(graph.read().items.length).toBe(before);
  });

  test('без подзадач и под несуществующей задачей — отказ', () => {
    expect(() => toTaskCommand('addTaskWithSubtasks', { parentId: 'inbox', title: 'Цель', subtasks: [] })).toThrow('Нет подзадач');
    const graph = new TaskGraph();
    expect(() => graph.apply([toTaskCommand('addTaskWithSubtasks', { parentId: 'нет', title: 'Цель', subtasks: ['Шаг'] })], 0)).toThrow('Задача не найдена');
  });

  test('голос подтверждает коротко и знает этот инструмент', () => {
    const command = toTaskCommand('addTaskWithSubtasks', { parentId: 'inbox', title: 'Послать деньги Маше', subtasks: ['Разблокировать аккаунт'] });
    expect(confirmText(command, [])).toBe('Добавил: Послать деньги Маше, внутри Разблокировать аккаунт');
    const two = toTaskCommand('addTaskWithSubtasks', { parentId: 'inbox', title: 'Цель', subtasks: ['Шаг', 'Ещё шаг'] });
    expect(confirmText(two, [])).toBe('Добавил: Цель, внутри Шаг и ещё 1');
    expect(LIVE_TOOL_NAMES).toContain('addTaskWithSubtasks');
  });
});

describe('через весь серверный путь', () => {
  test('голосовой вызов проходит порт, журнал и граф и отдаёт id цели', async () => {
    const { DocumentRuntime } = await import('../../src/v2/domain/document-runtime.js');
    const { CompatibilityPort } = await import('../../worker/v2/compatibility.js');
    const runtime = new DocumentRuntime(), port = new CompatibilityPort(runtime);
    const command = toTaskCommand('addTaskWithSubtasks', { parentId: 'inbox', title: 'Послать деньги Маше', subtasks: ['Разблокировать аккаунт'] }, 'gemini-live');
    const ack = await port.applyCommand(command, { message: { clientKey: 'gemini:gm_1:c1', seq: 1 } });

    expect(ack.status).toBe('applied');
    const goal = runtime.graph.read().items.find(item => item.id === ack.newTarget);
    expect(goal.title).toBe('Послать деньги Маше');
    expect(runtime.graph.read().items.find(item => item.parentId === goal.id).title).toBe('Разблокировать аккаунт');
  });
});
