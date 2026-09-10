import { describe, expect, test } from 'vitest';

import { taskFrontierFromItems } from '../../worker/task-frontier.js';

function task(id, parentId, status, order, line1, deadline = null) {
  return { id, parentId, status, order, line1, deadline };
}

describe('Worker task frontier endpoint data', () => {
  test('returns the endpoint contract sorted by deadline', () => {
    const frontier = taskFrontierFromItems([
      task('project-a', null, 'Open', 10, 'Проект А'),
      task('later', 'project-a', 'Open', 10, 'Позже', '2026-09-10'),
      task('project-b', null, 'Open', 20, 'Проект Б'),
      task('soon', 'project-b', 'Focus', 10, 'Скоро', '2026-09-02'),
      task('root-task', null, 'Open', 30, 'Без срока')
    ], new Date(2026, 8, 1, 12));

    expect(frontier).toEqual([
      {
        parentTitle: 'Проект Б',
        taskId: 'soon',
        taskTitle: 'Скоро',
        status: 'Focus',
        deadline: '2026-09-02'
      },
      {
        parentTitle: 'Проект А',
        taskId: 'later',
        taskTitle: 'Позже',
        status: 'Open',
        deadline: '2026-09-10'
      },
      {
        parentTitle: 'Мой список',
        taskId: 'root-task',
        taskTitle: 'Без срока',
        status: 'Open',
        deadline: null
      }
    ]);
  });
});
