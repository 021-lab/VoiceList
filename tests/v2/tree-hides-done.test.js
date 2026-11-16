import { describe, expect, it } from 'vitest';
import { Presentation } from '../../src/v2/domain/presentation.js';

const task = (id, parentId, status, order = 10) => ({ id, parentId, order, status, title: id, collapsed: false, tags: [] });
const journal = { actions: () => [], cursor: 0 };
const rows = (items) => new Presentation().compose({ revision: 1, items }, journal, { view: 'list' })
  .root.children[1].children.map(row => row.props);

describe('the task tree', () => {
  it('leaves done and archived tasks out', () => {
    const shown = rows([task('open', null, 'Open'), task('done', null, 'Done', 20), task('archived', null, 'Archive', 30)]);
    expect(shown.map(row => row.taskId)).toEqual(['open']);
  });

  it('keeps a done task that still holds open work, so the work stays visible', () => {
    const shown = rows([task('done', null, 'Done'), task('child', 'done', 'Open'), task('gone', 'done', 'Done', 20)]);
    expect(shown.map(row => row.taskId)).toEqual(['done', 'child']);
  });

  it('shows no chevron on a task whose subtasks are all done', () => {
    const [parent] = rows([task('parent', null, 'Open'), task('child', 'parent', 'Done')]);
    expect(parent.hasChildren).toBe(false);
  });
});
