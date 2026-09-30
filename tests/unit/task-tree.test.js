import { describe, expect, test } from 'vitest';

import { taskTreeFromItems } from '../../worker/task-tree.js';

describe('Worker task tree endpoint data', () => {
  test('exports a nested tree with only id, title, status, and children', () => {
    const tree = taskTreeFromItems([
      { id: 'apple', parentId: null, title: 'Яблоки', status: 'Focus', order: 1 },
      { id: 'goldn', parentId: 'apple', title: 'Голден', status: 'Open', order: 2 },
      { id: 'fudji', parentId: 'apple', title: 'Фуджи', status: 'Pause', tags: ['hidden'], order: 3 }
    ]);

    expect(tree).toEqual([{
      id: 'apple',
      title: 'Яблоки',
      status: 'Focus',
      children: [
        { id: 'goldn', title: 'Голден', status: 'Open', children: [] },
        { id: 'fudji', title: 'Фуджи', status: 'Pause', children: [] }
      ]
    }]);
    expect(JSON.stringify(tree)).not.toContain('hidden');
  });
});
