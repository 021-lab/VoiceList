import { describe, expect, test } from 'vitest';

import { taskTreeFromItems } from '../../worker/task-tree.js';

describe('Worker task tree endpoint data', () => {
  test('exports a nested tree with only id, title, status, and children', () => {
    const tree = taskTreeFromItems([
      { id: 'apple', parentId: null, line1: 'Яблоки', line2: 'hidden', status: 'Focus', order: 1 },
      { id: 'goldn', parentId: 'apple', line1: 'Голден', status: 'Open', order: 2 },
      { id: 'fudji', parentId: 'apple', line1: 'Фуджи', status: 'Pause', tags: ['hidden'], order: 3 }
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
