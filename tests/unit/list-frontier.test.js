import { describe, expect, test } from 'vitest';
import { calculateFrontier } from '../../src/list-frontier.js';

function task(id, parentId, status, order = 10) {
  return { id, parentId, status, order, line1: id, line2: '', tags: [], collapsed: false };
}

describe('task frontier', () => {
  test('returns the most specific active tasks from focused and paused branches', () => {
    const result = calculateFrontier([
      task('A', null, 'Open', 10),
      task('A1', 'A', 'Open', 10),
      task('A1.1', 'A1', 'Focus', 10),
      task('A1.1.1', 'A1.1', 'Done', 10),
      task('A1.1.2', 'A1.1', 'Open', 20),
      task('A1.2', 'A1', 'Open', 20),
      task('A1.2.1', 'A1.2', 'Open', 10),
      task('A2', 'A', 'Open', 20),
      task('A2.1', 'A2', 'Done', 10),
      task('A2.2', 'A2', 'Archive', 20),
      task('D', null, 'Open', 20),
      task('D1', 'D', 'Pause', 10),
      task('D1.1', 'D1', 'Focus', 10),
      task('D1.1.1', 'D1.1', 'Open', 10),
      task('D1.1.2', 'D1.1', 'Archive', 20),
      task('D1.2', 'D1', 'Open', 20),
      task('D1.2.1', 'D1.2', 'Open', 10),
      task('D2', 'D', 'Open', 20),
      task('E', null, 'Pause', 30),
      task('E1', 'E', 'Open', 10),
      task('F', null, 'Done', 40),
      task('G', null, 'Archive', 50)
    ]);

    expect(result.frontier.map((item) => item.id)).toEqual(['A1.1.2', 'A2', 'D1.1', 'D1.1.1', 'D2']);
    expect(result.focusHighlights.map((item) => item.id)).toEqual(['A1.1', 'D1.1']);
  });

  test('uses all focused siblings and blocks unfocused sibling branches', () => {
    const result = calculateFrontier([
      task('P', null, 'Open'),
      task('A', 'P', 'Focus', 10),
      task('A1', 'A', 'Open', 10),
      task('B', 'P', 'Focus', 20),
      task('B1', 'B', 'Open', 10),
      task('C', 'P', 'Open', 30),
      task('C1', 'C', 'Open', 10)
    ]);

    expect(result.frontier.map((item) => item.id)).toEqual(['A1', 'B1']);
    expect(result.focusHighlights.map((item) => item.id)).toEqual(['A', 'B']);
  });

  test('allows focus below closed ancestors while keeping closed ancestors out of frontier', () => {
    const result = calculateFrontier([
      task('apple', null, 'Open', 10),
      task('fudji', 'apple', 'Pause', 10),
      task('done-parent', 'fudji', 'Done', 10),
      task('open-bridge', 'done-parent', 'Open', 10),
      task('first-backlog', 'open-bridge', 'Focus', 10)
    ]);

    expect(result.frontier.map((item) => item.id)).toEqual(['first-backlog']);
    expect(result.focusHighlights.map((item) => item.id)).toEqual(['first-backlog']);
  });

  test('throws on missing parent and duplicate task ids', () => {
    expect(() => calculateFrontier([task('A', 'missing', 'Open')])).toThrow(/parent/i);
    expect(() => calculateFrontier([
      task('A', null, 'Open'),
      task('A', null, 'Open')
    ])).toThrow(/duplicate/i);
  });

  test('throws when tasks are not reachable from a root', () => {
    expect(() => calculateFrontier([
      task('root', null, 'Open'),
      task('A', 'B', 'Open'),
      task('B', 'A', 'Open')
    ])).toThrow(/cycle|reachable/i);
  });
});
