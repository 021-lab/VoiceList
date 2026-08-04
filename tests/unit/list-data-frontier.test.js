import { describe, expect, test } from 'vitest';
import { seedState } from '../../list-data.js';
import { calculateFrontier } from '../../src/list-frontier.js';

describe('seed data frontier', () => {
  test('keeps paused seed descendants out of the frontier', () => {
    const result = calculateFrontier(seedState.snapshot.items);

    expect(result.frontier.map((item) => item.line1)).toEqual(['Голден', 'Гренни Смит']);
  });
});
