import { describe, expect, test } from 'vitest';
import { seedState } from '../../list-data.js';
import { calculateFrontier } from '../../src/list-frontier.js';

describe('seed data frontier', () => {
  test('places Первый позад into the frontier', () => {
    const result = calculateFrontier(seedState.snapshot.items);

    expect(result.frontier.map((item) => item.line1)).toContain('Первый позад');
  });
});
