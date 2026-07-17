import { describe, expect, test } from 'vitest';
import { createStore } from '../../src/list-store.js';

describe('list store', () => {
  test('loads seeded state when persistent storage is empty', () => {
    const store = createStore({
      storageKey: 'test-list-store',
      storage: window.localStorage,
      seedState: { snapshot: { items: [{ id: 'abc12', parentId: null, order: 10, status: 'Open', line1: 'Seed', tags: [], collapsed: false }] }, actionLog: [] }
    });

    const state = store.load();

    expect(state.snapshot.items).toHaveLength(1);
    expect(state.snapshot.items[0].id).toBe('abc12');
  });
});
