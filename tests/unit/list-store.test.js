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

    expect(state.snapshot.items.map((item) => item.id)).toEqual(['inbox', 'abc12']);
  });

  test('adds inbox when loading legacy state without it', () => {
    window.localStorage.setItem('legacy-list-store', JSON.stringify({
      snapshot: {
        items: [{ id: 'abc12', parentId: null, order: 10, status: 'Open', line1: 'Seed', tags: [], collapsed: false }]
      },
      actionLog: []
    }));
    const store = createStore({
      storageKey: 'legacy-list-store',
      storage: window.localStorage,
      seedState: { snapshot: { items: [] }, actionLog: [] }
    });

    const state = store.load();

    expect(state.snapshot.items[0]).toMatchObject({
      id: 'inbox',
      parentId: null,
      order: 0,
      status: 'Open',
      line1: 'Входящие'
    });
    expect(state.snapshot.items.some((item) => item.id === 'abc12')).toBe(true);
  });
});
