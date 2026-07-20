import { describe, expect, test } from 'vitest';
import { createStore } from '../../src/list-store.js';

describe('list store', () => {
  test('loads an empty state when persistent storage is empty', () => {
    const store = createStore({
      storageKey: 'test-list-store',
      storage: window.localStorage
    });

    const state = store.load();

    expect(state).toEqual({ snapshot: { items: [] }, actionLog: [] });
  });
});
