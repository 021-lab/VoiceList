import { beforeEach, describe, expect, test } from 'vitest';

import { createLogStore } from '../../src/list-log-store.js';

describe('list log store', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  test('persists each log entry under its own storage key and returns them in index order', () => {
    const store = createLogStore({
      storage: window.localStorage,
      storageKeyPrefix: 'test-log-store'
    });

    store.createEntry({
      id: 'log-a',
      createdAt: '2026-08-02T10:00:00.000Z',
      transcript: 'добавь молоко',
      command: { command: 'addItem', payload: { line1: 'молоко' } },
      patch: [{ op: 'replace', path: '/snapshot/items', value: [] }],
      syncStatus: 'pending',
      comments: []
    });
    store.createEntry({
      id: 'log-b',
      createdAt: '2026-08-02T10:01:00.000Z',
      transcript: null,
      command: { command: 'setStatus', actId: 'milk1', payload: { status: 'Done' } },
      patch: [{ op: 'replace', path: '/snapshot/items', value: [] }],
      syncStatus: 'pending',
      comments: []
    });

    expect(window.localStorage.getItem('test-log-store:index')).toBe(JSON.stringify(['log-a', 'log-b']));
    expect(window.localStorage.getItem('test-log-store:entry:log-a')).toContain('"transcript":"добавь молоко"');
    expect(store.listEntries().map((entry) => entry.id)).toEqual(['log-a', 'log-b']);
  });

  test('imports legacy action log once when separate log storage is empty', () => {
    const store = createLogStore({
      storage: window.localStorage,
      storageKeyPrefix: 'test-log-store'
    });

    store.importLegacyEntries([
      {
        id: 'legacy-1',
        createdAt: '2026-08-01T10:00:00.000Z',
        transcript: null,
        command: { command: 'addItem', payload: { line1: 'из прошлого' } },
        patch: [],
        syncStatus: 'synced',
        comments: []
      }
    ]);
    store.importLegacyEntries([
      {
        id: 'legacy-2',
        createdAt: '2026-08-01T10:01:00.000Z',
        transcript: null,
        command: { command: 'addItem', payload: { line1: 'не должен импортироваться второй раз' } },
        patch: [],
        syncStatus: 'synced',
        comments: []
      }
    ]);

    expect(store.listEntries().map((entry) => entry.id)).toEqual(['legacy-1']);
  });
});
