import { describe, expect, test, vi } from 'vitest';

import { createSync } from '../../src/list-sync.js';

describe('list sync', () => {
  test('marks log entry synced after create transport succeeds', async () => {
    const transport = {
      create: vi.fn(async () => {}),
      update: vi.fn(async () => {})
    };
    const logStore = {
      updateEntry: vi.fn((id, next) => next({ id, syncStatus: 'pending', comments: [] })),
      listEntries: vi.fn(() => [{ id: 'log1', syncStatus: 'synced', comments: [] }])
    };
    const onLogEntriesChange = vi.fn();

    const sync = createSync({ transport, logStore, onLogEntriesChange });
    const actionLogEntry = { id: 'log1', syncStatus: 'pending', comments: [] };

    await sync.enqueueCreate(actionLogEntry);

    expect(transport.create).toHaveBeenCalledWith(actionLogEntry);
    expect(logStore.updateEntry).toHaveBeenCalledWith('log1', expect.any(Function));
    expect(onLogEntriesChange).toHaveBeenCalledWith([{ id: 'log1', syncStatus: 'synced', comments: [] }]);
  });

  test('marks log entry failed after comment update transport rejects', async () => {
    const transport = {
      create: vi.fn(async () => {}),
      update: vi.fn(async () => {
        throw new Error('boom');
      })
    };
    const logStore = {
      updateEntry: vi.fn((id, next) => next({ id, syncStatus: 'pending', comments: [{ id: 'c1', text: 'коммент' }] })),
      listEntries: vi.fn(() => [{ id: 'log1', syncStatus: 'failed', comments: [{ id: 'c1', text: 'коммент' }] }])
    };
    const onLogEntriesChange = vi.fn();

    const sync = createSync({ transport, logStore, onLogEntriesChange });
    const entry = { id: 'log1', syncStatus: 'pending', comments: [{ id: 'c1', text: 'коммент' }] };

    await sync.enqueueUpdate(entry);

    expect(transport.update).toHaveBeenCalledWith(entry);
    expect(onLogEntriesChange).toHaveBeenCalledWith([{ id: 'log1', syncStatus: 'failed', comments: [{ id: 'c1', text: 'коммент' }] }]);
  });
});
