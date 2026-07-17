import { describe, expect, test, vi } from 'vitest';

import { createSync } from '../../src/list-sync.js';

describe('list sync', () => {
  test('marks action log entry synced after mutation save', async () => {
    const adapter = {
      save: vi.fn(async () => {})
    };
    const store = {
      updateActionLogStatus: vi.fn(() => ({ ok: true }))
    };
    const onStateChange = vi.fn();

    const sync = createSync({ adapter, onStateChange, store });
    const state = { snapshot: { items: [] }, actionLog: [] };
    const actionLogEntry = { id: 'log1' };

    await sync.enqueue(state, actionLogEntry);

    expect(adapter.save).toHaveBeenCalledWith(state, { reason: 'mutation', createBackup: false });
    expect(store.updateActionLogStatus).toHaveBeenCalledWith('log1', 'synced');
    expect(onStateChange).toHaveBeenCalledWith({ ok: true });
  });

  test('runs autosave every minute with backup snapshots', async () => {
    vi.useFakeTimers();

    const adapter = {
      save: vi.fn(async () => {})
    };
    const store = {
      updateActionLogStatus: vi.fn()
    };
    const onStateChange = vi.fn();
    const getState = vi.fn(() => ({ snapshot: { items: [{ id: 'x' }] }, actionLog: [] }));

    const sync = createSync({ adapter, onStateChange, store, autoSaveMs: 60_000 });
    sync.start(getState);

    await vi.advanceTimersByTimeAsync(60_000);

    expect(getState).toHaveBeenCalled();
    expect(adapter.save).toHaveBeenCalledWith(
      { snapshot: { items: [{ id: 'x' }] }, actionLog: [] },
      { reason: 'autosave', createBackup: true }
    );

    sync.stop();
    vi.useRealTimers();
  });
});
