import { describe, expect, test, vi } from 'vitest';

import { createSync } from '../../src/list-sync.js';

describe('list sync', () => {
  test('marks action log entry synced after mutation save', async () => {
    const state = { snapshot: { items: [] }, actionLog: [] };
    const adapter = {
      save: vi.fn(async () => state)
    };
    const store = {
      updateActionLogStatus: vi.fn(() => ({ ok: true }))
    };
    const onStateChange = vi.fn();

    const sync = createSync({ adapter, onStateChange, store });
    const actionLogEntry = { id: 'log1' };

    await sync.enqueue(state, actionLogEntry);

    expect(adapter.save).toHaveBeenCalledWith(state, { reason: 'mutation', createBackup: false, actionLogEntry });
    expect(store.updateActionLogStatus).toHaveBeenCalledWith('log1', 'synced');
    expect(onStateChange).toHaveBeenCalledWith({ ok: true });
  });

  test('replaces local state with reconciled backend ids before marking synced', async () => {
    const state = { snapshot: { items: [{ id: 'local' }] }, actionLog: [{ id: 'log1', syncStatus: 'pending' }] };
    const reconciled = { snapshot: { items: [{ id: 't-server' }] }, actionLog: [{ id: 'log1', syncStatus: 'pending' }] };
    const adapter = {
      save: vi.fn(async () => reconciled)
    };
    const store = {
      replaceState: vi.fn((nextState) => nextState),
      updateActionLogStatus: vi.fn(() => ({ ...reconciled, actionLog: [{ id: 'log1', syncStatus: 'synced' }] }))
    };
    const onStateChange = vi.fn();

    const sync = createSync({ adapter, onStateChange, store });
    await sync.enqueue(state, { id: 'log1' });

    expect(store.replaceState).toHaveBeenCalledWith(reconciled);
    expect(store.updateActionLogStatus).toHaveBeenCalledWith('log1', 'synced');
    expect(onStateChange).toHaveBeenLastCalledWith({
      ...reconciled,
      actionLog: [{ id: 'log1', syncStatus: 'synced' }]
    });
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
