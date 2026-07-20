export function createSync({ adapter, onStateChange, store, autoSaveMs = 60_000 }) {
  let queue = Promise.resolve();
  let autoSaveTimer = null;

  function chain(task) {
    queue = queue.then(task);
    return queue;
  }

  function enqueue(state, actionLogEntry) {
    if (!actionLogEntry) return queue;

    return chain(async () => {
      try {
        const savedState = await adapter.save(state, { reason: 'mutation', createBackup: false, actionLogEntry });
        if (savedState && store.replaceState) store.replaceState(savedState);
        const nextState = store.updateActionLogStatus(actionLogEntry.id, 'synced');
        onStateChange(nextState);
      } catch (error) {
        const nextState = store.updateActionLogStatus(actionLogEntry.id, 'failed');
        onStateChange(nextState);
      }
    });
  }

  function start(getState) {
    if (autoSaveTimer) return;

    autoSaveTimer = setInterval(() => {
      const state = getState();
      if (!state) return;

      chain(async () => {
        try {
          await adapter.save(state, { reason: 'autosave', createBackup: true });
        } catch (error) {
          console.warn('Autosave failed', error);
        }
      });
    }, autoSaveMs);
  }

  function stop() {
    if (!autoSaveTimer) return;
    clearInterval(autoSaveTimer);
    autoSaveTimer = null;
  }

  return { enqueue, start, stop };
}
