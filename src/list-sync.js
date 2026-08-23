export function createSync({ transport, logStore, onLogEntriesChange }) {
  let queue = Promise.resolve();

  function chain(task) {
    queue = queue.then(task);
    return queue;
  }

  function finalize(id, syncStatus) {
    logStore.updateEntry(id, (entry) => ({ ...entry, syncStatus }));
    onLogEntriesChange?.(logStore.listEntries());
  }

  function enqueueCreate(entry) {
    if (!entry) return queue;
    return chain(async () => {
      try {
        await transport.create(entry);
        finalize(entry.id, 'synced');
      } catch (error) {
        finalize(entry.id, 'failed');
      }
    });
  }

  function enqueueUpdate(entry) {
    if (!entry) return queue;
    return chain(async () => {
      try {
        await transport.update(entry);
        finalize(entry.id, 'synced');
      } catch (error) {
        finalize(entry.id, 'failed');
      }
    });
  }

  return { enqueueCreate, enqueueUpdate };
}
