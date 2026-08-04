function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function createCommentRecord(text) {
  return {
    id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 5)}`.slice(0, 8),
    createdAt: new Date().toISOString(),
    text
  };
}

function isLocalUiCommand(input) {
  return input.command === 'showActionLog' ||
    input.command === 'showList' ||
    input.command === 'showFrontier' ||
    input.command === 'showSearch' ||
    input.command === 'showAddModal' ||
    input.command === 'showEditModal' ||
    input.command === 'showNestModal' ||
    input.command === 'viewItem';
}

function unpackServerState(serverState) {
  if (!serverState?.content) return null;
  return {
    snapshot: clone(serverState.content.snapshot || { items: [] }),
    actionLog: clone(serverState.content.actionLog || [])
  };
}

export function createApp({ adapter, documentClient = null, interpreter, renderer, store, logStore, sync, ui }) {
  let state = null;
  let snapshotState = null;
  let actionLog = [];
  let viewMode = 'list';
  let viewContext = {};

  function composeState(nextSnapshot = snapshotState, nextActionLog = actionLog) {
    return {
      ...clone(nextSnapshot),
      actionLog: clone(nextActionLog)
    };
  }

  function refreshState(nextSnapshot = snapshotState, nextActionLog = actionLog) {
    snapshotState = clone(nextSnapshot);
    actionLog = clone(nextActionLog);
    state = composeState(snapshotState, actionLog);
    return state;
  }

  function render() {
    renderer.render(state, viewMode, viewContext);
  }

  function applyServerState(serverState) {
    const unpacked = unpackServerState(serverState);
    if (!unpacked) return;
    refreshState({ snapshot: unpacked.snapshot }, unpacked.actionLog);
    render();
  }

  function handleEffect(effect) {
    if (!effect) return;
    if (effect.type === 'modal') ui.openModal(effect.mode, effect.itemId, state);
  }

  async function appendLogComment(input) {
    const text = String(input.payload?.text || '').trim();
    if (!text) return;
    const updatedEntry = logStore.updateEntry(input.actId, (entry) => ({
      ...entry,
      comments: [...(entry.comments || []), createCommentRecord(text)],
      syncStatus: 'pending'
    }));
    if (!updatedEntry) return;
    refreshState(snapshotState, logStore.listEntries());
    render();
    await sync.enqueueUpdate?.(updatedEntry);
  }

  async function dispatchUserInput(input) {
    if (documentClient && !isLocalUiCommand(input)) {
      await documentClient.sendCommand(input);
      return;
    }

    if (input.command === 'commentLogEntry') {
      await appendLogComment(input);
      return;
    }

    const result = interpreter.execute(state, input);

    if (result.viewMode) {
      viewMode = result.viewMode;
      viewContext = result.effect || {};
      render();
      return;
    }

    if (result.effect) {
      handleEffect(result.effect);
      return;
    }

    if (result.patch?.length || result.logEntryDraft) {
      let persistedLogEntry = null;
      if (result.logEntryDraft) {
        persistedLogEntry = logStore.createEntry(result.logEntryDraft);
        actionLog = logStore.listEntries();
      }
      if (result.patch?.length) {
        snapshotState = store.applyMutation({ patch: result.patch });
      }
      refreshState(snapshotState, actionLog);
      render();
      if (persistedLogEntry) await sync.enqueueCreate?.(persistedLogEntry);
    }
  }

  return {
    async init() {
      if (documentClient) {
        snapshotState = { snapshot: { items: [] } };
        actionLog = [];
        refreshState(snapshotState, actionLog);

        documentClient.onState?.(applyServerState);
        const serverState = await documentClient.connect();
        if (serverState) applyServerState(serverState);

        ui.setDispatch(dispatchUserInput);
        ui.setGetState(() => state);
        ui.bindGlobal();
        if (!serverState) render();
        return;
      }

      snapshotState = store.load();
      logStore.importLegacyEntries?.(store.takeLegacyActionLog?.() || []);
      actionLog = logStore.listEntries();
      const backendState = await adapter.load();
      if (backendState) snapshotState = store.replaceState(backendState);
      refreshState(snapshotState, actionLog);

      ui.setDispatch(dispatchUserInput);
      ui.setGetState(() => state);
      ui.bindGlobal();
      render();
    },
    getState() {
      return state;
    },
    onLogEntriesChange(nextActionLog) {
      refreshState(snapshotState, nextActionLog);
      render();
    }
  };
}
