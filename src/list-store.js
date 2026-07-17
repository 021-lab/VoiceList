function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function decodePathSegment(segment) {
  return segment.replace(/~1/g, '/').replace(/~0/g, '~');
}

function applyJsonPatch(document, patch) {
  const nextDocument = clone(document);

  for (const operation of patch) {
    const segments = operation.path.split('/').slice(1).map(decodePathSegment);
    const lastSegment = segments.pop();
    let target = nextDocument;

    for (const segment of segments) {
      target = target[Array.isArray(target) ? Number(segment) : segment];
    }

    if (operation.op === 'replace' || operation.op === 'add') {
      target[Array.isArray(target) ? Number(lastSegment) : lastSegment] = clone(operation.value);
    } else if (operation.op === 'remove') {
      if (Array.isArray(target)) target.splice(Number(lastSegment), 1);
      else delete target[lastSegment];
    } else {
      throw new Error(`Unsupported patch op: ${operation.op}`);
    }
  }

  return nextDocument;
}

export function createStore({ storageKey = 'voicelist.universal-list.state', storage = window.localStorage, seedState }) {
  let state = null;

  function persist() {
    storage.setItem(storageKey, JSON.stringify(state));
  }

  function load() {
    const raw = storage.getItem(storageKey);
    state = raw ? JSON.parse(raw) : clone(seedState);
    if (!raw) persist();
    return clone(state);
  }

  function getState() {
    if (!state) return load();
    return clone(state);
  }

  function replaceState(nextState) {
    state = clone(nextState);
    persist();
    return getState();
  }

  function applyMutation({ patch = [], actionLogEntry = null }) {
    if (!state) load();
    state = applyJsonPatch(state, patch);
    if (actionLogEntry) state.actionLog.push(actionLogEntry);
    persist();
    return getState();
  }

  function updateActionLogStatus(logId, syncStatus) {
    if (!state) load();
    state.actionLog = state.actionLog.map((entry) => (
      entry.id === logId ? { ...entry, syncStatus } : entry
    ));
    persist();
    return getState();
  }

  return {
    applyMutation,
    getState,
    load,
    replaceState,
    updateActionLogStatus
  };
}
