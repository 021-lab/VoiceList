function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

const INBOX_ITEM = {
  id: 'inbox',
  parentId: null,
  order: 0,
  status: 'Open',
  line1: 'Входящие',
  line2: '',
  collapsed: false,
  tags: []
};

function ensureInbox(nextState) {
  const state = clone(nextState);
  const items = state.snapshot?.items || [];
  if (!items.some((item) => item.id === INBOX_ITEM.id)) {
    state.snapshot.items = [clone(INBOX_ITEM), ...items];
  }
  return state;
}

function normalizeState(nextState) {
  if (nextState?.snapshot?.items) return { snapshot: { items: clone(nextState.snapshot.items) } };
  if (Array.isArray(nextState?.items)) return { snapshot: { items: clone(nextState.items) } };
  return { snapshot: { items: [] } };
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

export function createStore({ storageKey = 'searchmydata.list.state', storage = window.localStorage, seedState }) {
  let state = null;
  let legacyActionLog = null;

  function persist() {
    storage.setItem(storageKey, JSON.stringify(state));
  }

  function load() {
    const raw = storage.getItem(storageKey);
    const parsed = raw ? JSON.parse(raw) : seedState;
    legacyActionLog = Array.isArray(parsed?.actionLog) ? clone(parsed.actionLog) : [];
    state = ensureInbox(normalizeState(parsed));
    persist();
    return clone(state);
  }

  function getState() {
    if (!state) return load();
    return clone(state);
  }

  function replaceState(nextState) {
    state = ensureInbox(normalizeState(nextState));
    persist();
    return getState();
  }

  function applyMutation({ patch = [] }) {
    if (!state) load();
    state = applyJsonPatch(state, patch);
    persist();
    return getState();
  }

  return {
    applyMutation,
    getState,
    load,
    replaceState,
    takeLegacyActionLog() {
      const entries = legacyActionLog || [];
      legacyActionLog = [];
      return clone(entries);
    }
  };
}
