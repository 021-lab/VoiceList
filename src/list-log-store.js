function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function normalizeEntry(entry) {
  return {
    ...clone(entry),
    transcript: entry?.transcript ?? null,
    comments: Array.isArray(entry?.comments) ? clone(entry.comments) : []
  };
}

export function createLogStore({ storageKeyPrefix = 'searchmydata.list-interface.log', storage = window.localStorage } = {}) {
  const indexKey = `${storageKeyPrefix}:index`;
  const entryKey = (id) => `${storageKeyPrefix}:entry:${id}`;

  function readIndex() {
    const raw = storage.getItem(indexKey);
    return raw ? JSON.parse(raw) : [];
  }

  function writeIndex(ids) {
    storage.setItem(indexKey, JSON.stringify(ids));
  }

  function getEntry(id) {
    const raw = storage.getItem(entryKey(id));
    return raw ? normalizeEntry(JSON.parse(raw)) : null;
  }

  function listEntries() {
    return readIndex()
      .map((id) => getEntry(id))
      .filter(Boolean);
  }

  function persistEntry(entry) {
    const normalized = normalizeEntry(entry);
    const ids = readIndex();
    if (!ids.includes(normalized.id)) {
      ids.push(normalized.id);
      writeIndex(ids);
    }
    storage.setItem(entryKey(normalized.id), JSON.stringify(normalized));
    return clone(normalized);
  }

  function createEntry(entry) {
    return persistEntry(entry);
  }

  function updateEntry(id, next) {
    const current = getEntry(id);
    if (!current) return null;
    const value = typeof next === 'function' ? next(clone(current)) : next;
    return persistEntry(value);
  }

  function importLegacyEntries(entries = []) {
    if (readIndex().length || !entries.length) return listEntries();
    for (const entry of entries) persistEntry(entry);
    return listEntries();
  }

  return {
    createEntry,
    getEntry,
    importLegacyEntries,
    listEntries,
    updateEntry
  };
}
