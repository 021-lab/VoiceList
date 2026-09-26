/** Private persistence adapter. No journal-sized KV value or public storage schema. */
export class RuntimeStorage {
  constructor(storage) {
    this.storage = storage;
    this.sql = storage.sql;
    this.sql.exec('CREATE TABLE IF NOT EXISTS vl_projection (id TEXT PRIMARY KEY, value TEXT NOT NULL)');
    this.sql.exec('CREATE TABLE IF NOT EXISTS vl_journal (cursor INTEGER PRIMARY KEY, value TEXT NOT NULL)');
    this.sql.exec('CREATE TABLE IF NOT EXISTS vl_meta (id TEXT PRIMARY KEY, value INTEGER NOT NULL)');
    this.sql.exec('CREATE TABLE IF NOT EXISTS vl_technical (id TEXT PRIMARY KEY, value TEXT NOT NULL)');
    // What the database already holds, as it was last written. Renaming one task used to
    // rewrite every task and every journal entry — hundreds of statements twice per command,
    // because each transaction saved the whole document. Comparing against this is what turns
    // a save into the handful of rows that actually changed.
    this.written = { items: new Map(), entries: new Map(), technical: null, revision: null, nextId: null };
  }

  load() {
    const meta = this.sql.exec("SELECT value FROM vl_meta WHERE id = 'revision'").toArray()[0];
    if (!meta) return undefined;
    const technical = this.sql.exec("SELECT value FROM vl_technical WHERE id = 'state'").toArray()[0];
    const nextId = this.sql.exec("SELECT value FROM vl_meta WHERE id = 'nextId'").toArray()[0];
    const items = [];
    for (const row of this.sql.exec('SELECT id, value FROM vl_projection ORDER BY rowid').toArray()) {
      this.written.items.set(row.id, row.value);
      items.push(JSON.parse(row.value));
    }
    const entries = [];
    for (const row of this.sql.exec('SELECT cursor, value FROM vl_journal ORDER BY cursor').toArray()) {
      this.written.entries.set(row.cursor, row.value);
      entries.push(JSON.parse(row.value));
    }
    this.written.technical = technical?.value ?? null;
    this.written.revision = meta.value;
    this.written.nextId = nextId?.value ?? null;
    return {
      graph: { revision: meta.value, nextId: nextId?.value, items },
      entries,
      technical: technical ? JSON.parse(technical.value) : undefined
    };
  }

  async save(state) {
    // Serialised first, applied second, remembered third: the record of what the database
    // holds is updated only once the transaction has committed, so a failed write cannot
    // leave this thinking a row is saved when it is not.
    const items = state.graph.items.map(item => [item.id, JSON.stringify(item)]);
    const entries = state.entries.map(entry => [entry.cursor, JSON.stringify(entry)]);
    const technical = JSON.stringify(state.technical);
    const liveIds = new Set(items.map(([id]) => id));
    const liveCursors = new Set(entries.map(([cursor]) => cursor));

    const changedItems = items.filter(([id, value]) => this.written.items.get(id) !== value);
    const droppedItems = [...this.written.items.keys()].filter(id => !liveIds.has(id));
    const changedEntries = entries.filter(([cursor, value]) => this.written.entries.get(cursor) !== value);
    const droppedEntries = [...this.written.entries.keys()].filter(cursor => !liveCursors.has(cursor));
    const changedTechnical = technical !== this.written.technical;
    const changedRevision = state.graph.revision !== this.written.revision;
    const changedNextId = Number.isFinite(state.graph.nextId) && state.graph.nextId !== this.written.nextId;

    if (!changedItems.length && !droppedItems.length && !changedEntries.length &&
        !droppedEntries.length && !changedTechnical && !changedRevision && !changedNextId) return;

    this.storage.transactionSync(() => {
      for (const id of droppedItems) this.sql.exec('DELETE FROM vl_projection WHERE id = ?', id);
      for (const [id, value] of changedItems) {
        this.sql.exec('INSERT INTO vl_projection (id,value) VALUES (?,?) ON CONFLICT(id) DO UPDATE SET value=excluded.value', id, value);
      }
      if (changedRevision) this.sql.exec("INSERT INTO vl_meta (id,value) VALUES ('revision',?) ON CONFLICT(id) DO UPDATE SET value=excluded.value", state.graph.revision);
      if (changedNextId) this.sql.exec("INSERT INTO vl_meta (id,value) VALUES ('nextId',?) ON CONFLICT(id) DO UPDATE SET value=excluded.value", state.graph.nextId);
      for (const cursor of droppedEntries) this.sql.exec('DELETE FROM vl_journal WHERE cursor = ?', cursor);
      for (const [cursor, value] of changedEntries) {
        this.sql.exec('INSERT INTO vl_journal (cursor,value) VALUES (?,?) ON CONFLICT(cursor) DO UPDATE SET value=excluded.value', cursor, value);
      }
      if (changedTechnical) this.sql.exec("INSERT INTO vl_technical (id,value) VALUES ('state',?) ON CONFLICT(id) DO UPDATE SET value=excluded.value", technical);
    });

    for (const id of droppedItems) this.written.items.delete(id);
    for (const [id, value] of changedItems) this.written.items.set(id, value);
    for (const cursor of droppedEntries) this.written.entries.delete(cursor);
    for (const [cursor, value] of changedEntries) this.written.entries.set(cursor, value);
    if (changedTechnical) this.written.technical = technical;
    if (changedRevision) this.written.revision = state.graph.revision;
    if (changedNextId) this.written.nextId = state.graph.nextId;

    await this.storage.sync();
  }

  clear() {
    this.storage.transactionSync(() => {
      this.sql.exec('DELETE FROM vl_projection'); this.sql.exec('DELETE FROM vl_journal'); this.sql.exec('DELETE FROM vl_meta'); this.sql.exec('DELETE FROM vl_technical');
    });
    this.written = { items: new Map(), entries: new Map(), technical: null, revision: null, nextId: null };
  }
}
