/** Private persistence adapter. No journal-sized KV value or public storage schema. */
export class RuntimeStorage {
  constructor(storage) {
    this.storage = storage;
    this.sql = storage.sql;
    this.sql.exec('CREATE TABLE IF NOT EXISTS vl_projection (id TEXT PRIMARY KEY, value TEXT NOT NULL)');
    this.sql.exec('CREATE TABLE IF NOT EXISTS vl_journal (cursor INTEGER PRIMARY KEY, value TEXT NOT NULL)');
    this.sql.exec('CREATE TABLE IF NOT EXISTS vl_meta (id TEXT PRIMARY KEY, value INTEGER NOT NULL)');
    this.sql.exec('CREATE TABLE IF NOT EXISTS vl_technical (id TEXT PRIMARY KEY, value TEXT NOT NULL)');
  }
  load() {
    const meta = this.sql.exec("SELECT value FROM vl_meta WHERE id = 'revision'").toArray()[0];
    if (!meta) return undefined;
    const technical = this.sql.exec("SELECT value FROM vl_technical WHERE id = 'state'").toArray()[0];
    return {
      graph: { revision: meta.value, items: this.sql.exec('SELECT value FROM vl_projection ORDER BY rowid').toArray().map(row => JSON.parse(row.value)) },
      entries: this.sql.exec('SELECT value FROM vl_journal ORDER BY cursor').toArray().map(row => JSON.parse(row.value)),
      technical: technical ? JSON.parse(technical.value) : undefined
    };
  }
  async save(state) {
    this.storage.transactionSync(() => {
      const ids = new Set(state.graph.items.map(item => item.id));
      for (const row of this.sql.exec('SELECT id FROM vl_projection').toArray()) {
        if (!ids.has(row.id)) this.sql.exec('DELETE FROM vl_projection WHERE id = ?', row.id);
      }
      for (const item of state.graph.items) this.sql.exec('INSERT INTO vl_projection (id,value) VALUES (?,?) ON CONFLICT(id) DO UPDATE SET value=excluded.value WHERE value != excluded.value', item.id, JSON.stringify(item));
      this.sql.exec("INSERT INTO vl_meta (id,value) VALUES ('revision',?) ON CONFLICT(id) DO UPDATE SET value=excluded.value WHERE value != excluded.value", state.graph.revision);
      const cursors = new Set(state.entries.map(entry => entry.cursor));
      for (const row of this.sql.exec('SELECT cursor FROM vl_journal').toArray()) if (!cursors.has(row.cursor)) this.sql.exec('DELETE FROM vl_journal WHERE cursor = ?', row.cursor);
      for (const entry of state.entries) this.sql.exec('INSERT INTO vl_journal (cursor,value) VALUES (?,?) ON CONFLICT(cursor) DO UPDATE SET value=excluded.value WHERE value != excluded.value', entry.cursor, JSON.stringify(entry));
      this.sql.exec("INSERT INTO vl_technical (id,value) VALUES ('state',?) ON CONFLICT(id) DO UPDATE SET value=excluded.value WHERE value != excluded.value", JSON.stringify(state.technical));
    });
    await this.storage.sync();
  }
  clear() {
    this.storage.transactionSync(() => {
      this.sql.exec('DELETE FROM vl_projection'); this.sql.exec('DELETE FROM vl_journal'); this.sql.exec('DELETE FROM vl_meta'); this.sql.exec('DELETE FROM vl_technical');
    });
  }
}
