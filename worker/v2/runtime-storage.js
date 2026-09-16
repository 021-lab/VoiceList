/** Private persistence adapter. No journal-sized KV value or public storage schema. */
export class RuntimeStorage {
  constructor(storage) {
    this.storage = storage;
    this.sql = storage.sql;
    this.sql.exec('CREATE TABLE IF NOT EXISTS vl_projection (id TEXT PRIMARY KEY, value TEXT NOT NULL)');
    this.sql.exec('CREATE TABLE IF NOT EXISTS vl_journal (cursor INTEGER PRIMARY KEY, value TEXT NOT NULL)');
    this.sql.exec('CREATE TABLE IF NOT EXISTS vl_meta (id TEXT PRIMARY KEY, value INTEGER NOT NULL)');
  }
  load() {
    const meta = this.sql.exec("SELECT value FROM vl_meta WHERE id = 'revision'").toArray()[0];
    if (!meta) return undefined;
    return {
      graph: { revision: meta.value, items: this.sql.exec('SELECT value FROM vl_projection ORDER BY rowid').toArray().map(row => JSON.parse(row.value)) },
      entries: this.sql.exec('SELECT value FROM vl_journal ORDER BY cursor').toArray().map(row => JSON.parse(row.value))
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
      const last = this.sql.exec('SELECT COALESCE(MAX(cursor),0) AS cursor FROM vl_journal').toArray()[0].cursor;
      for (const entry of state.entries) if (entry.cursor > last) this.sql.exec('INSERT INTO vl_journal (cursor,value) VALUES (?,?)', entry.cursor, JSON.stringify(entry));
    });
    await this.storage.sync();
  }
  clear() {
    this.storage.transactionSync(() => {
      this.sql.exec('DELETE FROM vl_projection'); this.sql.exec('DELETE FROM vl_journal'); this.sql.exec('DELETE FROM vl_meta');
    });
  }
}
