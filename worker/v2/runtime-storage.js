/** Private persistence adapter. No journal-sized KV value or public storage schema. */

/** The ledgers, split into rows that can change independently.
 *
 *  They used to be one row holding the whole of `technical`. On a real document that row was
 *  640 KB, and every transaction rewrote it whole — twice per command, whatever the command
 *  touched. Almost all of it belongs to one journal entry each, so it is stored that way: a
 *  command writes the entry it is about and the small shared row, and leaves the rest alone. */
const CORE_ROW = 'core';
const LEGACY_ROW = 'state';
const entryRow = (id) => `e:${id}`;
const eventRow = (cursor) => `ev:${cursor}`;

/** Text of a row, reused when the records behind it are the same objects as last time.
 *
 *  Encoding is the cost, not writing: a save that re-encoded every task, entry and ledger in
 *  order to find the changed ones grew with the document, which is what made adding a task
 *  slower the more there already was. The runtime replaces a record it changes rather than
 *  editing it, so sameness of object is sameness of content. */
class RowText {
  constructor() { this.seen = new Map(); }
  of(id, record) { return this.encode(id, [record], () => JSON.stringify(record)); }
  ofLedgers(id, harness, executor) { return this.encode(id, [harness, executor], () => JSON.stringify({ harness, executor })); }
  encode(id, parts, write) {
    const known = this.seen.get(id);
    if (known && known.parts.length === parts.length && known.parts.every((part, at) => part === parts[at])) return known.text;
    const text = write();
    this.seen.set(id, { parts, text });
    return text;
  }
  keep(ids) { for (const id of this.seen.keys()) if (!ids.has(id)) this.seen.delete(id); }
}

function technicalRows(technical, text) {
  const rows = new Map([[CORE_ROW, JSON.stringify({ cursor: technical.cursor, undoneEntries: technical.undoneEntries })]]);
  for (const id of new Set([...Object.keys(technical.harness || {}), ...Object.keys(technical.executor || {})])) {
    rows.set(entryRow(id), text.ofLedgers(entryRow(id), technical.harness?.[id], technical.executor?.[id]));
  }
  // One row per notification. As one list they were fifteen kilobytes rewritten by every
  // command that raised a single event; the ones that fall off the end are deleted as rows
  // that no longer exist, which is the same comparison every other row goes through.
  for (const event of technical.events || []) rows.set(eventRow(event.cursor), text.of(eventRow(event.cursor), event));
  return rows;
}

function technicalFromRows(rows) {
  const legacy = rows.get(LEGACY_ROW);
  const technical = legacy ? JSON.parse(legacy) : { harness: {}, executor: {}, undoneEntries: {}, cursor: 0, events: [] };
  const core = rows.get(CORE_ROW);
  if (core) Object.assign(technical, JSON.parse(core));
  const events = [];
  for (const [id, value] of rows) {
    if (id.startsWith('ev:')) { events.push(JSON.parse(value)); continue; }
    if (!id.startsWith('e:')) continue;
    const { harness, executor } = JSON.parse(value);
    technical.harness ||= {}; technical.executor ||= {};
    if (harness) technical.harness[id.slice(2)] = harness; else delete technical.harness[id.slice(2)];
    if (executor) technical.executor[id.slice(2)] = executor; else delete technical.executor[id.slice(2)];
  }
  if (events.length || core) technical.events = events.sort((a, b) => a.cursor - b.cursor);
  return technical;
}

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
    this.written = { items: new Map(), entries: new Map(), technical: new Map(), revision: null, nextId: null };
    this.text = { items: new RowText(), entries: new RowText(), technical: new RowText() };
    this.legacyTechnical = false;
  }

  load() {
    const meta = this.sql.exec("SELECT value FROM vl_meta WHERE id = 'revision'").toArray()[0];
    if (!meta) return undefined;
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
    const technicalRowValues = new Map();
    for (const row of this.sql.exec('SELECT id, value FROM vl_technical').toArray()) technicalRowValues.set(row.id, row.value);
    // The single-row layout is read, then replaced by the split one on the first save.
    this.legacyTechnical = technicalRowValues.has(LEGACY_ROW);
    this.written.technical = new Map([...technicalRowValues].filter(([id]) => id !== LEGACY_ROW));
    this.written.revision = meta.value;
    this.written.nextId = nextId?.value ?? null;
    return {
      graph: { revision: meta.value, nextId: nextId?.value, items },
      entries,
      technical: technicalRowValues.size ? technicalFromRows(technicalRowValues) : undefined
    };
  }

  async save(state) {
    // Serialised first, applied second, remembered third: the record of what the database
    // holds is updated only once the transaction has committed, so a failed write cannot
    // leave this thinking a row is saved when it is not.
    const items = state.graph.items.map(item => [item.id, this.text.items.of(item.id, item)]);
    const entries = state.entries.map(entry => [entry.cursor, this.text.entries.of(entry.cursor, entry)]);
    const technical = technicalRows(state.technical, this.text.technical);
    const liveIds = new Set(items.map(([id]) => id));
    const liveCursors = new Set(entries.map(([cursor]) => cursor));

    const changedItems = items.filter(([id, value]) => this.written.items.get(id) !== value);
    const droppedItems = [...this.written.items.keys()].filter(id => !liveIds.has(id));
    const changedEntries = entries.filter(([cursor, value]) => this.written.entries.get(cursor) !== value);
    const droppedEntries = [...this.written.entries.keys()].filter(cursor => !liveCursors.has(cursor));
    const changedTechnical = [...technical].filter(([id, value]) => this.written.technical.get(id) !== value);
    const droppedTechnical = [...this.written.technical.keys()].filter(id => !technical.has(id));
    const changedRevision = state.graph.revision !== this.written.revision;
    const changedNextId = Number.isFinite(state.graph.nextId) && state.graph.nextId !== this.written.nextId;

    if (!changedItems.length && !droppedItems.length && !changedEntries.length && !droppedEntries.length &&
        !changedTechnical.length && !droppedTechnical.length && !changedRevision && !changedNextId) return;

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
      for (const id of droppedTechnical) this.sql.exec('DELETE FROM vl_technical WHERE id = ?', id);
      for (const [id, value] of changedTechnical) {
        this.sql.exec('INSERT INTO vl_technical (id,value) VALUES (?,?) ON CONFLICT(id) DO UPDATE SET value=excluded.value', id, value);
      }
      // Whatever the old single row held is now in the split rows.
      if (this.legacyTechnical) this.sql.exec('DELETE FROM vl_technical WHERE id = ?', LEGACY_ROW);
    });

    for (const id of droppedItems) this.written.items.delete(id);
    for (const [id, value] of changedItems) this.written.items.set(id, value);
    for (const cursor of droppedEntries) this.written.entries.delete(cursor);
    for (const [cursor, value] of changedEntries) this.written.entries.set(cursor, value);
    this.legacyTechnical = false;
    this.text.items.keep(liveIds); this.text.entries.keep(liveCursors); this.text.technical.keep(new Set(technical.keys()));
    for (const id of droppedTechnical) this.written.technical.delete(id);
    for (const [id, value] of changedTechnical) this.written.technical.set(id, value);
    if (changedRevision) this.written.revision = state.graph.revision;
    if (changedNextId) this.written.nextId = state.graph.nextId;

    // Measured: removing this flush changed nothing, so the guarantee is kept.
    await this.storage.sync();
  }

  clear() {
    this.storage.transactionSync(() => {
      this.sql.exec('DELETE FROM vl_projection'); this.sql.exec('DELETE FROM vl_journal'); this.sql.exec('DELETE FROM vl_meta'); this.sql.exec('DELETE FROM vl_technical');
    });
    this.written = { items: new Map(), entries: new Map(), technical: new Map(), revision: null, nextId: null };
    this.text = { items: new RowText(), entries: new RowText(), technical: new RowText() };
    this.legacyTechnical = false;
  }
}
