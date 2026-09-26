import { describe, expect, it } from 'vitest';
import { DatabaseSync } from 'node:sqlite';
import { RuntimeStorage } from '../../worker/v2/runtime-storage.js';

/** The durable object's storage, played by SQLite: the point of these tests is what reaches
 *  the database, so a stub that records calls would prove nothing about reading it back. */
function durableStorage() {
  const db = new DatabaseSync(':memory:');
  const statements = [];
  return {
    statements,
    sql: {
      exec(query, ...params) {
        statements.push(query);
        const prepared = db.prepare(query);
        if (/^\s*select/i.test(query)) { const rows = prepared.all(...params); return { toArray: () => rows }; }
        prepared.run(...params);
        return { toArray: () => [] };
      }
    },
    transactionSync(fn) { return fn(); },
    async sync() {}
  };
}

const item = (id, line1) => ({ id, parentId: null, order: 10, status: 'Open', line1, line2: '', collapsed: false, tags: [] });
const entry = (cursor) => ({ id: 'e' + cursor, cursor, type: 'interaction', kind: 'ui', key: { clientKey: 'c', seq: cursor }, context: {} });
const ledger = (id, outcomes = 1) => ({
  status: 'complete', nextIndex: outcomes,
  outcomes: Array.from({ length: outcomes }, (_, index) => ({ key: `${id}:${index}`, command: { command: 'addItem' }, changes: [{ id: 'x', fields: ['line1'] }] }))
});
const state = (ids, cursor = ids.length) => ({
  graph: { revision: cursor, nextId: 1000 + cursor, items: ids.map(id => item(id, 'Задача ' + id)) },
  entries: ids.map((_, index) => entry(index + 1)),
  technical: {
    harness: Object.fromEntries(ids.map((_, index) => ['e' + (index + 1), { status: 'bypassed' }])),
    executor: Object.fromEntries(ids.map((_, index) => ['e' + (index + 1), ledger('e' + (index + 1))])),
    undoneEntries: {}, cursor, events: [{ cursor, entryId: 'e' + cursor }]
  }
});

describe('what a save actually writes', () => {
  it('reads back exactly what it stored', async () => {
    const storage = durableStorage();
    const runtime = new RuntimeStorage(storage);
    const written = state(['inbox', 'a', 'b']);
    await runtime.save(written);

    const reread = new RuntimeStorage(storage).load();
    expect(reread.graph).toEqual(written.graph);
    expect(reread.entries).toEqual(written.entries);
    expect(reread.technical).toEqual(written.technical);
  });

  it('touches only the rows a command changed', async () => {
    const storage = durableStorage();
    const runtime = new RuntimeStorage(storage);
    await runtime.save(state(['inbox', 'a', 'b']));

    const next = state(['inbox', 'a', 'b', 'c'], 4);
    storage.statements.length = 0;
    await runtime.save(next);

    const writes = storage.statements.filter(query => /INSERT|DELETE/i.test(query));
    // The new task, the new journal entry, revision, nextId, and on the ledger side four
    // rows: this entry's ledger, the shared row, the new notification and the dropped one.
    expect(writes).toHaveLength(8);
    expect(writes.filter(query => query.includes('vl_technical'))).toHaveLength(4);
    expect(new RuntimeStorage(storage).load().technical).toEqual(next.technical);
  });

  it('writes nothing at all when nothing changed', async () => {
    const storage = durableStorage();
    const runtime = new RuntimeStorage(storage);
    const unchanged = state(['inbox', 'a']);
    await runtime.save(unchanged);
    storage.statements.length = 0;
    await runtime.save(unchanged);
    expect(storage.statements.filter(query => /INSERT|DELETE/i.test(query))).toHaveLength(0);
  });

  it('drops a deleted task, its journal entry and its ledger', async () => {
    const storage = durableStorage();
    const runtime = new RuntimeStorage(storage);
    await runtime.save(state(['inbox', 'a', 'b']));
    await runtime.save(state(['inbox', 'a'], 2));

    const reread = new RuntimeStorage(storage).load();
    expect(reread.graph.items.map(task => task.id)).toEqual(['inbox', 'a']);
    expect(Object.keys(reread.technical.executor)).toEqual(['e1', 'e2']);
    expect(storage.sql.exec('SELECT id FROM vl_technical').toArray().map(row => row.id)).not.toContain('e:e3');
  });
});

describe('a document stored under the single-row layout', () => {
  it('is read whole and split on the next save', async () => {
    const storage = durableStorage();
    const old = state(['inbox', 'a', 'b']);
    // What the previous layout left behind: every ledger in one row, under id 'state'.
    const seed = new RuntimeStorage(storage);
    seed.sql.exec("INSERT INTO vl_meta (id,value) VALUES ('revision',?)", old.graph.revision);
    seed.sql.exec("INSERT INTO vl_meta (id,value) VALUES ('nextId',?)", old.graph.nextId);
    for (const task of old.graph.items) seed.sql.exec('INSERT INTO vl_projection (id,value) VALUES (?,?)', task.id, JSON.stringify(task));
    for (const row of old.entries) seed.sql.exec('INSERT INTO vl_journal (cursor,value) VALUES (?,?)', row.cursor, JSON.stringify(row));
    seed.sql.exec("INSERT INTO vl_technical (id,value) VALUES ('state',?)", JSON.stringify(old.technical));

    const runtime = new RuntimeStorage(storage);
    expect(runtime.load().technical).toEqual(old.technical);

    await runtime.save(state(['inbox', 'a', 'b', 'c'], 4));
    const rows = storage.sql.exec('SELECT id FROM vl_technical').toArray().map(row => row.id);
    expect(rows).not.toContain('state');
    expect(rows).toContain('core');
    expect(rows).toContain('e:e1');
    expect(new RuntimeStorage(storage).load().technical).toEqual(state(['inbox', 'a', 'b', 'c'], 4).technical);
  });
});
