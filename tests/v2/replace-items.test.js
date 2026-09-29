import { describe, expect, it } from 'vitest';
import { DocumentRuntime } from '../../src/v2/domain/document-runtime.js';

const item = (id, line1, extra = {}) => ({
  id, parentId: null, order: 10, status: 'Open', line1, line2: '', collapsed: false, tags: [], ...extra
});
const command = (runtime, items, seq = 1) => ({
  key: { clientKey: 'import', seq },
  context: { elementId: 'app', view: 'list', revision: runtime.graph.revision },
  command: { command: 'replaceItems', actId: 'list', actType: 'list', source: 'import', payload: { items } }
});

/** TEMPORARY (перенос v1 → v2): удаляется вместе с командой replaceItems.
 *
 *  Carrying a document over from another installation: the list arrives whole, with the ids
 *  it already had, and replaces what is here. */
describe('replacing the whole list', () => {
  it('keeps ids, fields and nesting, and drops what was there', async () => {
    const runtime = new DocumentRuntime();
    const before = runtime.graph.read().items.length;
    expect(before).toBeGreaterThan(1);

    const carried = [
      item('inbox', 'Входящие'),
      item('s5', 'Легализовать землю', { status: 'Focus', deadline: '2026-10-01' }),
      item('s6', 'Собрать документы', { parentId: 's5', status: 'Archive', line2: 'у Андрея', tags: ['дом'] })
    ];
    const receipt = await runtime.executeAndWait(command(runtime, carried));
    expect(receipt.status).toBe('completed');

    expect(runtime.graph.read().items).toEqual(carried);
    expect(runtime.graph.read({ id: 's6' })).toMatchObject({ parentId: 's5', status: 'Archive', line2: 'у Андрея', tags: ['дом'] });
  });

  it('can be rolled back, so a wrong list is not the end of the document', async () => {
    const runtime = new DocumentRuntime();
    const original = runtime.graph.read().items;
    await runtime.executeAndWait(command(runtime, [item('inbox', 'Входящие'), item('x1', 'Не то')]));
    expect(runtime.graph.read().items).toHaveLength(2);

    const entry = runtime.journal.entries.at(-1);
    await runtime.executeAndWait({
      key: { clientKey: 'ui', seq: 9 },
      context: { elementId: 'action:' + entry.id, view: 'list', revision: runtime.graph.revision, actionId: entry.id },
      command: { command: 'rollbackAction', actId: entry.id, actType: 'action', payload: {} }
    });
    // Order in the array carries no meaning — each task has its own order field — so the
    // list is compared as the set of tasks it is.
    const byId = (items) => Object.fromEntries(items.map(task => [task.id, task]));
    expect(byId(runtime.graph.read().items)).toEqual(byId(original));
  });

  it('refuses a list that is not a list, and one that breaks the graph', async () => {
    const runtime = new DocumentRuntime();
    const untouched = runtime.graph.read().items.length;
    // A rejected command is answered in the receipt, not thrown at the caller.
    expect((await runtime.executeAndWait(command(runtime, []))).error?.message).toMatch(/список задач/);
    expect((await runtime.executeAndWait(command(runtime, [item('a', 'Сирота', { parentId: 'нет-такого' })], 2))).error?.message)
      .toMatch(/родител/);
    expect(runtime.graph.read().items).toHaveLength(untouched);
  });
});
