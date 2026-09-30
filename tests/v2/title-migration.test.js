import { describe, expect, it } from 'vitest';
import { DocumentRuntime } from '../../src/v2/domain/document-runtime.js';
import { TaskGraph, normalizeTask } from '../../src/v2/domain/task-graph.js';

/** Документы, записанные до переименования: заголовок звался line1, рядом жила вторая строка. */
const stored = [
  { id: 'inbox', parentId: null, order: 0, status: 'Open', line1: 'Входящие', line2: '', collapsed: false, tags: [] },
  { id: 'milk1', parentId: null, order: 10, status: 'Open', line1: 'Молоко', line2: '2 пакета', collapsed: false, tags: [] }
];

describe('переезд с line1 на title', () => {
  it('читает старый документ и отдаёт его в новой форме', () => {
    const graph = new TaskGraph({ items: stored, revision: 3, nextId: 1000 });
    const items = graph.read().items;
    expect(items.map(item => item.title)).toEqual(['Входящие', 'Молоко']);
    expect(JSON.stringify(items)).not.toContain('line1');
    expect(JSON.stringify(items)).not.toContain('line2');
  });

  it('принимает команду в старой форме и ничего из неё не сохраняет лишнего', () => {
    const graph = new TaskGraph({ items: stored, revision: 0, nextId: 1000 });
    graph.apply([{ command: 'addItem', actId: 'list', payload: { line1: 'Хлеб', line2: 'ржаной' } }], 0);
    const added = graph.read().items.find(item => item.title === 'Хлеб');
    expect(added).toBeTruthy();
    expect(added).not.toHaveProperty('line2');
    expect(added).not.toHaveProperty('line1');
  });

  it('откатывает действие, записанное в старой форме', () => {
    const graph = new TaskGraph({ items: stored, revision: 5, nextId: 1000 });
    // Исход из журнала тех времён: переименование, записанное полем line1.
    const outcome = { changes: [{
      id: 'milk1', fields: ['line1', 'line2'],
      before: { id: 'milk1', parentId: null, order: 10, status: 'Open', line1: 'Молоко', line2: '2 пакета', collapsed: false, tags: [] },
      after: { id: 'milk1', parentId: null, order: 10, status: 'Open', line1: 'Молоко 3.2%', line2: '', collapsed: false, tags: [] }
    }] };
    const current = new TaskGraph({ items: [stored[0], { ...stored[1], line1: 'Молоко 3.2%', line2: '' }], revision: 5, nextId: 1000 });
    current.rollback([outcome]);
    expect(current.read({ id: 'milk1' }).title).toBe('Молоко');
  });

  it('второй строки не остаётся нигде', () => {
    expect(normalizeTask({ id: 'a', line1: 'Название', line2: 'подпись' })).toEqual({ id: 'a', title: 'Название' });
  });
});

describe('документ целиком', () => {
  it('поднимается со старых данных и работает дальше', async () => {
    const runtime = new DocumentRuntime({ initialState: { graph: { items: stored, revision: 2, nextId: 1000 }, entries: [] } });
    const receipt = await runtime.executeAndWait({
      key: { clientKey: 'ui', seq: 1 },
      context: { elementId: 'task:milk1', view: 'list', revision: runtime.graph.revision },
      command: { command: 'editItem', actId: 'milk1', actType: 'task', payload: { title: 'Молоко 3.2%' } }
    });
    expect(receipt.status).toBe('completed');
    expect(runtime.graph.read({ id: 'milk1' })).toMatchObject({ title: 'Молоко 3.2%' });
    expect(JSON.stringify(runtime.graph.read().items)).not.toContain('line');
  });
});
