import { describe, expect, it } from 'vitest';
import { JSDOM } from 'jsdom';
import { Presentation } from '../../src/v2/domain/presentation.js';
import { graphCommands, uiCommands } from '../../src/v2/domain/contracts.js';
import { Client } from '../../src/v2/client/client.js';

const graph = {
  revision: 8,
  items: [
    { id: 'inbox', parentId: null, order: 0, status: 'Open', line1: 'Входящие', line2: '', collapsed: false, tags: [] },
    { id: 'milk1', parentId: null, order: 10, status: 'Open', line1: 'Купить молоко', line2: '', collapsed: false, tags: [] },
    { id: 'home', parentId: null, order: 20, status: 'Open', line1: 'Дом', line2: '', collapsed: false, tags: [] }
  ]
};

const payloads = {
  addItem: { line1: 'Новая задача', line2: 'Подробности' },
  addChild: { line1: 'Подзадача' }, editItem: { line1: 'Новое название' },
  setStatus: { status: 'Done' }, setParent: { parentId: 'home' }, setTags: { tag: 'важно' },
  setDeadline: { deadline: '2026-10-20' }, toggleCollapse: {}, deleteItem: {},
  reorderItems: { arranged: [{ id: 'milk1', parentId: 'home', order: 10 }] },
  importWorkflowyTree: { tree: { title: 'Проект', children: [{ title: 'Шаг', children: [] }] } }
};

function command(type) {
  return { command: type, actId: ['addItem', 'importWorkflowyTree', 'reorderItems'].includes(type) ? 'list' : 'milk1', payload: payloads[type] || {} };
}

function actionPage(records, action = {}) {
  const journal = {
    cursor: records.length,
    actions: () => [{ id: 'root', actionId: 'root', transcript: 'Измени задачу', canRollback: true, rolledBack: false, ...action }],
    chain: () => records
  };
  return new Presentation().compose(graph, journal, { view: 'action', actionId: 'root' }).root.children[1];
}

describe('v0.2 action page presentation', () => {
  it('builds human-readable view models for every accepted command and a safe future fallback', () => {
    const modelCommands = [...graphCommands].map(command);
    const uiEntries = Object.keys(uiCommands).map((type, index) => ({ id: `ui-${index}`, cursor: index + 2, kind: 'ui', command: command(type) }));
    const records = [
      { id: 'root', cursor: 1, kind: 'text', text: 'Измени задачу', answer: 'Готово', commands: modelCommands, modelContext: { text: 'Измени задачу' } },
      ...uiEntries,
      { id: 'rollback', cursor: 20, kind: 'ui', corrects: 'root', command: { command: 'rollbackAction', actId: 'root', payload: {} } },
      { id: 'future', cursor: 21, kind: 'ui', corrects: 'root', command: { command: 'futureCommand', actId: 'milk1', payload: { value: 'пример' } } }
    ];
    const page = actionPage(records);
    expect(page.props.title).toBe('История действия');
    expect(page.props.records[0].commandViews).toHaveLength(graphCommands.size);
    expect(page.props.records[0].commandViews.every(view => !view.actionLabel.startsWith('Команда «'))).toBe(true);
    expect(page.props.records[0].commandViews.find(view => view.type === 'setStatus')).toMatchObject({
      actionLabel: 'Изменить статус', targetLabel: 'Купить молоко', fields: [{ label: 'Новый статус', value: 'Выполнена' }]
    });
    expect(page.props.records.find(record => record.id === 'rollback').commandView.actionLabel).toBe('Откатить действие и все его корректировки');
    expect(page.props.records.find(record => record.id === 'future').commandView).toMatchObject({
      actionLabel: 'Команда «futureCommand»', targetLabel: 'Купить молоко'
    });
  });

  it('resolves a deleted target from the exact model context saved with its record', () => {
    const page = actionPage([{
      id: 'root', cursor: 1, kind: 'text', text: 'Заверши прежнюю задачу', answer: 'Готово',
      commands: [{ command: 'setStatus', actId: 'deleted-task', payload: { status: 'Done' } }],
      modelContext: { text: 'Заверши прежнюю задачу', tasks: [{ id: 'deleted-task', line1: 'Touch edit historical' }] }
    }]);

    expect(graph.items.some(item => item.id === 'deleted-task')).toBe(false);
    expect(page.props.records[0].commandViews[0]).toMatchObject({
      actionLabel: 'Изменить статус', targetLabel: 'Touch edit historical'
    });
  });

  it('renders semantic history, reveals exact context above model output and never shows raw command JSON', () => {
    const records = [
      {
        id: 'root', cursor: 1, kind: 'text', text: 'Сделай молоко выполненным', answer: 'Задача отмечена выполненной',
        commands: [{ command: 'setStatus', actId: 'milk1', payload: { status: 'Done' } }],
        modelContext: { text: 'Сделай молоко выполненным', target: 'milk1' }
      },
      { id: 'rollback', cursor: 2, kind: 'ui', corrects: 'root', command: { command: 'rollbackAction', actId: 'root', payload: {} } }
    ];
    const page = actionPage(records, { canRollback: false, rolledBack: true });
    const dom = new JSDOM('<!doctype html><html><head></head><body><div id="app-root"></div></body></html>', { url: 'https://example.test/' });
    const client = new Client({ document: dom.window.document, storage: dom.window.sessionStorage, fetch: async () => { throw new Error('unused'); }, pollMs: 0 });
    client.renderActionPage(page);

    const document = dom.window.document;
    expect(document.querySelector('.v02-action-body h2').textContent).toBe('История действия');
    expect([...document.querySelectorAll('.v02-record-marker')].map(item => item.textContent)).toEqual(['Исходная команда', 'Корректировка интерфейса']);
    expect(document.body.textContent).toContain('Команда пользователя');
    expect(document.body.textContent).toContain('Ответ модели');
    expect(document.body.textContent).toContain('Действия модели');
    expect(document.body.textContent).toContain('Изменить статус');
    expect(document.body.textContent).toContain('Купить молоко');
    expect(document.body.textContent).toContain('Откатить действие и все его корректировки');
    expect(document.body.textContent).not.toContain('setStatus');
    expect(document.body.textContent).not.toContain('rollbackAction');

    const context = document.querySelector('.v02-model-context');
    const output = document.querySelector('.v02-model-output');
    const answer = document.querySelector('.v02-model-answer');
    expect(output.previousElementSibling).toBe(context);
    expect(context.hidden).toBe(true);
    answer.click();
    expect(context.hidden).toBe(false);
    expect(context.textContent).toContain('"target": "milk1"');
    expect(document.querySelector('.v02-context-disclosure').getAttribute('aria-expanded')).toBe('true');
    document.querySelector('[data-entry-id="root"] .v02-command-card').click();
    expect(context.hidden).toBe(true);

    const uiRecord = document.querySelector('[data-entry-id="rollback"]');
    expect(uiRecord.querySelector('.v02-model-context')).toBeNull();
    expect(document.getElementById('action-rollback')).toMatchObject({ disabled: true, textContent: 'Откат выполнен' });
  });
});
