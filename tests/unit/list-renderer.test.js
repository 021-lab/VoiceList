import { describe, expect, test } from 'vitest';
import { createRenderer } from '../../src/list-renderer.js';

describe('list renderer', () => {
  test('renders nested rows from flat items and shows status badge', () => {
    document.body.innerHTML = `
      <div id="list-container"></div>
      <div id="action-log-panel"></div>
    `;

    const renderer = createRenderer({
      container: document.getElementById('list-container'),
      actionLogPanel: document.getElementById('action-log-panel')
    });

    renderer.render({
      snapshot: {
        items: [
          { id: 'root1', parentId: null, order: 10, status: 'Open', line1: 'Root', line2: '', tags: [], collapsed: false },
          { id: 'child', parentId: 'root1', order: 10, status: 'Focus', line1: 'Child', line2: '', tags: [], collapsed: false }
        ]
      },
      actionLog: []
    });

    expect(document.querySelectorAll('.list-item-wrapper')).toHaveLength(2);
    expect(document.body.textContent).toContain('Focus');
  });

  test('hides archived tasks and their subtree in list view', () => {
    document.body.innerHTML = `
      <div id="list-container"></div>
      <div id="action-log-panel"></div>
    `;

    const renderer = createRenderer({
      container: document.getElementById('list-container'),
      actionLogPanel: document.getElementById('action-log-panel')
    });

    renderer.render({
      snapshot: {
        items: [
          { id: 'visible', parentId: null, order: 10, status: 'Open', line1: 'Visible task', line2: '', tags: [], collapsed: false },
          { id: 'archived', parentId: null, order: 20, status: 'Archive', line1: 'Archived task', line2: '', tags: [], collapsed: false },
          { id: 'archived-child', parentId: 'archived', order: 10, status: 'Open', line1: 'Archived child', line2: '', tags: [], collapsed: false }
        ]
      },
      actionLog: []
    });

    expect(document.body.textContent).toContain('Visible task');
    expect(document.body.textContent).not.toContain('Archived task');
    expect(document.body.textContent).not.toContain('Archived child');
    expect(document.querySelectorAll('.list-item-wrapper')).toHaveLength(1);
  });

  test('renders only frontier tasks in frontier view and keeps task bindings', () => {
    document.body.innerHTML = `
      <div id="root"></div>
      <div id="list-container"></div>
      <div id="action-log-panel"></div>
    `;

    const boundIds = [];
    const renderer = createRenderer({
      rootPanel: document.getElementById('root'),
      container: document.getElementById('list-container'),
      actionLogPanel: document.getElementById('action-log-panel'),
      bindRow({ item }) {
        boundIds.push(item.id);
      }
    });

    renderer.render({
      snapshot: {
        items: [
          { id: 'parent', parentId: null, order: 10, status: 'Open', line1: 'Parent', line2: '', tags: [], collapsed: false },
          { id: 'child', parentId: 'parent', order: 10, status: 'Open', line1: 'Child', line2: '', tags: [], collapsed: false },
          { id: 'paused', parentId: null, order: 20, status: 'Pause', line1: 'Paused', line2: '', tags: [], collapsed: false }
        ]
      },
      actionLog: []
    }, 'frontier');

    expect(document.body.textContent).not.toContain('Parent');
    expect(document.body.textContent).toContain('Child');
    expect(document.body.textContent).not.toContain('Paused');
    expect(document.querySelector('[data-act-id="child"]')).toBeTruthy();
    expect(boundIds).toEqual(['child']);
  });

  test('renders focus context in frontier view when focus is replaced by a child', () => {
    document.body.innerHTML = `
      <div id="root"></div>
      <div id="list-container"></div>
      <div id="action-log-panel"></div>
    `;

    const renderer = createRenderer({
      rootPanel: document.getElementById('root'),
      container: document.getElementById('list-container'),
      actionLogPanel: document.getElementById('action-log-panel')
    });

    renderer.render({
      snapshot: {
        items: [
          { id: 'focus', parentId: null, order: 10, status: 'Focus', line1: 'Focused task', line2: '', tags: [], collapsed: false },
          { id: 'child', parentId: 'focus', order: 10, status: 'Open', line1: 'Action child', line2: '', tags: [], collapsed: false }
        ]
      },
      actionLog: []
    }, 'frontier');

    expect(document.querySelector('.frontier-focus-strip')?.textContent).toContain('Focused task');
    expect(document.body.textContent).toContain('Action child');
  });

  test('renders frontier rows full width and toggles parent context above tapped task', () => {
    document.body.innerHTML = `
      <div id="root"></div>
      <div id="list-container"></div>
      <div id="action-log-panel"></div>
    `;

    const state = {
      snapshot: {
        items: [
          { id: 'parent', parentId: null, order: 10, status: 'Open', line1: 'Parent task', line2: '', tags: [], collapsed: false },
          { id: 'child', parentId: 'parent', order: 10, status: 'Open', line1: 'Frontier child', line2: '', tags: [], collapsed: false }
        ]
      },
      actionLog: []
    };

    const renderer = createRenderer({
      rootPanel: document.getElementById('root'),
      container: document.getElementById('list-container'),
      actionLogPanel: document.getElementById('action-log-panel')
    });

    renderer.render(state, 'frontier');

    const childWrapper = document.querySelector('[data-id="child"]');
    expect(childWrapper.style.marginLeft).toBe('0px');
    expect(document.querySelector('.frontier-parent-wrapper')).toBeNull();

    childWrapper.querySelector('.list-item').click();

    expect(document.querySelector('.frontier-parent-wrapper')?.textContent).toContain('Parent task');
    expect(document.querySelector('[data-id="child"]').style.marginLeft).toBe('24px');

    document.querySelector('[data-id="child"] .list-item').click();

    expect(document.querySelector('.frontier-parent-wrapper')).toBeNull();
    expect(document.querySelector('[data-id="child"]').style.marginLeft).toBe('0px');
  });

  test('shows synthetic list parent for root-level frontier tasks', () => {
    document.body.innerHTML = `
      <div id="root"></div>
      <div id="list-container"></div>
      <div id="action-log-panel"></div>
    `;

    const renderer = createRenderer({
      rootPanel: document.getElementById('root'),
      container: document.getElementById('list-container'),
      actionLogPanel: document.getElementById('action-log-panel')
    });

    renderer.render({
      snapshot: {
        items: [
          { id: 'root-task', parentId: null, order: 10, status: 'Open', line1: 'Root frontier task', line2: '', tags: [], collapsed: false }
        ]
      },
      actionLog: []
    }, 'frontier');

    document.querySelector('[data-id="root-task"] .list-item').click();

    expect(document.querySelector('.frontier-parent-wrapper')?.textContent).toContain('Мой список');
    expect(document.querySelector('[data-id="root-task"]').style.marginLeft).toBe('24px');
  });

  test('does not render frontier rows from archived branches', () => {
    document.body.innerHTML = `
      <div id="root"></div>
      <div id="list-container"></div>
      <div id="action-log-panel"></div>
    `;

    const renderer = createRenderer({
      rootPanel: document.getElementById('root'),
      container: document.getElementById('list-container'),
      actionLogPanel: document.getElementById('action-log-panel')
    });

    renderer.render({
      snapshot: {
        items: [
          { id: 'archived', parentId: null, order: 10, status: 'Archive', line1: 'Archived task', line2: '', tags: [], collapsed: false },
          { id: 'archived-focus', parentId: 'archived', order: 10, status: 'Focus', line1: 'Archived focus', line2: '', tags: [], collapsed: false },
          { id: 'visible', parentId: null, order: 20, status: 'Open', line1: 'Visible frontier', line2: '', tags: [], collapsed: false }
        ]
      },
      actionLog: []
    }, 'frontier');

    expect(document.body.textContent).toContain('Visible frontier');
    expect(document.body.textContent).not.toContain('Archived task');
    expect(document.body.textContent).not.toContain('Archived focus');
  });

  test('renders search results from provided item ids', () => {
    document.body.innerHTML = `
      <div id="root"></div>
      <div id="list-container"></div>
      <div id="action-log-panel"></div>
    `;

    const boundIds = [];
    const renderer = createRenderer({
      rootPanel: document.getElementById('root'),
      container: document.getElementById('list-container'),
      actionLogPanel: document.getElementById('action-log-panel'),
      bindRow({ item }) {
        boundIds.push(item.id);
      }
    });

    renderer.render({
      snapshot: {
        items: [
          { id: 'match', parentId: null, order: 10, status: 'Open', line1: 'Купить молоко', line2: '', tags: [], collapsed: false },
          { id: 'other', parentId: null, order: 20, status: 'Open', line1: 'Купить хлеб', line2: '', tags: [], collapsed: false },
          { id: 'archived', parentId: null, order: 30, status: 'Archive', line1: 'Старое молоко', line2: '', tags: [], collapsed: false }
        ]
      },
      actionLog: []
    }, 'search', { query: 'молоко', itemIds: ['match', 'archived'] });

    expect(document.body.textContent).toContain('Поиск: молоко');
    expect(document.body.textContent).toContain('Купить молоко');
    expect(document.body.textContent).not.toContain('Купить хлеб');
    expect(document.body.textContent).not.toContain('Старое молоко');
    expect(boundIds).toEqual(['match']);
  });

  test('renders transcript, command, comments, and log row id in log view', () => {
    document.body.innerHTML = `
      <div id="root"></div>
      <div id="list-container"></div>
      <div id="action-log-panel"></div>
      <div id="action-log-list"></div>
    `;

    const renderer = createRenderer({
      rootPanel: document.getElementById('root'),
      container: document.getElementById('list-container'),
      actionLogPanel: document.getElementById('action-log-panel'),
      actionLogList: document.getElementById('action-log-list')
    });

    renderer.render({
      snapshot: { items: [] },
      actionLog: [{
        id: 'log-1',
        createdAt: '2026-08-02T10:00:00.000Z',
        label: 'Создана задача: молоко',
        transcript: 'добавь молоко',
        command: { command: 'addItem', payload: { line1: 'молоко' } },
        patch: [],
        syncStatus: 'pending',
        comments: [{ id: 'c1', createdAt: '2026-08-02T10:01:00.000Z', text: 'купить сегодня' }]
      }]
    }, 'log');

    const row = document.querySelector('.action-log-row');
    expect(row?.dataset.logId).toBe('log-1');
    expect(row?.textContent).toContain('добавь молоко');
    expect(row?.textContent).toContain('addItem');
    expect(row?.textContent).toContain('купить сегодня');
  });
});
