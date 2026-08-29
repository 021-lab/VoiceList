import { describe, expect, test } from 'vitest';
import { createUI } from '../../src/list-ui.js';

function el(id, tag = 'div') {
  const element = document.createElement(tag);
  element.id = id;
  document.body.appendChild(element);
  return element;
}

function createUiFixture(rootPanel = el('app-root')) {
  return createUI({
    rootPanel,
    header: document.createElement('header'),
    viewToggleButton: el('view-toggle-btn', 'button'),
    frontierButton: el('frontier-tab-btn', 'button'),
    settingsButton: el('settings-btn', 'button'),
    undoButton: el('undo-btn', 'button'),
    addButton: el('add-btn', 'button'),
    container: el('list-container'),
    toastEl: el('toast'),
    dropPanel: el('drop-zone-panel'),
    tagPanel: el('tag-panel'),
    overlay: el('modal-overlay'),
    input1: el('input-line1', 'input'),
    input2: el('input-line2', 'input'),
    modalTitle: el('modal-title'),
    btnConfirm: el('btn-confirm', 'button'),
    btnCancel: el('btn-cancel', 'button'),
    viewContent: el('modal-view-content'),
    viewLine1: el('view-line1'),
    viewLine2: el('view-line2'),
    viewTagsEl: el('view-tags'),
    actionLogPanel: el('action-log-panel'),
    taskPage: el('task-page'),
    taskPageClose: el('task-page-close', 'button'),
    taskPageSave: el('task-page-save', 'button'),
    taskPageLine1: el('task-page-line1', 'input'),
    taskPageStatus: el('task-page-status', 'select'),
    taskPageParent: el('task-page-parent', 'a'),
    taskPageSubtasks: el('task-page-subtasks'),
    taskPageChildInput: el('task-page-child-input', 'input'),
    taskPageAddChild: el('task-page-add-child', 'button'),
    settingsOverlay: el('settings-overlay'),
    settingsClose: el('settings-close', 'button'),
    workflowyUrlInput: el('workflowy-url-input', 'input'),
    workflowyImportButton: el('workflowy-import-btn', 'button'),
    workflowyImportStatus: el('workflowy-import-status')
  });
}

function dispatchTouch(node, type, x, y) {
  const event = new Event(type, { bubbles: true, cancelable: true });
  const touches = type === 'touchend' || type === 'touchcancel' ? [] : [{ clientX: x, clientY: y }];
  Object.defineProperty(event, 'touches', { value: touches });
  node.dispatchEvent(event);
}

describe('list UI', () => {
  test('dispatches showFrontier from the header frontier button', () => {
    document.body.innerHTML = '';
    const inputs = [];

    const ui = createUiFixture();

    ui.setDispatch((input) => inputs.push(input));
    ui.bindGlobal();

    document.getElementById('frontier-tab-btn').click();

    expect(inputs).toEqual([{
      actId: 'frontier',
      actType: 'tab',
      command: 'showFrontier',
      payload: {},
      source: 'frontier-tab'
    }]);
  });

  test('does not dispatch collapse toggle on frontier row click', () => {
    document.body.innerHTML = '';
    const inputs = [];
    const rootPanel = el('app-root');
    rootPanel.dataset.viewMode = 'frontier';
    const ui = createUiFixture(rootPanel);
    const wrapper = document.createElement('div');
    const row = document.createElement('div');
    const actionBg = document.createElement('div');

    ui.setDispatch((input) => inputs.push(input));
    ui.bindRow({
      actionBg,
      item: { id: 'task1' },
      row,
      wrapper
    });

    row.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(inputs).toEqual([]);
  });

  test('keeps a small touch drift as a collapse tap', () => {
    document.body.innerHTML = '';
    const inputs = [];
    const rootPanel = el('app-root');
    rootPanel.dataset.viewMode = 'list';
    const ui = createUiFixture(rootPanel);
    const wrapper = document.createElement('div');
    const row = document.createElement('div');
    const actionBg = document.createElement('div');

    ui.setDispatch((input) => inputs.push(input));
    ui.bindRow({
      actionBg,
      item: { id: 'task1' },
      row,
      wrapper
    });

    dispatchTouch(row, 'touchstart', 40, 40);
    dispatchTouch(row, 'touchmove', 40, 48);
    dispatchTouch(row, 'touchend', 40, 48);

    expect(inputs).toEqual([{
      actId: 'task1',
      actType: 'task',
      command: 'toggleCollapse',
      payload: {},
      source: 'tap'
    }]);
  });

  test('dispatches Workflowy import from the settings panel', () => {
    document.body.innerHTML = '';
    const inputs = [];
    const ui = createUiFixture();

    ui.setDispatch((input) => inputs.push(input));
    ui.bindGlobal();

    document.getElementById('settings-btn').click();
    expect(document.getElementById('settings-overlay').classList.contains('open')).toBe(true);
    document.getElementById('workflowy-url-input').value = 'https://workflowy.com/s/task-tree/iq43ak7FYqEEO1uO';
    document.getElementById('workflowy-import-btn').click();

    expect(inputs).toEqual([{
      actId: 'workflowy-import',
      actType: 'settings',
      command: 'importWorkflowy',
      payload: { url: 'https://workflowy.com/s/task-tree/iq43ak7FYqEEO1uO' },
      source: 'settings-import'
    }]);
    expect(document.getElementById('workflowy-import-status').textContent).toContain('Импорт');
  });
});
