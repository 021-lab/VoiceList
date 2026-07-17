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
    taskPageTitle: el('task-page-title'),
    taskPageLine1: el('task-page-line1', 'input'),
    taskPageLine2: el('task-page-line2', 'input'),
    taskPageStatus: el('task-page-status', 'select'),
    taskPageSubtasks: el('task-page-subtasks'),
    taskPageChildInput: el('task-page-child-input', 'input'),
    taskPageAddChild: el('task-page-add-child', 'button')
  });
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
});
