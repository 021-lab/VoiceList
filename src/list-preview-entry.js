import { createApp } from './list-app.js';
import { createBackendAdapter } from './list-backend-adapter.js';
import { createInterpreter } from './list-interpreter.js';
import { createRenderer } from './list-renderer.js';
import { createStore } from './list-store.js';
import { createSync } from './list-sync.js';
import { createUI } from './list-ui.js';

const PREVIEW_BUILD_HASH = __PREVIEW_BUILD_HASH__;

function previewApiBase() {
  const pathname = window.location.pathname;
  const directory = pathname.endsWith('/')
    ? pathname.replace(/\/$/, '')
    : pathname.slice(0, pathname.lastIndexOf('/'));
  return `${directory}/api` || '/api';
}

async function bootstrapListManagerPreview() {
  document.documentElement.dataset.previewBuildHash = PREVIEW_BUILD_HASH;
  window.__LIST_MANAGER_READY__ = false;

  const store = createStore({
    storageKey: 'voicelist.universal-interface.state',
    storage: window.localStorage
  });

  let app = null;
  const ui = createUI({
    rootPanel: document.getElementById('app-root'),
    header: document.querySelector('header'),
    viewToggleButton: document.getElementById('view-toggle-btn'),
    frontierButton: document.getElementById('frontier-tab-btn'),
    undoButton: document.getElementById('undo-btn'),
    addButton: document.getElementById('add-btn'),
    container: document.getElementById('list-container'),
    toastEl: document.getElementById('toast'),
    dropPanel: document.getElementById('drop-zone-panel'),
    tagPanel: document.getElementById('tag-panel'),
    overlay: document.getElementById('modal-overlay'),
    input1: document.getElementById('input-line1'),
    modalTitle: document.getElementById('modal-title'),
    btnConfirm: document.getElementById('btn-confirm'),
    btnCancel: document.getElementById('btn-cancel'),
    viewContent: document.getElementById('modal-view-content'),
    viewLine1: document.getElementById('view-line1'),
    viewTagsEl: document.getElementById('view-tags'),
    actionLogPanel: document.getElementById('action-log-panel'),
    taskPage: document.getElementById('task-page'),
    taskPageClose: document.getElementById('task-page-close'),
    taskPageSave: document.getElementById('task-page-save'),
    taskPageTitle: document.getElementById('task-page-title'),
    taskPageLine1: document.getElementById('task-page-line1'),
    taskPageStatus: document.getElementById('task-page-status'),
    taskPageSubtasks: document.getElementById('task-page-subtasks'),
    taskPageChildInput: document.getElementById('task-page-child-input'),
    taskPageAddChild: document.getElementById('task-page-add-child')
  });

  const renderer = createRenderer({
    rootPanel: document.getElementById('app-root'),
    container: document.getElementById('list-container'),
    actionLogPanel: document.getElementById('action-log-panel'),
    actionLogList: document.getElementById('action-log-list'),
    bindRow: ui.bindRow,
    onRendered: ui.onRendered
  });

  const adapter = createBackendAdapter({ apiBase: previewApiBase() });
  let sync = null;

  app = createApp({
    adapter,
    interpreter: createInterpreter(),
    renderer,
    store,
    sync: {
      enqueue(state, actionLogEntry) {
        return sync.enqueue(state, actionLogEntry);
      }
    },
    ui
  });

  sync = createSync({
    adapter,
    store,
    onStateChange(nextState) {
      app.onSyncedState(nextState);
    }
  });

  await app.init();
  window.__LIST_MANAGER_READY__ = true;
}

bootstrapListManagerPreview().catch((error) => {
  console.error('Failed to bootstrap list manager', error);
});
