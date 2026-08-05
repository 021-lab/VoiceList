import { seedState } from '../list-data.js';
import { createApp } from './list-app.js';
import { createCloudflareDocumentClient } from './cloudflare-document-client.js';
import { createInterpreter } from './list-interpreter.js';
import { createLogStore } from './list-log-store.js';
import { createRenderer } from './list-renderer.js';
import { createStore } from './list-store.js';
import { createSync } from './list-sync.js';
import { createUI } from './list-ui.js';

const PREVIEW_BUILD_HASH = __PREVIEW_BUILD_HASH__;

async function bootstrapListManagerPreview() {
  document.documentElement.dataset.previewBuildHash = PREVIEW_BUILD_HASH;

  const store = createStore({
    storageKey: 'searchmydata.list-interface.state',
    storage: window.localStorage,
    seedState
  });
  const logStore = createLogStore({
    storageKeyPrefix: 'searchmydata.list-interface.log',
    storage: window.localStorage
  });

  let app = null;
  const ui = createUI({
    rootPanel: document.getElementById('app-root'),
    header: document.querySelector('header'),
    viewToggleButton: document.getElementById('view-toggle-btn'),
    frontierButton: document.getElementById('frontier-tab-btn'),
    settingsButton: document.getElementById('settings-btn'),
    undoButton: document.getElementById('undo-btn'),
    addButton: document.getElementById('add-btn'),
    container: document.getElementById('list-container'),
    toastEl: document.getElementById('toast'),
    dropPanel: document.getElementById('drop-zone-panel'),
    tagPanel: document.getElementById('tag-panel'),
    voiceOverlay: document.getElementById('voice-overlay'),
    overlay: document.getElementById('modal-overlay'),
    input1: document.getElementById('input-line1'),
    input2: document.getElementById('input-line2'),
    modalTitle: document.getElementById('modal-title'),
    btnConfirm: document.getElementById('btn-confirm'),
    btnCancel: document.getElementById('btn-cancel'),
    viewContent: document.getElementById('modal-view-content'),
    viewLine1: document.getElementById('view-line1'),
    viewLine2: document.getElementById('view-line2'),
    viewTagsEl: document.getElementById('view-tags'),
    actionLogPanel: document.getElementById('action-log-panel'),
    taskPage: document.getElementById('task-page'),
    taskPageClose: document.getElementById('task-page-close'),
    taskPageSave: document.getElementById('task-page-save'),
    taskPageTitle: document.getElementById('task-page-title'),
    taskPageLine1: document.getElementById('task-page-line1'),
    taskPageLine2: document.getElementById('task-page-line2'),
    taskPageStatus: document.getElementById('task-page-status'),
    taskPageSubtasks: document.getElementById('task-page-subtasks'),
    taskPageChildInput: document.getElementById('task-page-child-input'),
    taskPageAddChild: document.getElementById('task-page-add-child'),
    settingsOverlay: document.getElementById('settings-overlay'),
    settingsClose: document.getElementById('settings-close'),
    workflowyUrlInput: document.getElementById('workflowy-url-input'),
    workflowyImportButton: document.getElementById('workflowy-import-btn'),
    workflowyImportStatus: document.getElementById('workflowy-import-status')
  });

  const renderer = createRenderer({
    rootPanel: document.getElementById('app-root'),
    container: document.getElementById('list-container'),
    actionLogPanel: document.getElementById('action-log-panel'),
    actionLogList: document.getElementById('action-log-list'),
    bindRow: ui.bindRow,
    onRendered: ui.onRendered
  });

  const adapter = {
    async load() {
      return null;
    },
    async save(state) {
      return state;
    }
  };
  const transport = {
    async create(entry) {
      return entry;
    },
    async update(entry) {
      return entry;
    }
  };
  let sync = null;
  const useCloudflareBackend = !window.location.hash.includes('v=local-dev') &&
    window.location.port !== '4511';
  const documentClient = useCloudflareBackend ? createCloudflareDocumentClient() : null;

  app = createApp({
    adapter,
    documentClient,
    interpreter: createInterpreter(),
    renderer,
    store,
    logStore,
    sync: {
      enqueueCreate(entry) {
        return sync.enqueueCreate(entry);
      },
      enqueueUpdate(entry) {
        return sync.enqueueUpdate(entry);
      }
    },
    ui
  });

  sync = createSync({
    transport,
    logStore,
    onLogEntriesChange(nextEntries) {
      app.onLogEntriesChange(nextEntries);
    }
  });

  await app.init();
}

bootstrapListManagerPreview().catch((error) => {
  console.error('Failed to bootstrap list manager', error);
});
