import { seedState } from '../list-data.js';
import { createApp } from './list-app.js';
import { createBackendAdapter } from './list-backend-adapter.js';
import { createInterpreter } from './list-interpreter.js';
import { createRenderer } from './list-renderer.js';
import { createStore } from './list-store.js';
import { createSync } from './list-sync.js';
import { createUI } from './list-ui.js';
import {
  createRecognitionFactory,
  createVoiceController,
  createVoiceUI,
  isSpeechRecognitionSupported
} from './list-voice.js';

const PREVIEW_BUILD_HASH = __PREVIEW_BUILD_HASH__;

async function bootstrapListManagerPreview() {
  document.documentElement.dataset.previewBuildHash = PREVIEW_BUILD_HASH;

  const store = createStore({
    storageKey: 'voicelist.universal-interface.state',
    storage: window.localStorage,
    seedState
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

  const adapter = createBackendAdapter();
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
  bootstrapVoiceControl(app);
}

function bootstrapVoiceControl(app) {
  const button = document.getElementById('voice-btn');
  const statusEl = document.getElementById('voice-status');
  const transcriptEl = document.getElementById('voice-transcript');
  const helpButton = document.getElementById('voice-help-btn');
  const helpList = document.getElementById('voice-help');

  if (helpButton && helpList) {
    helpButton.addEventListener('click', () => {
      helpList.hidden = !helpList.hidden;
      helpButton.setAttribute('aria-expanded', String(!helpList.hidden));
    });
  }

  const supported = isSpeechRecognitionSupported(window);
  let voiceUI = null;

  const controller = createVoiceController({
    recognitionFactory: createRecognitionFactory(window),
    dispatch: (input) => app.dispatch(input),
    getState: () => app.getState(),
    onStatus: (event) => voiceUI?.onStatus(event)
  });

  voiceUI = createVoiceUI({ button, statusEl, transcriptEl, controller, supported });

  // Exposed so end-to-end tests can drive commands without a real microphone.
  window.__voiceControl__ = controller;
}

bootstrapListManagerPreview().catch((error) => {
  console.error('Failed to bootstrap list manager', error);
});
