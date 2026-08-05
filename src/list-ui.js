import { BrowserASR } from './asr-browser.js';
import { MockASR } from './asr-mock.js';
import { C as VOICE_C, selectAt as selectVoiceAt } from './gesture.js';
import { VoiceSession, validate as validateVoiceCommand } from './voice-session.js';

const AVAILABLE_TAGS = ['Важное', 'Срочно', 'Купить', 'Дом', 'Работа', 'Отложить'];
const STATUS_ACTIONS = new Set(['Open', 'Done', 'Focus', 'Archive', 'Pause']);
const PANEL_ITEM_HEIGHT = 72;
const VOICE_LONGPRESS_MS = 400;

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function escHtml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function findItem(state, id) {
  return state.snapshot.items.find((item) => item.id === id) || null;
}

function deriveArrangedFromWrappers(wrappers) {
  const stack = [];
  const siblingOrders = new Map();

  return wrappers.map((wrapper) => {
    const id = wrapper.dataset.id;
    const level = Number(wrapper.dataset.level || 0);

    while (stack.length > level) stack.pop();

    const parentId = level === 0 ? null : stack[level - 1];
    const currentOrder = (siblingOrders.get(parentId) || 0) + 10;
    siblingOrders.set(parentId, currentOrder);
    stack[level] = id;
    stack.length = level + 1;

    return { id, parentId, order: currentOrder };
  });
}

export function createUI({ rootPanel, header, viewToggleButton, frontierButton, settingsButton, undoButton, addButton, container, toastEl, dropPanel, tagPanel, voiceOverlay, overlay, input1, input2, modalTitle, btnConfirm, btnCancel, viewContent, viewLine1, viewLine2, viewTagsEl, actionLogPanel, taskPage, taskPageClose, taskPageSave, taskPageTitle, taskPageLine1, taskPageLine2, taskPageStatus, taskPageSubtasks, taskPageChildInput, taskPageAddChild, settingsOverlay, settingsClose, workflowyUrlInput, workflowyImportButton, workflowyImportStatus }) {
  let dispatchUserInput = () => {};
  let getState = () => ({ snapshot: { items: [] }, actionLog: [] });
  let boundGlobals = false;
  let undoSnapshot = null;
  let modalMode = 'add';
  let modalTargetId = null;
  let modalOpen = false;
  let dragState = null;
  let autoScrollRAF = null;
  let lastTouchEndTime = 0;
  let wasDragging = false;
  let currentMouseGesture = null;
  let dropAction = null;
  let tagAction = null;
  let taskPageOpen = false;
  let taskPageTargetId = null;
  let settingsOpen = false;
  let voiceState = null;
  let voicePressTimer = null;
  let voicePress = null;

  function showToast(message) {
    if (!toastEl) return;
    toastEl.textContent = message;
    toastEl.classList.add('show');
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => toastEl.classList.remove('show'), 1800);
  }

  function itemLabel(id) {
    return findItem(getState(), id)?.line1 || '';
  }

  function logEntryLabel(id) {
    return getState().actionLog?.find((entry) => entry.id === id)?.label || 'Запись журнала';
  }

  function createVoiceAsr() {
    const testConfig = window.__voiceTest;
    if (testConfig) return new MockASR(testConfig);
    return new BrowserASR({ finalTimeoutMs: VOICE_C.FINAL_TIMEOUT_MS });
  }

  function renderVoiceOverlay(session = voiceState?.session) {
    if (!voiceOverlay || !voiceState) return;
    if (voiceState.kind === 'log-comment') {
      renderLogCommentOverlay();
      return;
    }
    const activeSession = voiceState?.session;
    if (!voiceOverlay || !activeSession) return;
    if (session && session !== activeSession) return;
    session = activeSession;
    const selection = selectVoiceAt(voiceState?.dy || 0, session.overlay.stack.length, VOICE_C);
    const fingerLabel = labelVoiceFinger(session);
    const candidateRows = session.overlay.stack
      .map((candidate, index) => ({
        label: candidate.label,
        selected: selection.zone === 'candidate' && selection.index === index,
        overlayIndex: index
      }))
      .reverse();
    const fingerSelected = selection.zone !== 'cancel' && selection.zone !== 'candidate';
    const cancelSelected = selection.zone === 'cancel';

    voiceOverlay.innerHTML = `
      <div class="voice-transcript">${escHtml(session.text || 'Слушаю...')}</div>
      ${session.context ? `<div class="voice-context">${escHtml(itemLabel(session.context))}</div>` : ''}
      <div class="voice-stack">
        ${candidateRows.map((row) => `<div class="voice-candidate${row.selected ? ' selected' : ''}" data-voice-index="${row.overlayIndex}">${escHtml(row.label)}</div>`).join('')}
        <div class="voice-candidate voice-finger${fingerSelected ? ' selected' : ''}" data-voice-index="-1">${escHtml(fingerLabel)}</div>
        <div class="voice-cancel${cancelSelected ? ' selected' : ''}">Отмена</div>
      </div>
    `;
    voiceOverlay.classList.add('open');
    positionVoiceOverlay();
  }

  function renderLogCommentOverlay() {
    const selection = selectVoiceAt(voiceState?.dy || 0, 0, VOICE_C);
    const fingerSelected = selection.zone !== 'cancel';
    const cancelSelected = selection.zone === 'cancel';
    const transcript = voiceState.finalText || voiceState.transcript || 'Слушаю комментарий...';

    voiceOverlay.innerHTML = `
      <div class="voice-transcript">${escHtml('Комментарий')}</div>
      <div class="voice-context">${escHtml(logEntryLabel(voiceState.logEntryId))}</div>
      <div class="voice-stack">
        <div class="voice-candidate voice-finger${fingerSelected ? ' selected' : ''}" data-voice-index="-1">${escHtml(transcript)}</div>
        <div class="voice-cancel${cancelSelected ? ' selected' : ''}">Отмена</div>
      </div>
    `;
    voiceOverlay.classList.add('open');
    positionVoiceOverlay();
  }

  function labelVoiceCommand(command) {
    if (!command) return '';
    if (command.command === 'addChild') return `Добавить: задачу ${command.payload.line1}`;
    if (command.command === 'setStatus') return `Статус: ${command.payload.status}`;
    if (command.command === 'setParent') return 'Перенести';
    if (command.command === 'editItem') return `Переименовать: ${command.payload.line1}`;
    if (command.command === 'showSearch') return `Поиск: ${command.payload.query}`;
    if (command.command === 'undo') return 'Отменить';
    return command.command;
  }

  function labelVoiceFinger(session) {
    const selected = session.overlay.finger;
    if (selected?.kind === 'command') return labelVoiceCommand(selected.command);
    if (selected?.kind === 'blocked') return 'Не найдено';
    return session.text || 'Слушаю...';
  }

  function positionVoiceOverlay() {
    if (!voiceOverlay || !voiceState) return;
    const fingerRow = voiceOverlay.querySelector('.voice-finger');
    if (!fingerRow) return;
    const overlayRect = voiceOverlay.getBoundingClientRect();
    const fingerRect = fingerRow.getBoundingClientRect();
    const fingerCenter = fingerRect.top + (fingerRect.height / 2);
    const unclampedTop = overlayRect.top + (voiceState.anchorY - fingerCenter);
    const maxTop = Math.max(12, window.innerHeight - overlayRect.height - 12);
    const nextTop = Math.min(Math.max(12, unclampedTop), maxTop);
    voiceOverlay.style.top = `${Math.round(nextTop)}px`;
  }

  function isFrontierView() {
    return rootPanel.dataset.viewMode === 'frontier';
  }

  function startVoice(contextId, anchorY) {
    if (!isFrontierView()) return false;
    if (voiceState || modalOpen || settingsOpen || taskPageOpen) return false;
    hideDrop();
    hideTagPanel();
    const asr = createVoiceAsr();
    const session = new VoiceSession({
      tasks: getState().snapshot.items,
      asr,
      onUpdate: renderVoiceOverlay
    });
    voiceState = { session, anchorY, dy: 0 };
    const armed = session.arm(contextId);
    if (!armed) {
      const message = session.messages.at(-1) || 'Нет доступа к микрофону';
      showToast(message);
      voiceState = null;
      return false;
    }
    renderVoiceOverlay(session);
    asr.speak?.();
    renderVoiceOverlay(session);
    return true;
  }

  function startLogCommentVoice(logEntryId, anchorY) {
    if (voiceState || modalOpen || settingsOpen || taskPageOpen) return false;
    const asr = createVoiceAsr();
    const nextVoiceState = {
      kind: 'log-comment',
      asr,
      logEntryId,
      anchorY,
      dy: 0,
      transcript: '',
      finalText: '',
      error: null
    };
    voiceState = nextVoiceState;
    const armed = asr.start({
      onInterim(text) {
        if (voiceState !== nextVoiceState) return;
        nextVoiceState.transcript = text;
        renderVoiceOverlay();
      },
      onFinal(text) {
        if (voiceState !== nextVoiceState) return;
        nextVoiceState.finalText = text;
        nextVoiceState.transcript = text;
        renderVoiceOverlay();
      },
      onError(code) {
        if (voiceState !== nextVoiceState) return;
        nextVoiceState.error = code;
      }
    });
    if (!armed) {
      voiceState = null;
      showToast('Нет доступа к микрофону');
      return false;
    }
    renderVoiceOverlay();
    asr.speak?.();
    renderVoiceOverlay();
    return true;
  }

  function resetRowGesture(row, actionBg) {
    row.classList.remove('pressing');
    row.style.transition = 'transform 0.3s cubic-bezier(.4,0,.2,1)';
    row.style.transform = '';
    actionBg.style.opacity = '0';
    hideDrop();
    hideTagPanel();
  }

  function updateVoice(clientY) {
    if (!voiceState) return;
    voiceState.dy = voiceState.anchorY - clientY;
    if (voiceState.session) voiceState.session.move(voiceState.dy);
    renderVoiceOverlay();
  }

  async function finishLogCommentVoice(current) {
    const stopped = current.asr.stop();
    if (stopped && typeof stopped.then === 'function') await stopped;
    if (current.error === 'aborted') {
      showToast('Распознавание прервано');
      return;
    }
    if (current.error === 'timeout') {
      showToast('Не расслышал');
      return;
    }
    if (selectVoiceAt(current.dy || 0, 0, VOICE_C).zone === 'cancel') return;
    const text = String(current.finalText || current.transcript || '').trim();
    if (!text) {
      showToast('Не расслышал');
      return;
    }
    dispatchUserInput({
      actId: current.logEntryId,
      actType: 'log-entry',
      command: 'commentLogEntry',
      payload: { text },
      source: 'voice-log-comment'
    });
  }

  async function finishVoice() {
    if (!voiceState) return;
    const current = voiceState;
    voiceState = null;
    voiceOverlay?.classList.remove('open');
    if (current.kind === 'log-comment') {
      await finishLogCommentVoice(current);
      return;
    }
    const transcript = String(current.session.finalText || current.session.text || '').trim() || null;
    const result = await current.session.release(current.dy || 0);
    handleVoiceResult(result, { transcript });
  }

  async function finishVoiceAtDy(dy) {
    if (!voiceState) return;
    const current = voiceState;
    voiceState = null;
    voiceOverlay?.classList.remove('open');
    current.dy = dy;
    if (current.kind === 'log-comment') {
      await finishLogCommentVoice(current);
      return;
    }
    const transcript = String(current.session.finalText || current.session.text || '').trim() || null;
    const result = await current.session.release(dy);
    handleVoiceResult(result, { transcript });
  }

  function cancelVoice() {
    if (!voiceState) return;
    if (voiceState.kind === 'log-comment') voiceState.asr.stop();
    else voiceState.session.asr.stop();
    voiceState = null;
    voiceOverlay?.classList.remove('open');
  }

  function handleVoiceResult(result, meta = {}) {
    if (!result || result.action === 'cancel') {
      if (result?.why) showToast(result.why);
      return;
    }
    if (result.action === 'fallback') {
      showToast(result.text ? `Не понял: ${result.text}` : 'Не расслышал');
      return;
    }
    const command = result.command;
    if (!command) return;
    const state = getState();
    if (!validateVoiceCommand(command, state.snapshot.items)) {
      showToast('Задача изменилась, повторите');
      return;
    }
    if (command.command === 'undo') {
      if (!undoSnapshot) {
        showToast('Нечего отменять');
        return;
      }
      dispatchUserInput({ ...command, payload: { snapshot: clone(undoSnapshot) }, source: 'voice', transcript: meta.transcript });
      undoSnapshot = null;
      undoButton.disabled = true;
      return;
    }
    if (!String(command.command).startsWith('show') && command.command !== 'viewItem') saveUndoSnapshot();
    dispatchUserInput({ ...command, actType: command.actId ? 'task' : 'list', source: 'voice', transcript: meta.transcript });
  }

  function queueVoicePress(nextPress) {
    voicePress = nextPress;
    if (voicePressTimer) clearTimeout(voicePressTimer);
    voicePressTimer = setTimeout(() => {
      const currentPress = voicePress;
      voicePressTimer = null;
      if (!currentPress) return;
      if (currentPress.kind === 'log-comment') startLogCommentVoice(currentPress.logEntryId, currentPress.y);
      else startVoice(currentPress.contextId ?? null, currentPress.y);
    }, VOICE_LONGPRESS_MS);
  }

  function setDispatch(nextDispatch) {
    dispatchUserInput = nextDispatch;
  }

  function setGetState(nextGetState) {
    getState = nextGetState;
  }

  function buildRightPanelActions(item) {
    const status = item?.status || 'Open';
    return [
      { id: 'done', label: 'Done', icon: '✓', color: '#34c759', kind: 'status', status: 'Done' },
      { id: 'pause-toggle', label: status === 'Pause' ? 'Open' : 'Pause', icon: status === 'Pause' ? '◯' : 'Ⅱ', color: '#5856d6', kind: 'status', status: status === 'Pause' ? 'Open' : 'Pause' },
      { id: 'focus-toggle', label: status === 'Focus' ? 'Open' : 'Focus', icon: status === 'Focus' ? '◯' : '◎', color: '#ff9500', kind: 'status', status: status === 'Focus' ? 'Open' : 'Focus' },
      { id: 'archive', label: 'Archive', icon: '▣', color: '#8e8e93', kind: 'status', status: 'Archive', default: true },
      { id: 'edit-page', label: 'Edit', icon: '✎', color: '#007aff', kind: 'editPage' }
    ];
  }

  function buildTagPanelActions(item) {
    const itemTags = item?.tags || [];
    const notApplied = AVAILABLE_TAGS.filter((tag) => !itemTags.includes(tag));
    const applied = AVAILABLE_TAGS.filter((tag) => itemTags.includes(tag));
    return [...notApplied, ...applied].map((tag) => {
      const isApplied = itemTags.includes(tag);
      return {
        id: `tag:${tag}`,
        label: tag,
        icon: isApplied ? '−' : '#',
        color: isApplied ? '#636366' : '#5856d6',
        kind: 'tag',
        tag
      };
    });
  }

  function renderPanel(panelEl, actions) {
    panelEl.innerHTML = '';
    panelEl._actions = actions;
    actions.forEach((action, index) => {
      const div = document.createElement('div');
      div.className = 'panel-item';
      div.dataset.index = String(index);
      div.dataset.action = action.id;
      div.style.background = action.color;
      div.innerHTML = `<span class="panel-icon">${escHtml(action.icon)}</span><span class="panel-label">${escHtml(action.label)}</span>`;
      panelEl.appendChild(div);
    });
  }

  function hidePanel(panelEl) {
    panelEl.classList.remove('visible');
    panelEl.querySelectorAll('.panel-item').forEach((element) => element.classList.remove('active'));
  }

  function showPanel(panelEl, anchorY, defaultIndex = 0) {
    const count = panelEl.querySelectorAll('.panel-item').length;
    const totalHeight = count * PANEL_ITEM_HEIGHT;
    const targetTop = anchorY - (defaultIndex * PANEL_ITEM_HEIGHT) - (PANEL_ITEM_HEIGHT / 2);
    const top = Math.max(0, Math.min(targetTop, window.innerHeight - totalHeight));
    panelEl.style.top = `${top}px`;
    panelEl._anchorY = anchorY;
    panelEl.classList.add('visible');
  }

  function alignDefaultActionToAnchor(actions, anchorY) {
    const defaultIndex = actions.findIndex((action) => action.default);
    if (defaultIndex < 0) return { actions, defaultIndex: 0 };

    const totalHeight = actions.length * PANEL_ITEM_HEIGHT;
    const targetTop = anchorY - (defaultIndex * PANEL_ITEM_HEIGHT) - (PANEL_ITEM_HEIGHT / 2);
    const top = Math.max(0, Math.min(targetTop, window.innerHeight - totalHeight));
    const landedIndex = Math.max(0, Math.min(actions.length - 1, Math.floor((anchorY - top) / PANEL_ITEM_HEIGHT)));

    if (landedIndex !== defaultIndex) {
      const nextActions = [...actions];
      [nextActions[defaultIndex], nextActions[landedIndex]] = [nextActions[landedIndex], nextActions[defaultIndex]];
      return { actions: nextActions, defaultIndex: landedIndex };
    }

    return { actions, defaultIndex };
  }

  function setActiveItem(panelEl, dy) {
    const items = [...panelEl.querySelectorAll('.panel-item')];
    if (!items.length) return null;
    const top = parseFloat(panelEl.style.top);
    const index = Math.max(0, Math.min(items.length - 1, Math.floor((panelEl._anchorY + dy - top) / PANEL_ITEM_HEIGHT)));
    items.forEach((element, currentIndex) => element.classList.toggle('active', currentIndex === index));
    return panelEl._actions?.[index] || null;
  }

  function showDropPanel(anchorY, itemId) {
    const aligned = alignDefaultActionToAnchor(buildRightPanelActions(findItem(getState(), itemId)), anchorY);
    const actions = aligned.actions;
    const defaultIndex = aligned.defaultIndex;
    renderPanel(dropPanel, actions);
    showPanel(dropPanel, anchorY, defaultIndex);
    dropAction = setActiveItem(dropPanel, 0);
  }

  function hideDrop() {
    hidePanel(dropPanel);
    dropAction = null;
  }

  function showTagPanel(anchorY, itemId) {
    renderPanel(tagPanel, buildTagPanelActions(findItem(getState(), itemId)));
    showPanel(tagPanel, anchorY, 0);
    tagAction = setActiveItem(tagPanel, 0);
  }

  function hideTagPanel() {
    hidePanel(tagPanel);
    tagAction = null;
  }

  function openModal(mode, targetId, state = getState()) {
    modalMode = mode || 'add';
    modalTargetId = targetId || null;
    modalOpen = true;

    viewContent.style.display = 'none';
    input1.style.display = '';
    input2.style.display = '';
    btnConfirm.style.display = '';
    btnCancel.textContent = 'Отмена';
    input1.style.borderColor = '';

    if (mode === 'view') {
      const item = findItem(state, targetId);
      viewLine1.textContent = item?.line1 || '';
      viewLine2.textContent = item?.line2 || '';
      viewLine2.style.display = item?.line2 ? '' : 'none';
      viewTagsEl.innerHTML = (item?.tags || []).map((tag) => `<span class="item-tag">${escHtml(tag)}</span>`).join('');
      viewTagsEl.style.display = item?.tags?.length ? '' : 'none';
      viewContent.style.display = '';
      input1.style.display = 'none';
      input2.style.display = 'none';
      btnConfirm.style.display = 'none';
      btnCancel.textContent = 'Закрыть';
      modalTitle.textContent = 'Просмотр';
    } else if (mode === 'edit') {
      const item = findItem(state, targetId);
      input1.value = item?.line1 || '';
      input2.value = item?.line2 || '';
      modalTitle.textContent = 'Редактировать';
      btnConfirm.textContent = 'Сохранить';
    } else if (mode === 'nest') {
      input1.value = '';
      input2.value = '';
      modalTitle.textContent = 'Новый вложенный';
      btnConfirm.textContent = 'Добавить';
    } else {
      input1.value = '';
      input2.value = '';
      modalTitle.textContent = 'Новый элемент';
      btnConfirm.textContent = 'Добавить';
    }

    if (mode !== 'view') {
      input1.readOnly = true;
      input1.focus();
      input1.readOnly = false;
    }

    overlay.classList.add('open');
  }

  function closeModal() {
    overlay.classList.remove('open');
    modalOpen = false;
  }

  function openSettings() {
    if (!settingsOverlay) return;
    settingsOpen = true;
    settingsOverlay.classList.add('open');
    settingsOverlay.setAttribute('aria-hidden', 'false');
    if (workflowyImportStatus) workflowyImportStatus.textContent = '';
    workflowyUrlInput?.focus();
  }

  function closeSettings() {
    if (!settingsOverlay) return;
    settingsOpen = false;
    settingsOverlay.classList.remove('open');
    settingsOverlay.setAttribute('aria-hidden', 'true');
  }

  async function importWorkflowyFromSettings() {
    const url = workflowyUrlInput?.value.trim() || '';
    if (!url) {
      if (workflowyImportStatus) workflowyImportStatus.textContent = 'Введите ссылку Workflowy';
      return;
    }

    if (workflowyImportButton) workflowyImportButton.disabled = true;
    if (workflowyImportStatus) workflowyImportStatus.textContent = 'Импорт...';
    try {
      await dispatchUserInput({
        actId: 'workflowy-import',
        actType: 'settings',
        command: 'importWorkflowy',
        payload: { url },
        source: 'settings-import'
      });
      if (workflowyImportStatus) workflowyImportStatus.textContent = 'Импорт запущен';
      showToast('Импорт Workflowy запущен');
    } catch (error) {
      if (workflowyImportStatus) workflowyImportStatus.textContent = error.message || 'Импорт не выполнен';
      showToast('Импорт не выполнен');
    } finally {
      if (workflowyImportButton) workflowyImportButton.disabled = false;
    }
  }

  function confirmModal() {
    const line1 = input1.value.trim();
    const line2 = input2.value.trim();

    if (modalMode !== 'view' && !line1) {
      input1.focus();
      input1.style.borderColor = '#ff3b30';
      return;
    }

    input1.style.borderColor = '';

    if (modalMode === 'edit') {
      dispatchUserInput({
        actId: modalTargetId,
        actType: 'task',
        command: 'editItem',
        payload: { line1, line2 },
        source: 'modal-confirm'
      });
      showToast('Сохранено');
    } else if (modalMode === 'nest') {
      dispatchUserInput({
        actId: modalTargetId,
        actType: 'task',
        command: 'addChild',
        payload: { line1, line2 },
        source: 'modal-confirm'
      });
      showToast('Добавлен вложенный');
    } else if (modalMode === 'add') {
      dispatchUserInput({
        actId: 'list',
        actType: 'list',
        command: 'addItem',
        payload: { line1, line2 },
        source: 'modal-confirm'
      });
      showToast('Элемент добавлен');
    }

    closeModal();
  }

  function renderTaskPageSubtasks(itemId, state = getState()) {
    const children = state.snapshot.items
      .filter((item) => item.parentId === itemId)
      .sort((left, right) => (left.order || 0) - (right.order || 0));

    if (!children.length) {
      taskPageSubtasks.innerHTML = '<div class="task-page-empty">Подзадач пока нет</div>';
      return;
    }

    taskPageSubtasks.innerHTML = children.map((child) => `
      <div class="task-page-subtask">
        ${escHtml(child.line1)}
        ${child.line2 ? `<small>${escHtml(child.line2)}</small>` : ''}
      </div>
    `).join('');
  }

  function openTaskPage(itemId, state = getState()) {
    const item = findItem(state, itemId);
    if (!item || !taskPage) return;

    taskPageTargetId = itemId;
    taskPageOpen = true;
    taskPageTitle.textContent = item.line1 || 'Задача';
    taskPageLine1.value = item.line1 || '';
    taskPageLine2.value = item.line2 || '';
    taskPageStatus.value = item.status || 'Open';
    taskPageChildInput.value = '';
    renderTaskPageSubtasks(itemId, state);
    taskPage.classList.add('open');
    taskPage.setAttribute('aria-hidden', 'false');
  }

  function closeTaskPage() {
    if (!taskPage) return;
    taskPage.classList.remove('open');
    taskPage.setAttribute('aria-hidden', 'true');
    taskPageOpen = false;
    taskPageTargetId = null;
  }

  function saveTaskPage() {
    if (!taskPageTargetId) return;
    const line1 = taskPageLine1.value.trim();
    const line2 = taskPageLine2.value.trim();
    const status = taskPageStatus.value;
    if (!line1) {
      taskPageLine1.focus();
      return;
    }

    saveUndoSnapshot();

    dispatchUserInput({
      actId: taskPageTargetId,
      actType: 'task',
      command: 'editItem',
      payload: { line1, line2 },
      source: 'task-page-save'
    });
    dispatchUserInput({
      actId: taskPageTargetId,
      actType: 'task',
      command: 'setStatus',
      payload: { status },
      source: 'task-page-save'
    });
    showToast('Сохранено');
    closeTaskPage();
  }

  function addTaskPageChild() {
    if (!taskPageTargetId) return;
    const line1 = taskPageChildInput.value.trim();
    if (!line1) {
      taskPageChildInput.focus();
      return;
    }

    dispatchUserInput({
      actId: taskPageTargetId,
      actType: 'task',
      command: 'addChild',
      payload: { line1, line2: '' },
      source: 'task-page-add-child'
    });
    taskPageChildInput.value = '';
    renderTaskPageSubtasks(taskPageTargetId);
    showToast('Добавлен вложенный');
  }

  function bindGlobal() {
    if (boundGlobals) return;
    boundGlobals = true;

    addButton.addEventListener('click', () => {
      dispatchUserInput({ actId: 'list', actType: 'list', command: 'showAddModal', payload: {}, source: 'add-button' });
    });
    settingsButton?.addEventListener('click', openSettings);
    settingsClose?.addEventListener('click', closeSettings);
    settingsOverlay?.addEventListener('click', (event) => {
      if (event.target === settingsOverlay) closeSettings();
    });
    workflowyImportButton?.addEventListener('click', () => {
      importWorkflowyFromSettings();
    });
    workflowyUrlInput?.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') importWorkflowyFromSettings();
    });

    btnCancel.addEventListener('click', closeModal);
    overlay.addEventListener('click', (event) => {
      if (event.target === overlay) closeModal();
    });
    voiceOverlay?.addEventListener('click', (event) => {
      if (!voiceState) return;
      const cancelEl = event.target.closest('.voice-cancel');
      if (cancelEl) {
        finishVoiceAtDy(-VOICE_C.CANCEL_ZONE_PX);
        return;
      }

      const candidateEl = event.target.closest('.voice-candidate');
      if (!candidateEl) return;

      if (voiceState.kind === 'log-comment') {
        finishVoiceAtDy(0);
        return;
      }

      const overlayIndex = Number(candidateEl.dataset.voiceIndex || '-1');
      if (overlayIndex < 0) {
        finishVoiceAtDy(0);
        return;
      }

      finishVoiceAtDy(VOICE_C.DEADZONE_PX + (overlayIndex * VOICE_C.ROW_H_PX) + 1);
    });
    btnConfirm.addEventListener('click', confirmModal);
    [input1, input2].forEach((input) => input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') confirmModal();
    }));
    taskPageClose?.addEventListener('click', closeTaskPage);
    taskPageSave?.addEventListener('click', saveTaskPage);
    taskPageAddChild?.addEventListener('click', addTaskPageChild);
    taskPageChildInput?.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') addTaskPageChild();
    });

    undoButton.addEventListener('click', () => {
      if (!undoSnapshot) return;
      dispatchUserInput({
        actId: 'list',
        actType: 'list',
        command: 'undo',
        payload: { snapshot: clone(undoSnapshot) },
        source: 'undo-button'
      });
      undoSnapshot = null;
      undoButton.disabled = true;
      showToast('Отменено');
    });

    viewToggleButton.addEventListener('click', () => {
      const wantsLog = rootPanel.dataset.viewMode !== 'log' && rootPanel.dataset.viewMode !== 'search';
      dispatchUserInput({
        actId: wantsLog ? 'actionLog' : 'list',
        actType: wantsLog ? 'panel' : 'list',
        command: wantsLog ? 'showActionLog' : 'showList',
        payload: {},
        source: 'view-toggle'
      });
    });

    frontierButton?.addEventListener('click', () => {
      const wantsFrontier = rootPanel.dataset.viewMode !== 'frontier' && rootPanel.dataset.viewMode !== 'search';
      dispatchUserInput({
        actId: wantsFrontier ? 'frontier' : 'list',
        actType: wantsFrontier ? 'tab' : 'list',
        command: wantsFrontier ? 'showFrontier' : 'showList',
        payload: {},
        source: 'frontier-tab'
      });
    });

    container.addEventListener('mousedown', (event) => {
      if (event.button !== 0 || event.target.closest('.list-item,button,input,select,textarea')) return;
      queueVoicePress({ kind: 'command', contextId: null, x: event.clientX, y: event.clientY });
    });

    container.addEventListener('touchstart', (event) => {
      if (event.target.closest('.list-item,button,input,select,textarea')) return;
      const touch = event.touches[0];
      queueVoicePress({ kind: 'command', contextId: null, x: touch.clientX, y: touch.clientY });
    }, { passive: true });

    actionLogPanel?.addEventListener('mousedown', (event) => {
      if (event.button !== 0) return;
      const row = event.target.closest('.action-log-row');
      if (!row?.dataset.logId) return;
      queueVoicePress({ kind: 'log-comment', logEntryId: row.dataset.logId, x: event.clientX, y: event.clientY });
    });

    actionLogPanel?.addEventListener('touchstart', (event) => {
      const row = event.target.closest('.action-log-row');
      if (!row?.dataset.logId) return;
      const touch = event.touches[0];
      queueVoicePress({ kind: 'log-comment', logEntryId: row.dataset.logId, x: touch.clientX, y: touch.clientY });
    }, { passive: true });

    document.addEventListener('touchmove', (event) => {
      if (voicePressTimer) {
        const touch = event.touches[0];
        if (Math.abs(touch.clientX - voicePress.x) > 10 || Math.abs(touch.clientY - voicePress.y) > 10) {
          clearTimeout(voicePressTimer);
          voicePressTimer = null;
        }
      }
      if (voiceState) {
        event.preventDefault();
        updateVoice(event.touches[0].clientY);
        return;
      }
      if (!dragState) return;
      event.preventDefault();
      updateDrag(event.touches[0].clientY, event.touches[0].clientX);
    }, { passive: false });

    document.addEventListener('touchend', () => {
      if (voicePressTimer) {
        clearTimeout(voicePressTimer);
        voicePressTimer = null;
      }
      if (voiceState) {
        finishVoice();
        return;
      }
      if (dragState) finalizeDrag();
    }, { passive: true });

    document.addEventListener('touchcancel', () => {
      if (voicePressTimer) {
        clearTimeout(voicePressTimer);
        voicePressTimer = null;
      }
      if (voiceState) {
        cancelVoice();
        return;
      }
      if (dragState) finalizeDrag();
    }, { passive: true });

    document.addEventListener('mousemove', (event) => {
      if (voicePressTimer && (Math.abs(event.clientX - voicePress.x) > 10 || Math.abs(event.clientY - voicePress.y) > 10)) {
        clearTimeout(voicePressTimer);
        voicePressTimer = null;
      }
      if (voiceState) {
        updateVoice(event.clientY);
        return;
      }
      if (dragState) {
        updateDrag(event.clientY, event.clientX);
        return;
      }
      currentMouseGesture?.move(event.clientX, event.clientY);
    });

    document.addEventListener('mouseup', () => {
      if (voicePressTimer) {
        clearTimeout(voicePressTimer);
        voicePressTimer = null;
      }
      if (voiceState) {
        finishVoice();
        return;
      }
      if (dragState) {
        finalizeDrag();
        currentMouseGesture = null;
        wasDragging = true;
        setTimeout(() => { wasDragging = false; }, 300);
      } else {
        currentMouseGesture?.end();
      }
    });
  }

  function saveUndoSnapshot() {
    undoSnapshot = clone(getState().snapshot);
    undoButton.disabled = false;
  }

  function startAutoScroll() {
    if (autoScrollRAF) return;
    autoScrollRAF = requestAnimationFrame(autoScrollStep);
  }

  function stopAutoScroll() {
    if (autoScrollRAF) {
      cancelAnimationFrame(autoScrollRAF);
      autoScrollRAF = null;
    }
  }

  function syncChildTransforms() {
    if (!dragState?.children?.length) return;
    const transform = `translateY(${dragState.offsetY}px)`;
    for (const child of dragState.children) child.style.transform = transform;
  }

  function clampToBoundary() {
    if (!dragState) return;
    const viewportHeight = window.innerHeight;
    const headerBottom = header.getBoundingClientRect().bottom;
    const rect = dragState.wrapper.getBoundingClientRect();

    if (rect.top < headerBottom - 30) {
      dragState.offsetY += (headerBottom - 30) - rect.top;
      dragState.wrapper.style.transform = `translateY(${dragState.offsetY}px)`;
    } else if (rect.bottom > viewportHeight + 30) {
      dragState.offsetY -= rect.bottom - (viewportHeight + 30);
      dragState.wrapper.style.transform = `translateY(${dragState.offsetY}px)`;
    }
  }

  function detectRightShift(clientX) {
    if (dragState.nestBaseX === undefined) {
      dragState.nestBaseX = clientX;
      return;
    }
    if (clientX - dragState.nestBaseX > 36 && !dragState.rightShifted) {
      dragState.rightShifted = true;
      dragState.nestBaseX = clientX;
    }
  }

  function updateDraggedLevel() {
    const all = [...container.querySelectorAll('.list-item-wrapper')];
    const index = all.indexOf(dragState.wrapper);
    let nextLevel = dragState.originalLevel;

    if (dragState.rightShifted && index > 0) {
      const aboveLevel = Number(all[index - 1].dataset.level || 0);
      nextLevel = aboveLevel + 1;
    }

    dragState.pendingLevel = nextLevel;
    dragState.wrapper.dataset.level = String(nextLevel);
    dragState.wrapper.style.marginLeft = `${nextLevel * 24}px`;
  }

  function snapUnshiftedDragOutOfDeeperRows() {
    const { wrapper, children, originalLevel, offsetY } = dragState;
    if (offsetY >= 0) return;

    const all = [...container.querySelectorAll('.list-item-wrapper')];
    const index = all.indexOf(wrapper);
    const groupEnd = index + children.length;
    const aboveLevel = index > 0 ? Number(all[index - 1].dataset.level || 0) : originalLevel;
    const belowLevel = groupEnd < all.length - 1 ? Number(all[groupEnd + 1].dataset.level || 0) : originalLevel;
    if (aboveLevel <= originalLevel && belowLevel <= originalLevel) return;

    let ancestor = null;
    for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
      if (Number(all[cursor].dataset.level || 0) <= originalLevel) {
        ancestor = all[cursor];
        break;
      }
    }
    if (!ancestor) return;
    container.insertBefore(wrapper, ancestor);
    for (const child of children) container.insertBefore(child, ancestor);
  }

  function checkSwap() {
    const { wrapper, children } = dragState;
    const all = [...container.querySelectorAll('.list-item-wrapper')];
    const index = all.indexOf(wrapper);
    const rect = wrapper.getBoundingClientRect();
    const midpoint = rect.top + rect.height / 2;

    if (index > 0) {
      const previous = all[index - 1];
      const previousRect = previous.getBoundingClientRect();
      if (midpoint < previousRect.top + previousRect.height / 2) {
        container.insertBefore(wrapper, previous);
        for (const child of children) container.insertBefore(child, previous);
        dragState.offsetY += previousRect.height + 1;
        wrapper.style.transform = `translateY(${dragState.offsetY}px)`;
        return;
      }
    }

    const groupEnd = index + children.length;
    if (groupEnd < all.length - 1) {
      const next = all[groupEnd + 1];
      const nextRect = next.getBoundingClientRect();
      if (midpoint > nextRect.top + nextRect.height / 2) {
        container.insertBefore(next, wrapper);
        dragState.offsetY -= nextRect.height + 1;
        wrapper.style.transform = `translateY(${dragState.offsetY}px)`;
      }
    }
  }

  function updateDrag(clientY, clientX) {
    dragState.offsetY += (clientY - dragState.lastClientY) * 2;
    dragState.lastClientY = clientY;
    dragState.wrapper.style.transform = `translateY(${dragState.offsetY}px)`;
    clampToBoundary();
    checkSwap();
    syncChildTransforms();
    if (clientX !== undefined) {
      detectRightShift(clientX);
      updateDraggedLevel();
    }
  }

  function autoScrollStep() {
    if (!dragState) {
      autoScrollRAF = null;
      return;
    }

    const viewportHeight = window.innerHeight;
    const headerBottom = header.getBoundingClientRect().bottom;
    const rect = dragState.wrapper.getBoundingClientRect();
    const all = [...container.querySelectorAll('.list-item-wrapper')];
    const index = all.indexOf(dragState.wrapper);
    const dragIsFirst = index === 0;
    const dragIsLast = index + dragState.children.length === all.length - 1;

    if (rect.top < headerBottom && !dragIsFirst) {
      const speed = -Math.min((headerBottom - rect.top) / 5 + 2, 14);
      window.scrollBy(0, speed);
      dragState.offsetY += speed;
      dragState.wrapper.style.transform = `translateY(${dragState.offsetY}px)`;
      clampToBoundary();
      checkSwap();
      syncChildTransforms();
      updateDraggedLevel();
    } else if (rect.bottom > viewportHeight && !dragIsLast) {
      const speed = Math.min((rect.bottom - viewportHeight) / 5 + 2, 14);
      window.scrollBy(0, speed);
      dragState.offsetY += speed;
      dragState.wrapper.style.transform = `translateY(${dragState.offsetY}px)`;
      clampToBoundary();
      checkSwap();
      syncChildTransforms();
      updateDraggedLevel();
    }

    autoScrollRAF = requestAnimationFrame(autoScrollStep);
  }

  function startDrag(wrapper, clientY, clientX) {
    saveUndoSnapshot();
    const row = wrapper.querySelector('.list-item');
    row.classList.remove('pressing');
    row.style.transition = 'none';
    wrapper.classList.add('is-dragging');
    const allWrappers = [...container.querySelectorAll('.list-item-wrapper')];
    const startIndex = allWrappers.indexOf(wrapper);
    const baseLevel = Number(wrapper.dataset.level || 0);
    const children = [];

    for (let index = startIndex + 1; index < allWrappers.length; index += 1) {
      if (Number(allWrappers[index].dataset.level || 0) <= baseLevel) break;
      children.push(allWrappers[index]);
    }

    dragState = { wrapper, lastClientY: clientY, offsetY: 0, nestBaseX: clientX, children, originalLevel: baseLevel };
    for (const child of children) child.style.display = 'none';
    startAutoScroll();
  }

  function finalizeDrag() {
    stopAutoScroll();
    const { wrapper, pendingLevel, rightShifted, children, originalLevel } = dragState;

    if (!rightShifted) {
      snapUnshiftedDragOutOfDeeperRows();
    }

    dragState = null;

    if (pendingLevel !== undefined) {
      const delta = pendingLevel - originalLevel;
      wrapper.dataset.level = String(pendingLevel);
      wrapper.style.marginLeft = `${pendingLevel * 24}px`;
      for (const child of children) {
        const nextLevel = Math.max(0, Number(child.dataset.level || 0) + delta);
        child.dataset.level = String(nextLevel);
        child.style.marginLeft = `${nextLevel * 24}px`;
      }
    }

    const row = wrapper.querySelector('.list-item');
    row.style.transition = '';
    wrapper.style.transition = 'transform 0.22s cubic-bezier(.4,0,.2,1)';
    wrapper.style.transform = '';
    for (const child of children) {
      child.style.transition = 'transform 0.22s cubic-bezier(.4,0,.2,1)';
      child.style.transform = '';
    }

    setTimeout(() => {
      wrapper.style.transition = '';
      wrapper.classList.remove('is-dragging');
      const wrappers = [...container.querySelectorAll('.list-item-wrapper')];
      const arranged = deriveArrangedFromWrappers(wrappers);
      dispatchUserInput({
        actId: wrapper.dataset.id,
        actType: 'task',
        command: 'reorderItems',
        payload: { arranged },
        source: 'drag'
      });
    }, 220);
  }

  function execPanelAction(action, itemId, source) {
    if (!action) return;
    if (action.kind === 'editPage') {
      openTaskPage(itemId);
      return;
    }
    if (action.kind === 'status' && STATUS_ACTIONS.has(action.status)) {
      saveUndoSnapshot();
      dispatchUserInput({ actId: itemId, actType: 'task', command: 'setStatus', payload: { status: action.status }, source });
      showToast(`Статус: ${action.status}`);
      return;
    }
    if (action.kind === 'tag') {
      dispatchUserInput({ actId: itemId, actType: 'task', command: 'setTags', payload: { tag: action.tag }, source });
      showToast(`Тег: ${action.tag}`);
    }
  }

  function bindGesture(row, actionBg, itemId, wrapper) {
    let startX;
    let startY;
    let curX;
    let curY;
    let active = false;
    let longTimer = null;
    let lockedH = false;
    let rdAnchor = null;
    let ldAnchor = null;
    let mouseSwipeDone = false;

    const handleMoveXY = (clientX, clientY) => {
      if (dragState || !active) return;
      curX = clientX;
      curY = clientY;
      const dx = curX - startX;
      const dy = curY - startY;

      if (longTimer && (Math.abs(dx) > 8 || Math.abs(dy) > 8)) {
        clearTimeout(longTimer);
        longTimer = null;
        row.classList.remove('pressing');
      }

      if (!lockedH) {
        if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 6) {
          lockedH = true;
        } else if (Math.abs(dy) > Math.abs(dx) && Math.abs(dy) > 6) {
          row.style.transform = '';
          actionBg.style.opacity = '0';
          actionBg.className = 'action-bg del';
          active = false;
          currentMouseGesture = null;
          return;
        } else {
          return;
        }
      }

      const threshold = window.innerWidth * 0.20;
      const offsetY = curY - startY;

      if (rdAnchor !== null || (ldAnchor === null && dx > threshold)) {
        if (rdAnchor !== null && dx < threshold) {
          hideDrop();
          rdAnchor = null;
          row.style.transform = `translate(0px, ${offsetY}px)`;
          return;
        }
        if (!rdAnchor) {
          rdAnchor = row.getBoundingClientRect().top + row.offsetHeight / 2;
          showDropPanel(rdAnchor, itemId);
        }
        row.style.transform = `translate(${Math.min(dx, 110)}px, ${offsetY}px)`;
        actionBg.style.opacity = '0';
        dropAction = setActiveItem(dropPanel, offsetY);
        return;
      }

      if (ldAnchor !== null || dx < -threshold) {
        if (isFrontierView()) {
          ldAnchor = row.getBoundingClientRect().top + row.offsetHeight / 2;
          active = false;
          if (startVoice(itemId, curY)) {
            if (isFinite(curY)) updateVoice(curY);
          }
          resetRowGesture(row, actionBg);
          return;
        }
        if (ldAnchor !== null && dx > -threshold) {
          hideTagPanel();
          ldAnchor = null;
          row.style.transform = `translate(0px, ${offsetY}px)`;
          return;
        }
        if (!ldAnchor) {
          ldAnchor = row.getBoundingClientRect().top + row.offsetHeight / 2;
          showTagPanel(ldAnchor, itemId);
        }
        row.style.transform = `translate(${Math.max(dx, -110)}px, ${offsetY}px)`;
        actionBg.style.opacity = '0';
        tagAction = setActiveItem(tagPanel, offsetY);
        return;
      }

      row.style.transform = `translate(${dx}px, ${offsetY}px)`;
      actionBg.style.opacity = '0';
    };

    const handleEnd = (isMouse = false) => {
      clearTimeout(longTimer);
      longTimer = null;
      row.classList.remove('pressing');
      if (isMouse) currentMouseGesture = null;

      const rightAction = dropAction;
      const leftAction = tagAction;
      const wasRight = !!rdAnchor;
      const wasLeft = !!ldAnchor;
      rdAnchor = null;
      ldAnchor = null;
      hideDrop();
      hideTagPanel();

      if (dragState || !active) return;
      active = false;

      const dx = curX - startX;
      const dy = curY - startY;

      row.style.transition = 'transform 0.3s cubic-bezier(.4,0,.2,1)';
      row.style.transform = '';
      actionBg.style.opacity = '0';

      if (wasRight && dx > 30 && rightAction) {
        if (isMouse) mouseSwipeDone = true;
        execPanelAction(rightAction, itemId, 'right-swipe-panel');
      } else if (wasLeft && dx < -30 && leftAction) {
        if (isMouse) mouseSwipeDone = true;
        execPanelAction(leftAction, itemId, 'left-swipe-panel');
      } else if (!isMouse && Math.abs(dx) < 10 && Math.abs(dy) < 10 && rootPanel.dataset.viewMode !== 'frontier') {
        dispatchUserInput({ actId: itemId, actType: 'task', command: 'toggleCollapse', payload: {}, source: 'tap' });
      }
    };

    row.addEventListener('touchstart', (event) => {
      if (dragState || modalOpen || settingsOpen || taskPageOpen) return;
      const touch = event.touches[0];
      startX = curX = touch.clientX;
      startY = curY = touch.clientY;
      active = true;
      lockedH = false;
      rdAnchor = null;
      ldAnchor = null;
      row.style.transition = 'none';
      row.classList.add('pressing');
      longTimer = setTimeout(() => {
        longTimer = null;
        active = false;
        rdAnchor = null;
        ldAnchor = null;
        if (isFrontierView()) {
          startVoice(null, curY);
          resetRowGesture(row, actionBg);
          return;
        }
        actionBg.style.opacity = '0';
        hideDrop();
        hideTagPanel();
        startDrag(wrapper, curY, curX);
      }, VOICE_LONGPRESS_MS);
    }, { passive: true });

    row.addEventListener('touchmove', (event) => {
      handleMoveXY(event.touches[0].clientX, event.touches[0].clientY);
      if (lockedH) event.preventDefault();
    }, { passive: false });

    row.addEventListener('touchend', () => {
      lastTouchEndTime = Date.now();
      handleEnd(false);
    });
    row.addEventListener('touchcancel', () => handleEnd(false));

    row.addEventListener('mousedown', (event) => {
      if (Date.now() - lastTouchEndTime < 500) return;
      if (dragState || modalOpen || settingsOpen || taskPageOpen || event.button !== 0) return;
      startX = curX = event.clientX;
      startY = curY = event.clientY;
      active = true;
      lockedH = false;
      rdAnchor = null;
      ldAnchor = null;
      row.style.transition = 'none';
      row.classList.add('pressing');
      currentMouseGesture = { move: handleMoveXY, end: () => handleEnd(true) };
      longTimer = setTimeout(() => {
        longTimer = null;
        active = false;
        rdAnchor = null;
        ldAnchor = null;
        if (isFrontierView()) {
          startVoice(null, curY);
          resetRowGesture(row, actionBg);
          return;
        }
        currentMouseGesture = null;
        actionBg.style.opacity = '0';
        hideDrop();
        hideTagPanel();
        startDrag(wrapper, curY, curX);
      }, VOICE_LONGPRESS_MS);
    });

    row.addEventListener('click', () => {
      if (Date.now() - lastTouchEndTime < 500) return;
      if (dragState || wasDragging) return;
      if (mouseSwipeDone) {
        mouseSwipeDone = false;
        return;
      }
      if (isFrontierView()) return;
      dispatchUserInput({ actId: itemId, actType: 'task', command: 'toggleCollapse', payload: {}, source: 'click' });
    });
  }

  function bindRow({ actionBg, item, row, wrapper }) {
    if (row.dataset.bound === '1') return;
    row.dataset.bound = '1';
    bindGesture(row, actionBg, item.id, wrapper);
  }

  function onRendered(state, viewMode) {
    rootPanel.dataset.viewMode = viewMode;
    viewToggleButton.textContent = viewMode === 'log' || viewMode === 'search' ? 'Список' : 'Журнал';
    if (frontierButton) {
      frontierButton.textContent = viewMode === 'frontier' || viewMode === 'search' ? 'Список' : 'Фронтир';
      frontierButton.classList.toggle('active', viewMode === 'frontier');
    }
    if (taskPageOpen && taskPageTargetId) renderTaskPageSubtasks(taskPageTargetId, state);
  }

  return {
    bindGlobal,
    bindRow,
    onRendered,
    openModal,
    setDispatch,
    setGetState
  };
}
