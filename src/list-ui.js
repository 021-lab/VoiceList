const AVAILABLE_TAGS = ['Важное', 'Срочно', 'Купить', 'Дом', 'Работа', 'Отложить'];
const STATUS_ACTIONS = new Set(['Open', 'Done', 'Focus', 'Archive', 'Pause']);
const PANEL_ITEM_HEIGHT = 72;

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

export function createUI({ rootPanel, header, viewToggleButton, frontierButton, undoButton, addButton, container, toastEl, dropPanel, tagPanel, overlay, input1, input2, modalTitle, btnConfirm, btnCancel, viewContent, viewLine1, viewLine2, viewTagsEl, actionLogPanel, taskPage, taskPageClose, taskPageSave, taskPageTitle, taskPageLine1, taskPageLine2, taskPageStatus, taskPageSubtasks, taskPageChildInput, taskPageAddChild }) {
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

  function showToast(message) {
    if (!toastEl) return;
    toastEl.textContent = message;
    toastEl.classList.add('show');
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => toastEl.classList.remove('show'), 1800);
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

    btnCancel.addEventListener('click', closeModal);
    overlay.addEventListener('click', (event) => {
      if (event.target === overlay) closeModal();
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
      const wantsLog = rootPanel.dataset.viewMode !== 'log';
      dispatchUserInput({
        actId: wantsLog ? 'actionLog' : 'list',
        actType: wantsLog ? 'panel' : 'list',
        command: wantsLog ? 'showActionLog' : 'showList',
        payload: {},
        source: 'view-toggle'
      });
    });

    frontierButton?.addEventListener('click', () => {
      const wantsFrontier = rootPanel.dataset.viewMode !== 'frontier';
      dispatchUserInput({
        actId: wantsFrontier ? 'frontier' : 'list',
        actType: wantsFrontier ? 'tab' : 'list',
        command: wantsFrontier ? 'showFrontier' : 'showList',
        payload: {},
        source: 'frontier-tab'
      });
    });

    document.addEventListener('touchmove', (event) => {
      if (!dragState) return;
      event.preventDefault();
      updateDrag(event.touches[0].clientY, event.touches[0].clientX);
    }, { passive: false });

    document.addEventListener('touchend', () => {
      if (dragState) finalizeDrag();
    }, { passive: true });

    document.addEventListener('touchcancel', () => {
      if (dragState) finalizeDrag();
    }, { passive: true });

    document.addEventListener('mousemove', (event) => {
      if (dragState) {
        updateDrag(event.clientY, event.clientX);
        return;
      }
      currentMouseGesture?.move(event.clientX, event.clientY);
    });

    document.addEventListener('mouseup', () => {
      if (dragState) {
        finalizeDrag();
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
    let nextLevel = 0;

    if (index > 0) {
      const aboveLevel = Number(all[index - 1].dataset.level || 0);
      nextLevel = dragState.rightShifted ? aboveLevel + 1 : aboveLevel;
    }

    dragState.pendingLevel = nextLevel;
    dragState.wrapper.dataset.level = String(nextLevel);
    dragState.wrapper.style.marginLeft = `${nextLevel * 24}px`;
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
    dragState = null;

    if (!rightShifted) {
      const all = [...container.querySelectorAll('.list-item-wrapper')];
      const index = all.indexOf(wrapper);
      const groupEnd = index + children.length;
      if (index > 0 && groupEnd < all.length - 1) {
        const aboveLevel = Number(all[index - 1].dataset.level || 0);
        const belowLevel = Number(all[groupEnd + 1].dataset.level || 0);
        if (belowLevel > aboveLevel) {
          const aboveWrapper = all[index - 1];
          container.insertBefore(wrapper, aboveWrapper);
          for (const child of children) container.insertBefore(child, aboveWrapper);
        }
      }
    }

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

      const action = dropAction;
      const wasRight = !!rdAnchor;
      const wasLeft = !!ldAnchor;
      const savedTagAction = tagAction;
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

      if (wasRight && dx > 30 && action) {
        if (isMouse) mouseSwipeDone = true;
        execPanelAction(action, itemId, 'right-swipe-panel');
      } else if (wasLeft && dx < -30 && savedTagAction) {
        if (isMouse) mouseSwipeDone = true;
        execPanelAction(savedTagAction, itemId, 'left-swipe-panel');
      } else if (!isMouse && Math.abs(dx) < 10 && Math.abs(dy) < 10 && rootPanel.dataset.viewMode !== 'frontier') {
        dispatchUserInput({ actId: itemId, actType: 'task', command: 'toggleCollapse', payload: {}, source: 'tap' });
      }
    };

    row.addEventListener('touchstart', (event) => {
      if (dragState || modalOpen || taskPageOpen) return;
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
        hideDrop();
        hideTagPanel();
        row.style.transform = '';
        actionBg.style.opacity = '0';
        if (rootPanel.dataset.viewMode === 'frontier') return;
        startDrag(wrapper, curY, curX);
      }, 370);
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
      if (dragState || modalOpen || taskPageOpen || event.button !== 0) return;
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
        hideDrop();
        hideTagPanel();
        row.style.transform = '';
        actionBg.style.opacity = '0';
        if (rootPanel.dataset.viewMode === 'frontier') return;
        startDrag(wrapper, curY, curX);
      }, 370);
    });

    row.addEventListener('click', () => {
      if (Date.now() - lastTouchEndTime < 500) return;
      if (dragState || wasDragging) return;
      if (mouseSwipeDone) {
        mouseSwipeDone = false;
        return;
      }
      if (rootPanel.dataset.viewMode === 'frontier') return;
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
    viewToggleButton.textContent = viewMode === 'log' ? 'Список' : 'Журнал';
    if (frontierButton) {
      frontierButton.textContent = viewMode === 'frontier' ? 'Список' : 'Фронтир';
      frontierButton.classList.toggle('active', viewMode === 'frontier');
    }
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
