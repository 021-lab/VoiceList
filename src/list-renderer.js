import { calculateFrontier } from './list-frontier.js';

function isArchived(item) {
  return String(item?.status || '').toLowerCase() === 'archive';
}

function escHtml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function groupByParent(items) {
  const byParent = new Map();

  for (const item of items) {
    const parentId = item.parentId || null;
    if (!byParent.has(parentId)) byParent.set(parentId, []);
    byParent.get(parentId).push(item);
  }

  for (const siblings of byParent.values()) {
    siblings.sort((left, right) => left.order - right.order);
  }

  return byParent;
}

function statusColor(status) {
  if (status === 'Done') return '#34c759';
  if (status === 'Focus') return '#ff9500';
  if (status === 'Archive') return '#8e8e93';
  if (status === 'Pause') return '#5856d6';
  return '#007aff';
}

export function createRenderer({ container, actionLogPanel, actionLogList, rootPanel, emptyStateLabel, bindRow, bindGlobal, onRendered } = {}) {
  const expandedFrontierParents = new Set();
  let lastState = null;

  function renderActionLog(actionLog) {
    if (!actionLogList) return;
    actionLogList.innerHTML = '';

    for (const entry of [...actionLog].reverse()) {
      const row = document.createElement('div');
      row.className = 'action-log-row';
      row.innerHTML = `
        <div class="action-log-line">
          <span class="action-log-label">${escHtml(entry.label)}</span>
          <span class="action-log-status action-log-status-${escHtml(entry.syncStatus)}">${escHtml(entry.syncStatus)}</span>
        </div>
        <div class="action-log-meta">${escHtml(entry.createdAt)}</div>
      `;
      actionLogList.appendChild(row);
    }
  }

  function renderRow({ actionBgClass = 'del', fragment, hasChildren, hidden = false, index, item, level, onRowClick = null }) {
    const wrapper = document.createElement('div');
    wrapper.className = 'list-item-wrapper';
    wrapper.dataset.id = item.id;
    wrapper.dataset.actId = item.id;
    wrapper.dataset.actType = 'task';
    wrapper.dataset.level = String(level);
    wrapper.style.marginLeft = `${level * 24}px`;
    if (hidden) wrapper.style.display = 'none';

    const actionBg = document.createElement('div');
    actionBg.className = `action-bg ${actionBgClass}`;
    actionBg.innerHTML = `<span class="action-icon">🗑</span><span class="action-label">Удалить</span>`;

    const row = document.createElement('div');
    row.className = 'list-item';
    row.dataset.actId = item.id;
    row.dataset.actType = 'task';
    const chevron = hasChildren ? `<span class="chevron">${item.collapsed ? '▶' : '▼'}</span>` : '';
    row.innerHTML = `
      <div class="item-head">
        <div class="item-copy">
          <div class="item-line1">${chevron}${escHtml(item.line1)}</div>
          ${item.tags?.length ? `<div class="item-tags">${item.tags.map((tag) => `<span class="item-tag">${escHtml(tag)}</span>`).join('')}</div>` : ''}
        </div>
        <div class="item-side">
          <span class="status-badge" style="--badge-color:${statusColor(item.status)}">${escHtml(item.status)}</span>
          <div class="item-index">${index}</div>
        </div>
      </div>
    `;

    wrapper.appendChild(actionBg);
    wrapper.appendChild(row);
    fragment.appendChild(wrapper);

    if (typeof bindRow === 'function') bindRow({ actionBg, item, level, row, wrapper });
    if (typeof onRowClick === 'function') row.addEventListener('click', onRowClick);
  }

  function renderFrontierParentRow({ fragment, parent }) {
    const wrapper = document.createElement('div');
    wrapper.className = 'list-item-wrapper frontier-parent-wrapper';
    wrapper.dataset.id = `parent:${parent.id}`;
    wrapper.dataset.actId = parent.id;
    wrapper.dataset.actType = 'task-parent-context';
    wrapper.dataset.level = '0';
    wrapper.style.marginLeft = '0px';

    const row = document.createElement('div');
    row.className = 'list-item frontier-parent-item';
    row.innerHTML = `
      <div class="item-head">
        <div class="item-copy">
          <div class="item-line1">${escHtml(parent.line1)}</div>
          ${parent.tags?.length ? `<div class="item-tags">${parent.tags.map((tag) => `<span class="item-tag">${escHtml(tag)}</span>`).join('')}</div>` : ''}
        </div>
        <div class="item-side">
          <span class="status-badge" style="--badge-color:${statusColor(parent.status)}">${escHtml(parent.status)}</span>
        </div>
      </div>
    `;

    wrapper.appendChild(row);
    fragment.appendChild(wrapper);
  }

  function renderFrontier(state) {
    const items = state.snapshot.items || [];
    const itemById = new Map(items.map((item) => [item.id, item]));
    const childIds = new Set(items.map((item) => item.parentId).filter(Boolean));
    let result;

    try {
      result = calculateFrontier(items);
    } catch (error) {
      container.innerHTML = `<div class="empty-state frontier-error"><div class="icon">⚠</div><p>${escHtml(error.message)}</p></div>`;
      return;
    }

    if (!result.frontier.length) {
      container.innerHTML = `<div class="empty-state"><div class="icon">◎</div><p>Во фронтире нет доступных задач.</p></div>`;
      return;
    }

const focusIds = new Set(result.focusHighlights.map((item) => item.id));
    const fragment = document.createDocumentFragment();

    if (result.focusHighlights.length) {
      const focusStrip = document.createElement('div');
      focusStrip.className = 'frontier-focus-strip';
      focusStrip.innerHTML = `
        <span class="frontier-focus-label">Фокус</span>
        ${result.focusHighlights.map((item) => `<span class="frontier-focus-chip">${escHtml(item.line1 || item.id)}</span>`).join('')}
      `;
      fragment.appendChild(focusStrip);
    }

    result.frontier.forEach((item, position) => {
      const parent = item.parentId ? itemById.get(item.parentId) : {
        id: '__root__',
        status: 'Open',
        line1: 'Мой список',
        tags: []
      };
      const parentExpanded = expandedFrontierParents.has(item.id);

      if (parentExpanded) {
        renderFrontierParentRow({ fragment, parent });
      }

      renderRow({
        actionBgClass: focusIds.has(item.id) ? 'focus' : 'del',
        fragment,
        hasChildren: childIds.has(item.id),
        index: position + 1,
        item,
        level: parentExpanded ? 1 : 0,
        onRowClick: parent ? () => {
          if (expandedFrontierParents.has(item.id)) expandedFrontierParents.delete(item.id);
          else expandedFrontierParents.add(item.id);
          render(lastState || state, 'frontier');
        } : null
      });
    });

    container.appendChild(fragment);
  }

  function render(state, viewMode = 'list') {
    if (!container) return;
    lastState = state;

    if (rootPanel) rootPanel.dataset.viewMode = viewMode;
    if (actionLogPanel) actionLogPanel.hidden = viewMode !== 'log';
    if (container) container.hidden = viewMode === 'log';

    container.innerHTML = '';
    renderActionLog(state.actionLog || []);

    if (viewMode === 'frontier') {
      renderFrontier(state);
      if (typeof bindGlobal === 'function') bindGlobal();
      if (typeof onRendered === 'function') onRendered(state, viewMode);
      return;
    }

    const items = state.snapshot.items || [];
    if (!items.length) {
      container.innerHTML = `<div class="empty-state"><div class="icon">📋</div><p>${escHtml(emptyStateLabel || 'Список пуст. Нажмите + чтобы добавить.')}</p></div>`;
      return;
    }

    const byParent = groupByParent(items);
    let index = 0;

    const fragment = document.createDocumentFragment();

    function walk(parentId, level, hidden) {
      for (const item of byParent.get(parentId) || []) {
        if (isArchived(item)) continue;
        index += 1;
        const visibleChildren = (byParent.get(item.id) || []).filter((child) => !isArchived(child));
        const hasChildren = visibleChildren.length > 0;
        renderRow({ fragment, hasChildren, hidden, index, item, level });

        walk(item.id, level + 1, hidden || !!item.collapsed);
      }
    }

    walk(null, 0, false);
    container.appendChild(fragment);

    if (typeof bindGlobal === 'function') bindGlobal();
    if (typeof onRendered === 'function') onRendered(state, viewMode);
  }

  return { render };
}
