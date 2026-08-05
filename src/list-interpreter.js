import { findCandidates } from './resolver.js';
import { adaptSnapshot } from './snapshot-adapter.js';

const STATUS_VALUES = new Set(['Open', 'Done', 'Focus', 'Archive', 'Pause', 'Info']);
const INBOX_ID = 'inbox';

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function randomId(existingIds) {
  const alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789';
  let candidate = '';
  do {
    candidate = Array.from({ length: 5 }, () => alphabet[Math.floor(Math.random() * alphabet.length)]).join('');
  } while (existingIds.has(candidate));
  return candidate;
}

function nextOrder(items, parentId) {
  const siblings = items.filter((item) => item.parentId === parentId);
  if (!siblings.length) return 10;
  return Math.max(...siblings.map((item) => item.order || 0)) + 10;
}

function noChange() {
  return { patch: [], logEntryDraft: null };
}

function buildLabel(command, payload = {}) {
  if (command === 'addItem') return `Создана задача: ${payload.line1}`;
  if (command === 'addChild') return `Создана подзадача: ${payload.line1}`;
  if (command === 'setStatus') return `Статус изменён: ${payload.status}`;
  if (command === 'editItem') return `Изменена задача: ${payload.line1}`;
  if (command === 'deleteItem') return 'Удалена задача';
  if (command === 'setTags') return `Изменены теги: ${payload.tag}`;
  if (command === 'toggleCollapse') return 'Переключено сворачивание';
  if (command === 'reorderItems') return 'Изменён порядок списка';
  if (command === 'importWorkflowyTree') return `Импортировано дерево Workflowy: ${payload.tree?.title || 'без названия'}`;
  if (command === 'undo') return 'Выполнен undo';
  return command;
}

function createDefaultLogId() {
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 5)}`.slice(0, 8);
}

function createLogEntryDraft(input, patch, { createLogId = createDefaultLogId, now = () => new Date() } = {}) {
  return {
    id: createLogId(input, patch),
    createdAt: now().toISOString(),
    transcript: input.transcript ?? null,
    command: clone(input),
    patch,
    label: buildLabel(input.command, input.payload),
    syncStatus: 'pending',
    comments: []
  };
}

function descendants(items, rootId) {
  const blocked = new Set([rootId]);
  let changed = true;

  while (changed) {
    changed = false;
    for (const item of items) {
      if (!blocked.has(item.id) && blocked.has(item.parentId)) {
        blocked.add(item.id);
        changed = true;
      }
    }
  }

  return blocked;
}

function appendImportedTree({ createItemId, existingIds, input, nextItems, parentId, tree, order }) {
  const title = String(tree?.title || '').trim();
  if (!title) return null;
  const id = createItemId(existingIds, input);
  existingIds.add(id);
  nextItems.push({
    id,
    parentId,
    order,
    status: 'Open',
    line1: title,
    line2: '',
    collapsed: false,
    tags: []
  });

  let childOrder = 10;
  for (const child of tree.children || []) {
    appendImportedTree({
      createItemId,
      existingIds,
      input,
      nextItems,
      parentId: id,
      tree: child,
      order: childOrder
    });
    childOrder += 10;
  }

  return id;
}

export function createInterpreter({ createItemId = randomId, createLogId = createDefaultLogId, now = () => new Date() } = {}) {
  return {
    execute(state, input) {
      const currentItems = clone(state.snapshot.items);
      const nextItems = clone(currentItems);
      const existingIds = new Set(nextItems.map((item) => item.id));
      let patch = [];
      let logEntryDraft = null;
      const payload = input.payload || {};

      if (input.command === 'showActionLog' || input.command === 'showList' || input.command === 'showFrontier') {
        return {
          patch: [],
          logEntryDraft: null,
          viewMode: input.command === 'showActionLog' ? 'log' :
            input.command === 'showFrontier' ? 'frontier' : 'list'
        };
      }

      if (input.command === 'showSearch') {
        const query = String(payload.query || '').trim();
        const rows = findCandidates(query, adaptSnapshot(currentItems));
        return {
          patch: [],
          logEntryDraft: null,
          viewMode: 'search',
          effect: {
            type: 'search',
            query,
            itemIds: rows.map((row) => row.id)
          }
        };
      }

      if (input.command === 'showAddModal' || input.command === 'showEditModal' || input.command === 'showNestModal' || input.command === 'viewItem') {
        return {
          patch: [],
          logEntryDraft: null,
          effect: {
            type: 'modal',
            mode: input.command === 'showAddModal' ? 'add' :
              input.command === 'showEditModal' ? 'edit' :
              input.command === 'showNestModal' ? 'nest' : 'view',
            itemId: input.actId === 'list' ? null : input.actId
          }
        };
      }

      if ((input.command === 'editItem' || input.command === 'deleteItem' || input.command === 'setParent') &&
          input.actId === INBOX_ID) {
        return noChange();
      }

      if (input.command === 'addItem') {
        nextItems.push({
          id: createItemId(existingIds, input),
          parentId: null,
          order: nextOrder(nextItems, null),
          status: 'Open',
          line1: payload.line1,
          line2: payload.line2 || '',
          collapsed: false,
          tags: []
        });
      } else if (input.command === 'addChild') {
        nextItems.push({
          id: createItemId(existingIds, input),
          parentId: input.actId,
          order: nextOrder(nextItems, input.actId),
          status: 'Open',
          line1: payload.line1,
          line2: payload.line2 || '',
          collapsed: false,
          tags: []
        });
      } else if (input.command === 'editItem') {
        const item = nextItems.find((candidate) => candidate.id === input.actId);
        if (item) {
          item.line1 = payload.line1;
          item.line2 = payload.line2 || '';
        }
      } else if (input.command === 'setStatus') {
        const item = nextItems.find((candidate) => candidate.id === input.actId);
        if (item && STATUS_VALUES.has(payload.status)) item.status = payload.status;
      } else if (input.command === 'setParent') {
        const item = nextItems.find((candidate) => candidate.id === input.actId);
        const parentId = payload.parentId ?? null;
        if (!item || parentId === input.actId) return noChange();
        if (parentId !== null && !nextItems.some((candidate) => candidate.id === parentId)) return noChange();
        if (parentId !== null && descendants(nextItems, input.actId).has(parentId)) return noChange();
        item.parentId = parentId;
        item.order = nextOrder(nextItems.filter((candidate) => candidate.id !== input.actId), parentId);
      } else if (input.command === 'setTags') {
        const item = nextItems.find((candidate) => candidate.id === input.actId);
        if (item) {
          const tags = new Set(item.tags || []);
          if (tags.has(payload.tag)) tags.delete(payload.tag);
          else tags.add(payload.tag);
          item.tags = Array.from(tags);
        }
      } else if (input.command === 'toggleCollapse') {
        const item = nextItems.find((candidate) => candidate.id === input.actId);
        if (item) item.collapsed = !item.collapsed;
      } else if (input.command === 'deleteItem') {
        const deletedIds = descendants(nextItems, input.actId);
        patch = [
          {
            op: 'replace',
            path: '/snapshot/items',
            value: nextItems.filter((item) => !deletedIds.has(item.id))
          }
        ];
      } else if (input.command === 'reorderItems') {
        const arranged = new Map((payload.arranged || []).map((item) => [item.id, item]));
        patch = [
          {
            op: 'replace',
            path: '/snapshot/items',
            value: nextItems
              .map((item) => arranged.has(item.id) ? { ...item, ...arranged.get(item.id) } : item)
              .sort((left, right) => left.order - right.order)
          }
        ];
      } else if (input.command === 'importWorkflowyTree') {
        const rootId = appendImportedTree({
          createItemId,
          existingIds,
          input,
          nextItems,
          parentId: null,
          tree: payload.tree,
          order: nextOrder(nextItems, null)
        });
        if (!rootId) return noChange();
      } else if (input.command === 'undo') {
        patch = [{ op: 'replace', path: '/snapshot', value: clone(payload.snapshot) }];
      }

      if (!patch.length) {
        patch = [{ op: 'replace', path: '/snapshot/items', value: nextItems }];
      }

      if (!String(input.command).startsWith('show') &&
          input.command !== 'viewItem' &&
          input.command !== 'toggleCollapse') {
        logEntryDraft = createLogEntryDraft(input, patch, { createLogId, now });
      }

      return { patch, logEntryDraft };
    }
  };
}
