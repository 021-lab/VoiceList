const STATUS_VALUES = new Set(['Open', 'Done', 'Focus', 'Archive', 'Pause']);

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

function buildLabel(command, payload = {}) {
  if (command === 'addItem') return `Создана задача: ${payload.line1}`;
  if (command === 'addChild') return `Создана подзадача: ${payload.line1}`;
  if (command === 'setStatus') return `Статус изменён: ${payload.status}`;
  if (command === 'editItem') return `Изменена задача: ${payload.line1}`;
  if (command === 'deleteItem') return 'Удалена задача';
  if (command === 'setTags') return `Изменены теги: ${payload.tag}`;
  if (command === 'toggleCollapse') return 'Переключено сворачивание';
  if (command === 'reorderItems') return 'Изменён порядок списка';
  if (command === 'undo') return 'Выполнен undo';
  return command;
}

function createLogEntry(input, patch) {
  const command = clone(input);
  if (command.payload) delete command.payload.line2;
  return {
    id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 5)}`.slice(0, 8),
    createdAt: new Date().toISOString(),
    command,
    patch,
    label: buildLabel(input.command, input.payload),
    syncStatus: 'pending'
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

export function createInterpreter() {
  return {
    execute(state, input) {
      const currentItems = clone(state.snapshot.items);
      const nextItems = clone(currentItems);
      const existingIds = new Set(nextItems.map((item) => item.id));
      let patch = [];
      let actionLogEntry = null;
      const payload = input.payload || {};

      if (input.command === 'showActionLog' || input.command === 'showList' || input.command === 'showFrontier') {
        return {
          patch: [],
          actionLogEntry: null,
          viewMode: input.command === 'showActionLog' ? 'log' :
            input.command === 'showFrontier' ? 'frontier' : 'list'
        };
      }

      if (input.command === 'showAddModal' || input.command === 'showEditModal' || input.command === 'showNestModal' || input.command === 'viewItem') {
        return {
          patch: [],
          actionLogEntry: null,
          effect: {
            type: 'modal',
            mode: input.command === 'showAddModal' ? 'add' :
              input.command === 'showEditModal' ? 'edit' :
              input.command === 'showNestModal' ? 'nest' : 'view',
            itemId: input.actId === 'list' ? null : input.actId
          }
        };
      }

      if (input.command === 'addItem') {
        nextItems.push({
          id: randomId(existingIds),
          parentId: null,
          order: nextOrder(nextItems, null),
          status: 'Open',
          line1: payload.line1,
          collapsed: false,
          tags: []
        });
      } else if (input.command === 'addChild') {
        nextItems.push({
          id: randomId(existingIds),
          parentId: input.actId,
          order: nextOrder(nextItems, input.actId),
          status: 'Open',
          line1: payload.line1,
          collapsed: false,
          tags: []
        });
      } else if (input.command === 'editItem') {
        const item = nextItems.find((candidate) => candidate.id === input.actId);
        if (item) {
          item.line1 = payload.line1;
        }
      } else if (input.command === 'setStatus') {
        const item = nextItems.find((candidate) => candidate.id === input.actId);
        if (item && STATUS_VALUES.has(payload.status)) item.status = payload.status;
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
      } else if (input.command === 'undo') {
        patch = [{ op: 'replace', path: '/snapshot', value: clone(payload.snapshot) }];
      }

      if (!patch.length) {
        patch = [{ op: 'replace', path: '/snapshot/items', value: nextItems }];
      }

      if (!String(input.command).startsWith('show') && input.command !== 'viewItem') {
        actionLogEntry = createLogEntry(input, patch);
      }

      return { patch, actionLogEntry };
    }
  };
}
