import { calculateFrontier } from '../../list-frontier.js';
import { compareByDeadline } from '../../task-deadline.js';
import { findCandidates } from '../../resolver.js';
import { adaptSnapshot } from '../../snapshot-adapter.js';
import { clone, views } from './contracts.js';
const node = (type, id, props = {}, children = []) => ({ type, id, props, children });
const statusOrder = ['Focus', 'Open', 'Pause', 'Info', 'Done', 'Archive'];
const statusLabels = { Open: 'Открыта', Focus: 'В фокусе', Pause: 'На паузе', Done: 'Выполнена', Archive: 'В архиве', Info: 'Информация' };
const commandLabels = {
  addItem: 'Добавить задачу', addChild: 'Добавить подзадачу', editItem: 'Изменить задачу',
  setStatus: 'Изменить статус', setParent: 'Изменить вложенность', setTags: 'Изменить теги',
  setDeadline: 'Изменить срок', toggleCollapse: 'Свернуть или развернуть задачу',
  deleteItem: 'Удалить задачу', reorderItems: 'Изменить порядок задач',
  importWorkflowyTree: 'Импортировать дерево задач',
  rollbackAction: 'Откатить действие и все его корректировки', undo: 'Откатить действие и все его корректировки',
  showList: 'Открыть список', showFrontier: 'Открыть фронтир', showActionLog: 'Открыть журнал действий',
  showSearch: 'Открыть поиск', showAddModal: 'Открыть создание задачи', showEditModal: 'Открыть редактирование задачи',
  showNestModal: 'Открыть изменение вложенности', viewItem: 'Открыть задачу',
  showSettings: 'Открыть настройки', showDialogues: 'Открыть диалоги'
};
const fieldLabels = {
  line1: 'Название', line2: 'Описание', status: 'Статус', parentId: 'Родительская задача',
  tag: 'Тег', tags: 'Теги', deadline: 'Срок', query: 'Запрос', title: 'Название', actionId: 'Действие'
};

function textValue(value, byId) {
  if (value == null || value === '') return 'Не задано';
  if (typeof value === 'boolean') return value ? 'Да' : 'Нет';
  if (typeof value === 'string') return statusLabels[value] || byId.get(value)?.line1 || value;
  if (typeof value === 'number') return String(value);
  if (Array.isArray(value)) return value.map(item => textValue(item, byId)).join(', ');
  if (typeof value === 'object') return Object.entries(value).map(([key, item]) => `${fieldLabels[key] || key}: ${textValue(item, byId)}`).join('; ');
  return String(value);
}

function targetLabel(command, byId) {
  const target = byId.get(command?.actId);
  if (target) return target.line1;
  if (!command?.actId || command.actId === 'list') return 'Список задач';
  if (['rollbackAction', 'undo'].includes(command.command)) return 'Текущее действие';
  return `Элемент ${command.actId}`;
}

function countTree(tree) {
  if (!tree || typeof tree !== 'object') return 0;
  return 1 + (Array.isArray(tree.children) ? tree.children.reduce((sum, child) => sum + countTree(child), 0) : 0);
}

function commandFields(command, byId) {
  const payload = command?.payload || {};
  if (command?.command === 'setStatus') return [{ label: 'Новый статус', value: textValue(payload.status, byId) }];
  if (command?.command === 'setParent') return [{ label: 'Новый родитель', value: payload.parentId == null ? 'Корень списка' : textValue(payload.parentId, byId) }];
  if (command?.command === 'setTags') return [{ label: 'Тег', value: textValue(payload.tag ?? payload.tags, byId) }];
  if (command?.command === 'setDeadline') return [{ label: 'Новый срок', value: textValue(payload.deadline, byId) }];
  if (command?.command === 'showSearch') return [{ label: 'Поисковый запрос', value: textValue(payload.query, byId) }];
  if (command?.command === 'reorderItems') {
    const arranged = Array.isArray(payload.arranged) ? payload.arranged : [];
    return arranged.map((item, index) => ({
      label: byId.get(item.id)?.line1 || `Элемент ${index + 1}`,
      value: `${item.parentId == null ? 'Корень списка' : `в ${textValue(item.parentId, byId)}`}, позиция ${Number(item.order) || index + 1}`
    }));
  }
  if (command?.command === 'importWorkflowyTree') return [{
    label: 'Дерево', value: `${payload.tree?.title || 'Без названия'} · ${countTree(payload.tree)} элементов`
  }];
  if (['rollbackAction', 'undo', 'toggleCollapse', 'deleteItem'].includes(command?.command)) return [];
  return Object.entries(payload)
    .filter(([key]) => key !== 'actionId')
    .map(([key, value]) => ({ label: fieldLabels[key] || key, value: textValue(value, byId) }));
}

function commandView(command, byId) {
  return {
    type: command?.command || 'unknown',
    actionLabel: commandLabels[command?.command] || `Команда «${command?.command || 'неизвестна'}»`,
    targetLabel: targetLabel(command, byId),
    fields: commandFields(command, byId)
  };
}

function recordTaskIndex(entry, currentById) {
  const contextual = new Map(currentById);
  for (const task of entry?.modelContext?.tasks || []) {
    if (task?.id) contextual.set(task.id, task);
  }
  return contextual;
}

/** How many actions the journal screen carries. */
export const LOG_PAGE_SIZE = 30;

export class Presentation {
  compose(graph, journal, context = {}) {
    const view = views.includes(context.view) ? context.view : 'list';
    const items = graph.items;
    const byId = new Map(items.map(x => [x.id, x]));
    const groups = new Map();
    for (const item of items) { const key = item.parentId || null; if (!groups.has(key)) groups.set(key, []); groups.get(key).push(item); }
    for (const siblings of groups.values()) siblings.sort((a,b) => a.order - b.order);
    const task = (item, level = 0, hidden = false) => node('task', 'task:' + item.id, {
      ...clone(item), taskId: item.id, level, hidden, hasChildren: (groups.get(item.id) || []).some(x => x.status !== 'Archive'),
      parent: item.parentId ? clone(byId.get(item.parentId)) : { id: '__root__', line1: 'Мой список', status: 'Open' },
      allowedActions: ['editItem','setStatus','setParent','setDeadline','setTags','addChild','toggleCollapse','reorderItems']
    });
    let body;
    if (view === 'log') {
      // The journal screen shows the recent past, not the whole history: a list that grows
      // without bound is downloaded in full every time it is opened and read at its end.
      const recent = journal.actions().slice(-LOG_PAGE_SIZE);
      body = node('action-list', 'screen:log', { total: journal.actions().length, shown: recent.length },
        recent.map(a => node('action', 'action:' + a.id, a)));
    } else if (view === 'action') {
      const a = journal.actions().find(x => x.id === context.actionId);
      body = a ? node('action-page', 'action:' + a.id, {
        ...a, title: 'История действия',
        records: journal.chain(a.id).map(entry => {
          if (entry.kind === 'speech') {
            return { id: entry.id, kind: 'speech', corrects: entry.corrects || null, role: entry.role || 'user',
              userText: entry.role === 'assistant' ? '' : entry.text, answer: entry.role === 'assistant' ? entry.text : '',
              source: entry.source || 'voice' };
          }
          if (entry.kind === 'text') {
            const recordById = recordTaskIndex(entry, byId);
            return {
              id: entry.id, kind: 'text', corrects: entry.corrects || null, userText: entry.text,
              answer: entry.answer || '', commands: clone(entry.commands || []),
              commandViews: (entry.commands || []).map(command => commandView(command, recordById)),
              modelContext: clone(entry.modelContext || null)
            };
          }
          return {
            id: entry.id, kind: 'ui', corrects: entry.corrects || null, command: clone(entry.command),
            source: entry.command?.source || 'ui', commandView: commandView(entry.command, byId)
          };
        })
      }) : node('text', 'screen:missing', { text: 'Действие не найдено' });
    } else if (view === 'edit') {
      const item = byId.get(context.taskId);
      const isAdd = context.mode === 'add' || !context.taskId;
      body = !item && !isAdd ? node('text', 'screen:missing', { text: 'Задача не найдена' }) : node('task-editor', item ? 'task:' + item.id : 'screen:add', {
        ...(item ? clone(item) : { line1: '', line2: '', status: 'Open' }),
        taskId: item?.id || null, mode: isAdd ? 'add' : 'edit', parentId: context.parentId || item?.parentId || null,
        parent: byId.get(item?.parentId || context.parentId) || null,
        subtasks: clone(groups.get(item?.id) || []).filter(x => x.status !== 'Archive').sort((a,b) => statusOrder.indexOf(a.status) - statusOrder.indexOf(b.status) || a.order - b.order),
        parentOptions: items.filter(x => x.id !== item?.id && x.status !== 'Archive').map(x => ({ id: x.id, line1: x.line1 }))
      });
    } else if (view === 'settings' || view === 'dialogues') {
      body = node(view, 'screen:' + view);
    } else {
      let rows = [], props = {};
      if (view === 'frontier') {
        const result = calculateFrontier(items);
        rows = [...result.frontier].sort(compareByDeadline).map(x => task(x));
        props.focusHighlights = result.focusHighlights.map(x => ({ id: x.id, line1: x.line1 }));
      } else if (view === 'search') {
        rows = findCandidates(context.query || '', adaptSnapshot(items)).map(x => byId.get(x.id)).filter(x => x && x.status !== 'Archive').map(x => task(x));
        props.query = context.query || '';
      } else {
        const walk = (id, level, hidden) => { for (const item of groups.get(id) || []) {
          if (item.status === 'Archive') continue;
          rows.push(task(item, level, hidden));
          walk(item.id, level + 1, hidden || item.collapsed);
        }};
        walk(null, 0, false);
      }
      body = node('task-list', 'screen:' + view, props, rows);
    }
    return { schemaVersion: 1, revision: graph.revision, cursor: journal.cursor, view,
      root: node('application', 'app', { title: 'Мой список' }, [
        node('toolbar', 'toolbar', { view }, [
          node('button', 'menu:list', { label: 'Список', view: 'list' }),
          node('button', 'menu:frontier', { label: 'Фронтир', view: 'frontier' }),
          node('button', 'menu:log', { label: 'Журнал', view: 'log' }),
          node('button', 'menu:settings', { label: 'Настройки', view: 'settings' })
        ]), body
      ])
    };
  }
}
