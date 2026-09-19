import { calculateFrontier } from '../../list-frontier.js';
import { compareByDeadline } from '../../task-deadline.js';
import { findCandidates } from '../../resolver.js';
import { adaptSnapshot } from '../../snapshot-adapter.js';
import { clone, views } from './contracts.js';
const node = (type, id, props = {}, children = []) => ({ type, id, props, children });
const statusOrder = ['Focus', 'Open', 'Pause', 'Info', 'Done', 'Archive'];
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
      body = node('action-list', 'screen:log', {}, journal.actions().map(a => node('action', 'action:' + a.id, a)));
    } else if (view === 'action') {
      const a = journal.actions().find(x => x.id === context.actionId);
      body = a ? node('action-page', 'action:' + a.id, {
        ...a, title: a.label, sourceText: a.transcript || '', result: a.label,
        records: journal.chain(a.id).map(entry => entry.kind === 'text' ? {
          id: entry.id, kind: 'text', corrects: entry.corrects || null, userText: entry.text,
          answer: entry.answer || '', commands: clone(entry.commands || []), modelContext: clone(entry.modelContext || null)
        } : { id: entry.id, kind: 'ui', corrects: entry.corrects || null, command: clone(entry.command) })
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
