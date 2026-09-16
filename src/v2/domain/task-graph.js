import { createInterpreter } from '../../list-interpreter.js';
import { clone, fail, stable, statuses, graphCommands } from './contracts.js';

const inbox = { id: 'inbox', parentId: null, order: 0, status: 'Open', line1: 'Входящие', line2: '', collapsed: false, tags: [] };
export function validateItems(items) {
  const byId = new Map(items.map(item => [item.id, item]));
  if (byId.size !== items.length || !byId.has('inbox')) fail('INVALID_GRAPH', 'Некорректные ID задач');
  for (const item of items) {
    if (!item.id || !statuses.includes(item.status) || typeof item.line1 !== 'string' || !item.line1.trim()) fail('INVALID_GRAPH', 'Некорректная задача');
    const seen = new Set([item.id]);
    let parent = item.parentId;
    while (parent != null) {
      if (!byId.has(parent) || seen.has(parent)) fail('INVALID_GRAPH', 'Перенос создаёт цикл или отсутствует родитель');
      seen.add(parent); parent = byId.get(parent).parentId;
    }
  }
}
export function changesBetween(before, after) {
  const a = new Map(before.map(x => [x.id, x])), b = new Map(after.map(x => [x.id, x]));
  return [...new Set([...a.keys(), ...b.keys()])].flatMap(id => {
    const prev = a.get(id), next = b.get(id);
    if (stable(prev) === stable(next)) return [];
    if (!prev || !next) return [{ id, before: prev || null, after: next || null, fields: null }];
    const fields = [...new Set([...Object.keys(prev), ...Object.keys(next)])].filter(k => stable(prev[k]) !== stable(next[k]));
    return [{ id, before: clone(prev), after: clone(next), fields }];
  });
}
export class TaskGraph {
  constructor(state = {}, seed = {}) {
    this.items = clone(state.items || seed.snapshot?.items || []);
    if (!this.items.some(x => x.id === 'inbox')) this.items.unshift(clone(inbox));
    this.revision = state.revision || 0;
    validateItems(this.items);
  }
  read(query = {}) {
    if (query.id) return clone(this.items.find(x => x.id === query.id) || null);
    return { items: clone(this.items), revision: this.revision };
  }
  apply(commands, expectedRevision = this.revision) {
    if (expectedRevision !== this.revision) fail('CONFLICT', 'Документ изменился. Повторите команду в актуальном контексте.');
    const before = clone(this.items);
    let next = clone(before), label = '';
    for (const input of commands) {
      const { command, actId, payload = {} } = input;
      if (!graphCommands.has(command)) fail('UNSUPPORTED_COMMAND', 'Команда не поддерживается графом');
      const target = next.find(x => x.id === actId);
      if (!['addItem', 'importWorkflowyTree', 'reorderItems'].includes(command) && !target) fail('NOT_FOUND', 'Задача не найдена');
      if (actId === 'inbox' && ['editItem', 'deleteItem', 'setParent'].includes(command)) fail('PROTECTED', 'Входящие нельзя изменить этой командой');
      if ((['addItem', 'addChild'].includes(command) || (command === 'editItem' && 'line1' in payload)) && (typeof payload.line1 !== 'string' || !payload.line1.trim())) fail('INVALID_INPUT', 'Введите название');
      if (command === 'setStatus' && !statuses.includes(payload.status)) fail('INVALID_INPUT', 'Неизвестный статус');
      if (command === 'reorderItems') {
        if (!Array.isArray(payload.arranged) || payload.arranged.some(x => !next.some(n => n.id === x.id) || Object.keys(x).some(k => !['id','order','parentId'].includes(k)) || !Number.isFinite(x.order))) fail('INVALID_INPUT', 'Некорректное перемещение');
        if (payload.arranged.some(x => x.id === 'inbox' && x.parentId != null)) fail('PROTECTED', 'Входящие остаются в корне');
      }
      const interpreter = createInterpreter({ createItemId: () => crypto.randomUUID(), createLogId: () => 'draft' });
      const result = interpreter.execute({ snapshot: { items: next }, actionLog: [] }, input);
      if (!result.patch?.length) fail('INVALID_INPUT', 'Команда не изменила документ');
      for (const patch of result.patch) {
        if (patch.op !== 'replace' || patch.path !== '/snapshot/items') fail('INVALID_INPUT', 'Недопустимое изменение графа');
        next = clone(patch.value);
      }
      validateItems(next);
      label = result.logEntryDraft?.label || '';
    }
    const changes = changesBetween(before, next);
    if (changes.length) { this.items = next; this.revision += 1; }
    return { changes, revision: this.revision, label, target: next.find(x => !before.some(b => b.id === x.id))?.id || commands.at(-1)?.actId || null };
  }
  rollback(outcomes) {
    const before = clone(this.items);
    const map = new Map(before.map(x => [x.id, clone(x)]));
    for (const outcome of [...outcomes].reverse()) {
      for (const change of [...(outcome.changes || [])].reverse()) {
        const current = map.get(change.id) || null;
        if (!change.fields) {
          if (stable(current) !== stable(change.after)) fail('CONFLICT', 'Задача изменена другим действием; безопасный откат невозможен');
          if (change.before) map.set(change.id, clone(change.before)); else map.delete(change.id);
        } else {
          if (!current || change.fields.some(field => stable(current[field]) !== stable(change.after[field]))) fail('CONFLICT', 'Изменённые поля уже изменены другим действием');
          for (const field of change.fields) {
            if (field in change.before) current[field] = clone(change.before[field]); else delete current[field];
          }
        }
      }
    }
    const next = [...map.values()]; validateItems(next);
    const changes = changesBetween(before, next);
    if (changes.length) { this.items = next; this.revision += 1; }
    return { changes, revision: this.revision, label: 'Действие и корректировки отменены', target: outcomes[0]?.target || null };
  }
}
