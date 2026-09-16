import { clone, fail, uiCommands, safeError } from './contracts.js';

export class ActionService {
  constructor({ graph, journal }) { this.graph = graph; this.journal = journal; }
  execute(commandEntryId, expectedRevision) {
    const entry = this.journal.get(commandEntryId);
    if (!entry || entry.type !== 'command') fail('NOT_FOUND', 'Команда журнала не найдена');
    const prior = this.journal.entries.find(e => e.type === 'result' && e.actionId === commandEntryId);
    if (prior) return clone(prior);
    const input = entry.command;
    let result;
    try {
      if ((expectedRevision ?? entry.expectedRevision) !== this.graph.revision) fail('CONFLICT', 'Документ изменился после подготовки команды');
      if (input.command === 'rollbackAction' || input.command === 'undo') {
        const actionId = input.command === 'undo'
          ? this.journal.actions().filter(a => a.canRollback).at(-1)?.id : input.actId;
        if (!actionId) fail('NOT_FOUND', 'Нечего откатывать');
        const chain = this.journal.chain(actionId);
        if (!chain.length) fail('ALREADY_ROLLED_BACK', 'Действие уже отменено или не меняет задачи');
        result = { ...this.graph.rollback(chain), undoneIds: chain.map(x => x.actionId) };
      } else if (uiCommands[input.command]) {
        result = { label: 'Открыт экран', revision: this.graph.revision, changes: [], uiEffect: {
          view: uiCommands[input.command], taskId: input.actId === 'list' ? null : input.actId,
          query: input.payload?.query || '', mode: input.command === 'showAddModal' ? 'add' : 'edit',
          parentId: input.payload?.parentId || null
        } };
      } else {
        result = this.graph.apply([input], entry.expectedRevision);
      }
      return this.journal.append({
        type: 'result', actionId: entry.id, requestId: entry.requestId, rootActionId: entry.rootActionId || entry.id,
        sessionId: entry.sessionId, command: input, status: 'applied', ...result,
        label: result.label || 'Готово', reply: entry.reply || '', silent: input.command === 'toggleCollapse'
      });
    } catch (error) {
      return this.journal.append({
        type: 'result', actionId: entry.id, requestId: entry.requestId, rootActionId: entry.rootActionId || entry.id,
        sessionId: entry.sessionId, command: input, status: 'failed', label: safeError(error).message,
        error: safeError(error), changes: [], revision: this.graph.revision
      });
    }
  }
}
