import { clone, fail, uiCommands } from './contracts.js';

/** Executes journal commands. All outcomes belong to the private executor ledger. */
export class ActionService {
  constructor({ graph, journal, technical }) { Object.assign(this, { graph, journal, technical }); }
  execute(entry, command, index) {
    const key = `${entry.id}:${index}`;
    // The entry's own ledger is the record. A second, document-wide copy of every outcome
    // used to be kept under the same key; it held nothing the ledger does not and made the
    // ledgers twice the size, which is paid on every transaction.
    const prior = this.technical.executor?.[entry.id]?.outcomes?.find(item => item.key === key);
    if (prior) return clone(prior);
    let result;
    if (command.command === 'rollbackAction' || command.command === 'undo') {
      const root = this.journal.rootFor(entry.corrects || command.actId);
      if (!root) fail('NOT_FOUND', 'Нечего откатывать');
      if (this.technical.undoneEntries?.[root.id]) fail('ALREADY_ROLLED_BACK', 'Действие уже отменено');
      const chain = this.journal.chain(root.id).filter(item => item.cursor < entry.cursor);
      const outcomes = chain.flatMap(item => this.technical.executor?.[item.id]?.outcomes || []).filter(item => item.changes?.length);
      if (!outcomes.length) fail('ALREADY_ROLLED_BACK', 'Действие не меняет задачи или уже отменено');
      result = { ...this.graph.rollback(outcomes), undoneEntryIds: chain.map(item => item.id), rootId: root.id };
    } else if (uiCommands[command.command]) {
      result = { label: 'Открыт экран', revision: this.graph.revision, changes: [], target: command.actId || null, uiEffect: {
        view: uiCommands[command.command], taskId: command.actId === 'list' ? null : command.actId,
        query: command.payload?.query || '', mode: command.command === 'showAddModal' ? 'add' : 'edit',
        parentId: command.payload?.parentId || null
      } };
    } else {
      result = this.graph.apply([command], index === 0 ? entry.context.revision : this.graph.revision);
    }
    return { key, command: clone(command), ...clone(result) };
  }
}
