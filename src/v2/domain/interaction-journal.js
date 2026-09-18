import { clone, stable, fail } from './contracts.js';

export class InteractionJournal {
  constructor(entries = []) { this.entries = clone(entries); }
  append(message) {
    const entry = { ...clone(message), id: 'e' + (this.entries.length + 1), cursor: this.entries.length + 1, at: new Date().toISOString() };
    this.entries.push(entry);
    return clone(entry);
  }
  read({ type, after = 0, limit = 100, requestId, actionId } = {}) {
    return clone(this.entries.filter(e => e.cursor > after && (!type || e.type === type) && (!requestId || e.requestId === requestId) && (!actionId || e.actionId === actionId || e.rootActionId === actionId)).slice(0, Math.min(limit, 500)));
  }
  get(id) { return clone(this.entries.find(e => e.id === id) || null); }
  get cursor() { return this.entries.length; }
  request(key, input) {
    const prior = this.entries.find(e => e.type === 'input' && stable(e.input.key) === stable(key));
    if (prior && stable(prior.input) !== stable(input)) fail('REQUEST_KEY_REUSED', 'Идентификатор запроса уже использован для другого ввода');
    return clone(prior || null);
  }
  pending() {
    return this.entries.filter(e => e.type === 'input' && !this.entries.some(x => x.type === 'settled' && x.requestId === e.id)).map(clone);
  }
  actions() {
    const results = this.entries.filter(e => e.type === 'result' && !e.silent && !['toggleCollapse', 'rollbackAction', 'undo'].includes(e.command?.command));
    // Silent rollback outcomes still change the public state of their source
    // actions even though they never become actions of their own.
    const undone = new Set(this.entries.filter(e => e.type === 'result' && e.status === 'applied').flatMap(e => e.undoneIds || []));
    return results.map(result => {
      const request = this.get(result.requestId);
      return {
        id: result.actionId, actionId: result.actionId, rootActionId: result.rootActionId, requestId: result.requestId,
        label: result.label, text: result.reply || result.label, status: result.status, createdAt: result.at,
        transcript: request?.input?.text || request?.input?.command?.transcript || null,
        context: request?.input?.context || null, command: result.command || null,
        target: result.target, error: result.error || null, rolledBack: undone.has(result.actionId),
        canRollback: result.status === 'applied' && Boolean(result.changes?.length) && !undone.has(result.actionId),
        sessionId: result.sessionId, cursor: result.cursor
      };
    });
  }
  chain(actionId) {
    const result = this.entries.find(e => e.type === 'result' && e.actionId === actionId);
    if (!result) fail('NOT_FOUND', 'Действие не найдено');
    const root = result.rootActionId || actionId;
    const undone = new Set(this.entries.flatMap(e => e.status === 'applied' ? e.undoneIds || [] : []));
    return this.entries.filter(e => e.type === 'result' && e.status === 'applied' && (e.rootActionId === root || e.actionId === root) && e.changes?.length && !undone.has(e.actionId)).map(clone);
  }
  dialogue(actionId) {
    const action = this.actions().find(a => a.id === actionId);
    if (!action) return [];
    const source = this.entries.find(e => e.type === 'result' && e.actionId === actionId);
    const root = action.rootActionId || actionId;
    const related = this.entries.filter(e => e.type === 'result' && (e.rootActionId === root || e.actionId === root));
    const ids = new Set(related.map(e => e.requestId));
    return this.entries.flatMap(e => {
      if (e.type === 'input' && e.id !== source?.requestId && (ids.has(e.id) || e.input.context.actionId === root || e.input.context.actionId === actionId)) return [{ role: 'user', text: e.input.text || e.input.command?.transcript || e.input.command?.command || '', id: e.id }];
      if (e.type === 'result' && (e.rootActionId === root || e.actionId === root)) return [{ role: 'assistant', text: e.reply || e.label, id: e.id }];
      return [];
    });
  }
}
