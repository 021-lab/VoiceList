import { createActiveTaskTree, createTaskSubgraph, findTaskTree, formatTaskTitleTree } from '../../src/v2/domain/task-queries.js';

export class CompatibilityPort {
  constructor(runtime) { this.runtime = runtime; }
  getSnapshot() { return { rev: this.runtime.graph.revision, content: { snapshot: { items: this.runtime.graph.read().items },
    actionLog: this.runtime.journal.actions().map(a => ({ ...a, syncStatus: a.status === 'applied' ? 'synced' : 'failed', comments: this.runtime.journal.dialogue(a.id).filter(m => m.role === 'user').slice(1) })) } }; }
  getTaskById(id) { return this.runtime.graph.read({ id }); }
  getActiveTaskTree() { return createActiveTaskTree(this.runtime.graph.read().items); }
  getTaskSubgraph(id) { return createTaskSubgraph(this.runtime.graph.read().items, id); }
  getTaskTree(query) { return findTaskTree(this.runtime.graph.read().items, query); }
  getTaskTitleTreeText() { return formatTaskTitleTree(this.runtime.graph.read().items); }
  async applyCommand(command, { message = {} } = {}) {
    const clientKey = message.clientKey === 'mcp' ? 'mcp:' + crypto.randomUUID() : message.clientKey || 'server:' + crypto.randomUUID();
    const seq = message.clientKey === 'mcp' ? 1 : Number(message.seq) || 1;
    const prior = this.runtime.journal.entries.find(e => e.type === 'input' && e.input.key.clientKey === clientKey && e.input.key.seq === seq);
    const isAction = ['rollbackAction','commentLogEntry'].includes(command.command);
    const elementId = isAction ? 'action:' + command.actId : this.getTaskById(command.actId) ? 'task:' + command.actId : 'app';
    const result = await this.runtime.executeAndWait({ key: { clientKey, seq }, context: prior?.input.context || {
      elementId, view: 'list', revision: this.runtime.graph.revision, ...(isAction ? { actionId: command.actId } : {})
    }, command });
    return { seq, id: result.actions[0]?.id || null, status: result.error ? 'rejected' : 'applied',
      reason: result.error?.message || null, newTarget: result.actions.find(a => a.target)?.target || command.actId || null };
  }
  async undoLastAction({ clientKey, seq, source } = {}) {
    const prior = this.runtime.journal.actions().filter(a => a.canRollback).at(-1);
    const ack = await this.applyCommand({ command:'undo', actId:'list', actType:'list', payload:{}, source }, { message:{clientKey, seq} });
    return { status: ack.status, ack, undone: {logId:prior?.id,command:prior?.command?.command,id:prior?.target}, node: prior?.target ? this.getTaskById(prior.target) : null };
  }
}
