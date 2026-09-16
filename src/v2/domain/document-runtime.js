import { seedState } from '../../../list-data.js';
import { importWorkflowyTreeFromUrl } from '../../workflowy-import.js';
import { clone, inputSchema, fail, safeError, graphCommands, uiCommands } from './contracts.js';
import { TaskGraph } from './task-graph.js';
import { InteractionJournal } from './interaction-journal.js';
import { ActionService } from './action-service.js';
import { TaskAgent } from './task-agent.js';
import { AgentHarness } from './agent-harness.js';
import { Presentation } from './presentation.js';

/** Platform-independent coordinator. The DO adapter owns SDK lifecycle and durable I/O. */
export class DocumentRuntime {
  constructor({ initialState, seed = seedState, persist = async () => {}, resolveModel, fetchImpl = fetch, scheduler } = {}) {
    this.seed = seed; this.persist = persist; this.fetchImpl = fetchImpl;
    this.state = clone(initialState || { graph: new TaskGraph({}, seed).read(), entries: [] });
    this.harness = new AgentHarness({ agent: new TaskAgent({ resolveModel }), scheduler });
    this.presentation = new Presentation();
    this.tail = Promise.resolve(); this.processing = null;
  }
  get graph() { return new TaskGraph(this.state.graph, this.seed); }
  get journal() { return new InteractionJournal(this.state.entries); }
  async transaction(fn) {
    const operation = this.tail.then(async () => {
      const graph = this.graph, journal = this.journal;
      const result = fn({ graph, journal });
      const next = { graph: graph.read(), entries: clone(journal.entries) };
      await this.persist(next);
      this.state = next;
      return result;
    });
    this.tail = operation.catch(() => {});
    return operation;
  }
  async submit(raw) {
    const parsed = inputSchema.safeParse(raw);
    if (!parsed.success) fail('INVALID_INPUT', 'Некорректная реплика или команда');
    const input = parsed.data;
    return this.transaction(({ graph, journal }) => {
      const prior = journal.request(input.key, input);
      if (prior) return this.receipt(prior, journal);
      const elementId = input.context.elementId;
      if (elementId.startsWith('task:') && !graph.read({ id: elementId.slice(5) })) fail('NOT_FOUND', 'Элемент задачи больше не существует');
      if (!elementId.startsWith('task:') && !elementId.startsWith('action:') && !['app', 'toolbar', 'list', ...['list','frontier','log','action','edit','add','settings','dialogues','search'].flatMap(v => ['menu:'+v,'screen:'+v])].includes(elementId)) fail('NOT_FOUND', 'Неизвестный контекст интерфейса');
      if (elementId.startsWith('action:') && !journal.actions().some(x => x.id === elementId.slice(7))) fail('NOT_FOUND', 'Действие не найдено');
      if (input.context.actionId && !journal.actions().some(x => x.id === input.context.actionId)) fail('NOT_FOUND', 'Контекст корректировки не найден');
      if (input.command && !graphCommands.has(input.command.command) && !uiCommands[input.command.command] && !['rollbackAction','undo','commentLogEntry','importWorkflowy','logFallbackUtterance'].includes(input.command.command)) fail('UNSUPPORTED_COMMAND', 'Команда не поддерживается');
      const entry = journal.append({ type: 'input', input, sessionId: input.key.clientKey });
      return this.receipt(entry, journal);
    });
  }
  receipt(entry, journal = this.journal) {
    const settled = journal.entries.find(e => e.type === 'settled' && e.requestId === entry.id);
    const results = journal.entries.filter(e => e.type === 'result' && e.requestId === entry.id);
    return { requestId: entry.id, cursor: journal.cursor, status: settled ? 'completed' : 'accepted',
      actions: journal.actions().filter(a => results.some(r => r.actionId === a.id)),
      error: results.find(r => r.error)?.error || null };
  }
  getDocument(context = {}) { return this.presentation.compose(this.graph.read(), this.journal, context); }
  follow(cursor = 0, clientKey = '') {
    const after = Math.max(0, Number(cursor) || 0);
    const journal = this.journal;
    const entries = journal.read({ after, limit: 100 });
    const nextCursor = entries.at(-1)?.cursor || journal.cursor;
    const actions = journal.actions().filter(a => a.cursor > after && a.cursor <= nextCursor);
    const uiEffects = entries.filter(e => e.type === 'result' && e.uiEffect && e.sessionId === clientKey).map(e => ({ ...e.uiEffect, actionId: e.actionId }));
    // Only public outcomes cross this boundary, never model prompts or private execution metadata.
    return { events: actions, actions, uiEffects, nextCursor, revision: this.graph.revision };
  }
  async processPending() {
    if (this.processing) return this.processing;
    this.processing = this.drain().finally(() => { this.processing = null; });
    return this.processing;
  }
  async drain() {
    for (let count = 0; count < 20; count++) {
      const request = this.journal.pending()[0];
      if (!request) return;
      let commands = this.journal.entries.filter(e => e.type === 'command' && e.requestId === request.id);
      if (!commands.length) {
        const graph = this.graph.read();
        let input = clone(request.input), decision;
        try {
          if (input.command?.command === 'commentLogEntry') {
            input.context.actionId = input.command.actId;
            input.text = input.command.payload?.text || '';
            delete input.command;
          }
          decision = await this.harness.handle({ input, graph, journal: this.journal });
          if (!decision || !Array.isArray(decision.commands || [])) fail('INVALID_DECISION', 'Некорректный ответ агента');
          if ((decision.commands || []).length > 10) fail('INVALID_DECISION', 'Слишком много команд');
          for (const command of decision.commands || []) {
            if (command.command === 'importWorkflowy') {
              const tree = await importWorkflowyTreeFromUrl(command.payload?.url || '', { fetchImpl: this.fetchImpl });
              command.command = 'importWorkflowyTree'; command.payload = { tree };
            }
          }
        } catch (error) { decision = { reply: safeError(error).message, error: safeError(error) }; }
        await this.transaction(({ graph: current, journal }) => {
          if (journal.entries.some(e => e.requestId === request.id && ['command', 'settled'].includes(e.type))) return;
          const original = input.context.actionId ? journal.actions().find(a => a.id === input.context.actionId) : null;
          const rootActionId = original?.rootActionId || original?.id || ((decision.commands || []).length ? 'e' + (journal.cursor + 1) : request.id);
          if (!(decision.commands || []).length) {
            journal.append({ type: 'result', requestId: request.id, actionId: request.id, rootActionId: rootActionId || request.id, sessionId: request.sessionId,
              status: decision.error ? 'failed' : 'needs-input', label: decision.reply || 'Уточните команду', reply: decision.reply || '', error: decision.error || null, changes: [], revision: current.revision });
            journal.append({ type: 'settled', requestId: request.id });
          } else {
            for (const command of decision.commands) journal.append({ type: 'command', requestId: request.id, rootActionId, sessionId: request.sessionId,
              command: clone(command), reply: decision.reply || '', expectedRevision: graph.revision });
          }
        });
        commands = this.journal.entries.filter(e => e.type === 'command' && e.requestId === request.id);
      }
      if (commands.length) {
        await this.transaction(({ graph, journal }) => {
          const service = new ActionService({ graph, journal });
          // A decision's commands form one ordered execution batch. Any failure stops the remainder.
          let failed = false, expected = commands[0].expectedRevision;
          for (const command of commands) {
            const done = journal.entries.find(e => e.type === 'result' && e.actionId === command.id);
            if (done) { failed ||= done.status !== 'applied'; expected = graph.revision; continue; }
            const outcome = service.execute(command.id, failed ? -1 : expected);
            failed ||= outcome.status !== 'applied'; expected = graph.revision;
          }
          if (!journal.entries.some(e => e.type === 'settled' && e.requestId === request.id)) journal.append({ type: 'settled', requestId: request.id });
        });
      }
    }
  }
  async executeAndWait(raw) {
    const receipt = await this.submit(raw);
    await this.processPending();
    return this.receipt(this.journal.get(receipt.requestId));
  }
  exportState() { return clone(this.state); }
}
