import { seedState } from '../../../list-data.js';
import { importWorkflowyTreeFromUrl } from '../../workflowy-import.js';
import { clone, inputSchema, fail, safeError, graphCommands, uiCommands } from './contracts.js';
import { TaskGraph } from './task-graph.js';
import { InteractionJournal } from './interaction-journal.js';
import { ActionService } from './action-service.js';
import { TaskAgent } from './task-agent.js';
import { AgentHarness, HARNESS_VERSIONS } from './agent-harness.js';
import { Presentation } from './presentation.js';

const emptyTechnical = () => ({ harness: {}, executor: {}, commandKeys: {}, undoneEntries: {}, cursor: 0, events: [] });

function normalizeState(initialState, seed) {
  const rawEntries = clone(initialState?.entries || []);
  const journal = new InteractionJournal(rawEntries);
  const technical = { ...emptyTechnical(), ...clone(initialState?.technical || {}) };
  for (const entry of journal.entries) {
    const legacyResults = rawEntries.filter(item => item.type === 'result' && item.requestId === entry.id);
    if (!technical.harness[entry.id]) technical.harness[entry.id] = { status: entry.kind === 'text' && !entry.versions ? 'pending' : entry.kind === 'text' ? 'complete' : 'bypassed' };
    if (!technical.executor[entry.id]) {
      const outcomes = legacyResults.filter(item => item.status === 'applied').map(item => ({
        key: `${entry.id}:0`, command: clone(item.command), changes: clone(item.changes || []), target: item.target,
        label: item.label, revision: item.revision, uiEffect: clone(item.uiEffect)
      }));
      const commands = entry.kind === 'ui' ? [entry.command] : entry.commands || [];
      technical.executor[entry.id] = { status: rawEntries.some(item => item.type === 'settled' && item.requestId === entry.id) ? 'complete' : commands.length ? 'pending' : 'complete', nextIndex: outcomes.length, outcomes };
      for (const outcome of outcomes) technical.commandKeys[outcome.key] = clone(outcome);
      for (const result of legacyResults) for (const id of result.undoneIds || []) technical.undoneEntries[id] = true;
    }
  }
  return { graph: clone(initialState?.graph || new TaskGraph({}, seed).read()), entries: journal.entries, technical };
}

/** Platform-independent coordinator. Model I/O always happens outside persistence transactions. */
export class DocumentRuntime {
  constructor({ initialState, seed = seedState, persist = async () => {}, resolveModel, fetchImpl = fetch, scheduler } = {}) {
    this.seed = seed; this.persist = persist; this.fetchImpl = fetchImpl;
    this.state = normalizeState(initialState, seed);
    this.harness = new AgentHarness({ agent: new TaskAgent({ resolveModel }), scheduler });
    this.presentation = new Presentation();
    this.tail = Promise.resolve(); this.processing = null;
  }
  get graph() { return new TaskGraph(this.state.graph, this.seed); }
  get journal() { return new InteractionJournal(this.state.entries, this.state.technical); }
  async transaction(fn) {
    const operation = this.tail.then(async () => {
      const graph = this.graph, technical = clone(this.state.technical), journal = new InteractionJournal(this.state.entries, technical);
      const result = fn({ graph, journal, technical });
      const next = { graph: graph.read(), entries: clone(journal.entries), technical };
      await this.persist(next);
      this.state = next;
      return result;
    });
    this.tail = operation.catch(() => {});
    return operation;
  }
  notification(technical, entry, extra = {}) {
    const event = { cursor: ++technical.cursor, entryId: entry.id, sessionId: entry.key.clientKey, ...clone(extra) };
    technical.events.push(event);
    if (technical.events.length > 500) technical.events.splice(0, technical.events.length - 500);
  }
  async submit(raw) {
    const parsed = inputSchema.safeParse(raw);
    if (!parsed.success) fail('INVALID_INPUT', 'Некорректная реплика или команда');
    let input = parsed.data;
    if (input.command?.command === 'commentLogEntry') input = { key: input.key, context: { ...input.context, actionId: input.command.actId }, text: String(input.command.payload?.text || '').trim() };
    if (!input.text && !input.command) fail('INVALID_INPUT', 'Пустая команда');
    return this.transaction(({ graph, journal, technical }) => {
      const prior = journal.request(input.key, input);
      if (prior) return this.receipt(prior, journal, technical);
      if (input.context.revision !== graph.revision) fail('CONFLICT', 'Документ изменился. Повторите команду.');
      const elementId = input.context.elementId;
      if (elementId.startsWith('task:') && !graph.read({ id: elementId.slice(5) })) fail('NOT_FOUND', 'Элемент задачи больше не существует');
      if (!elementId.startsWith('task:') && !elementId.startsWith('action:') && !['app', 'toolbar', 'list', ...['list','frontier','log','action','edit','add','settings','dialogues','search'].flatMap(view => ['menu:' + view, 'screen:' + view])].includes(elementId)) fail('NOT_FOUND', 'Неизвестный контекст интерфейса');
      let corrects = input.context.actionId;
      if (input.text !== undefined && corrects && journal.get(corrects)) corrects = journal.chain(corrects).at(-1)?.id || corrects;
      if (input.command?.command === 'rollbackAction') corrects ||= input.command.actId;
      if (input.command?.command === 'undo') corrects ||= [...journal.entries].reverse().find(item =>
        technical.executor[item.id]?.outcomes?.some(outcome => outcome.changes?.length) && !technical.undoneEntries[item.id]
      )?.id;
      if (corrects && !journal.get(corrects)) fail('NOT_FOUND', 'Корректируемое действие не найдено');
      if (input.context.actionId && !corrects) fail('NOT_FOUND', 'Контекст корректировки не найден');
      if (input.command && !graphCommands.has(input.command.command) && !uiCommands[input.command.command] && !['rollbackAction','undo','importWorkflowy'].includes(input.command.command)) fail('UNSUPPORTED_COMMAND', 'Команда не поддерживается');
      const entry = journal.appendInteraction(input, { corrects });
      technical.harness[entry.id] = { status: entry.kind === 'text' ? 'pending' : 'bypassed' };
      technical.executor[entry.id] = { status: entry.kind === 'text' ? 'waiting' : 'pending', nextIndex: 0, outcomes: [] };
      return this.receipt(entry, journal, technical);
    });
  }
  receipt(entry, journal = this.journal, technical = this.state.technical) {
    const harness = technical.harness[entry.id] || {};
    const executor = technical.executor[entry.id] || {};
    const terminal = harness.status === 'failed' || executor.status === 'failed' || executor.status === 'complete';
    const root = entry.kind === 'text' && !entry.corrects && journal.processed(entry) ? journal.action(entry) : null;
    const target = [...(executor.outcomes || [])].reverse().find(item => item.target)?.target || null;
    return { requestId: entry.id, cursor: technical.cursor, status: terminal ? 'completed' : 'accepted', actions: root ? [root] : [], target, error: executor.error || harness.error || null };
  }
  getDocument(context = {}) {
    return { ...this.presentation.compose(this.graph.read(), this.journal, context), cursor: this.state.technical.cursor };
  }
  follow(cursor = 0, clientKey = '') {
    const after = Math.max(0, Number(cursor) || 0), technical = this.state.technical, journal = this.journal;
    const events = technical.events.filter(event => event.cursor > after);
    const actions = events.flatMap(event => event.publicActionId ? [journal.actions().find(action => action.id === event.publicActionId)].filter(Boolean) : []);
    const uiEffects = events.filter(event => event.uiEffect && event.sessionId === clientKey).map(event => ({ ...event.uiEffect, actionId: event.entryId }));
    return { events: actions, actions, uiEffects, nextCursor: technical.cursor, revision: this.graph.revision };
  }
  pending() {
    return this.state.entries.filter(entry => ['pending', 'waiting'].includes(this.state.technical.harness[entry.id]?.status) || ['pending', 'waiting'].includes(this.state.technical.executor[entry.id]?.status));
  }
  async processPending() {
    if (this.processing) return this.processing;
    this.processing = this.drain().finally(() => { this.processing = null; });
    return this.processing;
  }
  async processHarness(entry) {
    let prepared = entry;
    if (!entry.modelContext) prepared = await this.transaction(({ graph, journal }) => {
      const current = journal.get(entry.id);
      if (!current.modelContext) journal.enrich(entry.id, { modelContext: this.harness.buildContext({ entry: current, graph: graph.read(), journal }), versions: HARNESS_VERSIONS });
      return journal.get(entry.id);
    });
    let raw;
    try {
      raw = await this.harness.invoke(prepared.modelContext);
      const parsed = this.harness.parse(raw, prepared.modelContext);
      await this.transaction(({ journal, technical }) => {
        if (technical.harness[entry.id]?.status !== 'pending') return;
        journal.enrich(entry.id, { rawModelResponse: clone(raw), answer: parsed.answer, commands: parsed.commands });
        technical.harness[entry.id] = { status: 'complete' };
        technical.executor[entry.id].status = parsed.commands.length ? 'pending' : 'complete';
        if (!parsed.commands.length) this.notification(technical, journal.get(entry.id), { publicActionId: journal.get(entry.id).corrects ? null : entry.id });
      });
    } catch (error) {
      await this.transaction(({ journal, technical }) => {
        if (raw !== undefined && journal.get(entry.id).rawModelResponse === undefined) journal.enrich(entry.id, { rawModelResponse: clone(raw), answer: '', commands: [] });
        technical.harness[entry.id] = { status: 'failed', error: safeError(error) };
        technical.executor[entry.id] = { ...technical.executor[entry.id], status: 'failed', error: safeError(error) };
        this.notification(technical, journal.get(entry.id), { publicActionId: journal.get(entry.id).corrects ? null : entry.id });
      });
    }
  }
  async processExecutor(entry) {
    const current = this.journal.get(entry.id);
    const ledger = this.state.technical.executor[entry.id];
    const commands = current.kind === 'ui' ? [current.command] : current.commands || [];
    let command = clone(commands[ledger.nextIndex]);
    if (command?.command === 'importWorkflowy') {
      const tree = await importWorkflowyTreeFromUrl(command.payload?.url || '', { fetchImpl: this.fetchImpl });
      command = { ...command, command: 'importWorkflowyTree', payload: { tree } };
    }
    try {
      await this.transaction(({ graph, journal, technical }) => {
        const state = technical.executor[entry.id];
        if (state.status !== 'pending' || state.nextIndex !== ledger.nextIndex) return;
        const latest = journal.get(entry.id);
        const service = new ActionService({ graph, journal, technical });
        const outcome = service.execute(latest, command, state.nextIndex);
        state.outcomes.push(clone(outcome)); state.nextIndex += 1;
        if (outcome.undoneEntryIds) for (const id of outcome.undoneEntryIds) technical.undoneEntries[id] = true;
        if (state.nextIndex >= commands.length) {
          state.status = 'complete';
          this.notification(technical, latest, { publicActionId: latest.kind === 'text' && !latest.corrects ? latest.id : null, uiEffect: outcome.uiEffect });
        }
      });
    } catch (error) {
      await this.transaction(({ journal, technical }) => {
        const state = technical.executor[entry.id];
        state.status = 'failed'; state.error = safeError(error);
        this.notification(technical, journal.get(entry.id), { publicActionId: journal.get(entry.id).kind === 'text' && !journal.get(entry.id).corrects ? entry.id : null });
      });
    }
  }
  async drain() {
    for (let count = 0; count < 100; count++) {
      const harnessEntry = this.state.entries.find(entry => this.state.technical.harness[entry.id]?.status === 'pending');
      if (harnessEntry) { await this.processHarness(harnessEntry); continue; }
      const executorEntry = this.state.entries.find(entry => this.state.technical.executor[entry.id]?.status === 'pending');
      if (executorEntry) { await this.processExecutor(executorEntry); continue; }
      return;
    }
  }
  async executeAndWait(raw) {
    const receipt = await this.submit(raw);
    await this.processPending();
    return this.receipt(this.journal.get(receipt.requestId));
  }
  exportState() { return clone(this.state); }
}
