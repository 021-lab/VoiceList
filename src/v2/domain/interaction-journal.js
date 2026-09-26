import { clone, stable, fail } from './contracts.js';

const commandLabel = command => command?.command || '';

function normalizeLegacy(entries) {
  if (!entries.some(entry => entry.type !== 'interaction')) return entries.map(clone);
  const modern = entries.filter(entry => entry.type === 'interaction').map(clone);
  const modernIds = new Set(modern.map(entry => entry.id));
  const legacyResults = entries.filter(entry => entry.type === 'result');
  const actionToRequest = new Map(legacyResults.map(entry => [entry.actionId, entry.requestId]));
  for (const entry of legacyResults) if (entry.rootActionId && !actionToRequest.has(entry.rootActionId)) actionToRequest.set(entry.rootActionId, entry.requestId);
  const migrated = entries.filter(entry => entry.type === 'input' && !modernIds.has(entry.id)).map(input => {
    const commands = entries.filter(entry => entry.type === 'command' && entry.requestId === input.id);
    const results = entries.filter(entry => entry.type === 'result' && entry.requestId === input.id);
    const decision = { answer: results.map(entry => entry.reply).find(Boolean) || '', commands: commands.map(entry => clone(entry.command)) };
    const corrects = actionToRequest.get(input.input.context?.actionId) || input.input.context?.actionId;
    return {
      id: input.id, cursor: input.cursor, at: input.at, type: 'interaction', key: clone(input.input.key),
      context: clone(input.input.context), kind: input.input.text !== undefined ? 'text' : 'ui',
      ...(input.input.text !== undefined ? { text: input.input.text } : { command: clone(input.input.command) }),
      ...(corrects ? { corrects } : {}),
      ...(input.input.text !== undefined ? {
        modelContext: { legacy: true, text: input.input.text }, rawModelResponse: JSON.stringify(decision),
        answer: decision.answer, commands: decision.commands,
        versions: { contextBuilder: 'legacy', prompt: 'legacy', parser: 'legacy' }
      } : {})
    };
  });
  return [...modern, ...migrated].sort((a, b) => a.cursor - b.cursor);
}

/** Append-only interaction bus. An entry may only gain processing fields, and it gains them
 *  as a replacement rather than in place, so a journal can share entries with the state it
 *  was built from without being able to change it. */
export class InteractionJournal {
  constructor(entries = [], technical = {}, { own = false } = {}) {
    this.entries = own ? entries : normalizeLegacy(clone(entries));
    this.technical = technical;
  }

  /** A journal over entries the caller already owns and has normalised.
   *
   *  Copying every entry on construction was the second most expensive thing a command did:
   *  a journal is built for each transaction and for each read of the document, and each
   *  build copied the whole journal. */
  static over(entries, technical = {}) { return new InteractionJournal(entries, technical, { own: true }); }
  appendInteraction(input, { corrects } = {}) {
    const cursor = Math.max(0, ...this.entries.map(entry => Number(entry.cursor) || 0)) + 1;
    const entry = {
      id: 'e' + cursor, cursor, at: new Date().toISOString(), type: 'interaction',
      key: clone(input.key), context: clone(input.context), kind: input.text !== undefined ? 'text' : 'ui',
      ...(input.text !== undefined ? { text: input.text } : { command: clone(input.command) }),
      ...(corrects ? { corrects } : {})
    };
    this.entries.push(entry);
    return clone(entry);
  }
  enrich(id, patch) {
    const index = this.entries.findIndex(item => item.id === id);
    if (index < 0) fail('NOT_FOUND', 'Запись журнала не найдена');
    const entry = this.entries[index];
    const enriched = { ...entry };
    for (const [key, value] of Object.entries(clone(patch))) {
      if (entry[key] !== undefined && stable(entry[key]) !== stable(value)) fail('CONFLICT', 'Запись журнала уже обработана иначе');
      enriched[key] = value;
    }
    // Replaced, not edited: the entry object may be shared with the state this journal was
    // built from, and that state must stay as it was until the transaction commits.
    this.entries[index] = enriched;
    return clone(enriched);
  }
  read({ after = 0, limit = 100 } = {}) {
    return clone(this.entries.filter(entry => entry.cursor > after).slice(0, Math.min(limit, 500)));
  }
  get(id) { return clone(this.entries.find(entry => entry.id === id) || null); }
  get cursor() { return Math.max(0, ...this.entries.map(entry => Number(entry.cursor) || 0)); }
  request(key, input) {
    // The key is a client id and a sequence number, so it is compared as such: serialising
    // every entry's key to find one made submitting a command cost the whole journal.
    const prior = this.entries.find(entry => entry.key?.seq === key.seq && entry.key?.clientKey === key.clientKey);
    if (!prior) return null;
    const saved = { key: prior.key, context: prior.context, ...(prior.kind === 'text' ? { text: prior.text } : { command: prior.command }) };
    if (stable(saved) !== stable(input)) fail('REQUEST_KEY_REUSED', 'Идентификатор запроса уже использован для другого ввода');
    return clone(prior);
  }
  rootFor(id) {
    let entry = this.entries.find(item => item.id === id) || null;
    const seen = new Set();
    while (entry?.corrects && !seen.has(entry.id)) { seen.add(entry.id); entry = this.entries.find(item => item.id === entry.corrects) || entry; }
    return clone(entry);
  }
  chain(id) {
    const root = this.rootFor(id);
    if (!root) fail('NOT_FOUND', 'Действие не найдено');
    const connected = new Set([root.id]);
    for (const entry of this.entries) if (entry.corrects && connected.has(entry.corrects)) connected.add(entry.id);
    return clone(this.entries.filter(entry => connected.has(entry.id)).sort((a, b) => a.cursor - b.cursor));
  }
  processed(entry) {
    return this.technical.harness?.[entry.id]?.status === 'complete' || Boolean(entry.versions && entry.rawModelResponse !== undefined);
  }
  action(entry, chains) {
    const execution = this.technical.executor?.[entry.id] || {};
    const outcomes = execution.outcomes || [];
    const target = [...outcomes].reverse().find(item => item.target)?.target || entry.commands?.find(command => command.actId)?.actId || null;
    const error = execution.error || this.technical.harness?.[entry.id]?.error || null;
    const rolledBack = Boolean(this.technical.undoneEntries?.[entry.id]);
    const chain = chains ? (chains.get(entry.id) || [entry]) : this.chain(entry.id);
    const hasChanges = chain.some(item => this.technical.executor?.[item.id]?.outcomes?.some(outcome => outcome.changes?.length));
    const label = entry.answer || entry.commands?.map(commandLabel).filter(Boolean).join(', ') || entry.text;
    return {
      id: entry.id, actionId: entry.id, rootActionId: entry.id, requestId: entry.id,
      label, text: label, status: error ? 'failed' : entry.commands?.length ? (execution.status === 'complete' ? 'applied' : 'pending') : 'needs-input',
      createdAt: entry.at, transcript: entry.text, context: entry.context, command: entry.commands?.[0] || null,
      target, error, rolledBack, canRollback: hasChanges && !rolledBack, sessionId: entry.key.clientKey, cursor: entry.cursor
    };
  }
  actions(chains) {
    return this.entries.filter(entry => entry.kind === 'text' && !entry.corrects && this.processed(entry)).map(entry => this.action(entry, chains));
  }
  pending() {
    return clone(this.entries.filter(entry => ['pending', 'waiting'].includes(this.technical.harness?.[entry.id]?.status) || ['pending', 'waiting'].includes(this.technical.executor?.[entry.id]?.status)));
  }
  messagesOf(entry) {
    if (entry.kind === 'ui') return [{ role: 'user', text: commandLabel(entry.command), id: entry.id, kind: 'ui' }];
    return [
      { role: 'user', text: entry.text, id: entry.id, kind: 'text' },
      ...(this.processed(entry) ? [{ role: 'assistant', text: entry.answer || entry.commands?.map(commandLabel).join(', ') || '', id: entry.id + ':answer' }] : [])
    ];
  }

  /** Every correction chain in one pass, keyed by the root it belongs to.
   *
   *  Asking per action — chain() inside action(), then dialogue() again — re-walked and
   *  re-cloned the whole journal each time. A snapshot of a few hundred entries cost around
   *  190 ms, and a broadcast spent almost all of its time here, on every tool call. */
  chains() {
    const byId = new Map(this.entries.map(entry => [entry.id, entry]));
    const rootOf = new Map();
    const resolve = (entry) => {
      const path = [];
      let current = entry;
      const seen = new Set();
      while (current && !rootOf.has(current.id) && current.corrects && !seen.has(current.id)) {
        seen.add(current.id);
        path.push(current);
        current = byId.get(current.corrects) || null;
      }
      const root = current ? (rootOf.get(current.id) || current) : entry;
      for (const item of path) rootOf.set(item.id, root);
      if (current) rootOf.set(current.id, root);
      return root;
    };
    const groups = new Map();
    for (const entry of this.entries) {
      const root = resolve(entry);
      const list = groups.get(root.id);
      if (list) list.push(entry); else groups.set(root.id, [entry]);
    }
    for (const list of groups.values()) list.sort((a, b) => a.cursor - b.cursor);
    return groups;
  }

  dialogues(chains = this.chains()) {
    const result = new Map();
    for (const [rootId, entries] of chains) result.set(rootId, entries.flatMap(entry => this.messagesOf(entry)));
    return result;
  }

  dialogue(actionId) {
    return this.chain(actionId).flatMap(entry => {
      return this.messagesOf(entry);
    });
  }
}
