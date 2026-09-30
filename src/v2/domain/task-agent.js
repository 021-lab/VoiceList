import { parseCommand, toCommand } from '../../command-resolver.js';
import { findCandidates } from '../../resolver.js';
import { adaptSnapshot } from '../../snapshot-adapter.js';
import { clone, fail } from './contracts.js';

/** Everything needed to correct another model's work, gathered from the journal.
 *
 *  A correction is not a new request. What it needs is the case file: the instructions the
 *  first model worked under, the context it was handed, the answer it gave, what the system
 *  did with that answer, and the person's objection. All of it is already in the journal —
 *  the entry being corrected carries its own model context and raw response — and none of it
 *  was reaching the model asked to fix it, which is why «так неправильно» produced a question
 *  instead of a fix.
 *
 *  A voice action has no context of ours to show: the model that made it heard the audio and
 *  its reasoning never passed through here. Then the case file is the command it issued and
 *  the words it reported hearing, which is what there is. */
/** The element the correction was spoken at, resolved into the thing itself. */
function elementOf(elementId, graph, journal) {
  if (elementId.startsWith('task:')) {
    const task = graph.items.find(item => item.id === elementId.slice(5));
    return task
      ? { kind: 'task', taskId: task.id, title: task.title, status: task.status, parentId: task.parentId }
      : { kind: 'task', taskId: elementId.slice(5), missing: true };
  }
  if (elementId.startsWith('action:')) {
    const action = journal.get(elementId.slice(7));
    return action
      ? { kind: 'action', actionId: elementId.slice(7), request: action.text || null }
      : { kind: 'action', actionId: elementId.slice(7), missing: true };
  }
  return { kind: 'screen', screenId: elementId };
}

function correctionOf({ entry, graph, journal }) {
  if (!entry.corrects) return null;
  const corrected = journal.get(entry.corrects);
  if (!corrected) return null;
  // The outcome of the entry being corrected, read from its own ledger. Taken over the whole
  // chain it would include this very correction, which has not run yet, and the case file
  // would say «pending» about an action that finished long ago.
  const ledger = journal.technical?.executor?.[corrected.id] || {};
  const outcomes = ledger.outcomes || [];
  const status = ledger.error ? 'failed' : ledger.status === 'complete' ? 'applied' : (ledger.status || 'unknown');
  const error = ledger.error || null;
  const target = [...outcomes].reverse().find(item => item.target)?.target || null;
  // The rows as they changed, not a list of field names: a correction is judged against what
  // the list looks like now, and «изменены title, status» says nothing about what it became.
  const shown = (task) => task && Object.fromEntries(
    ['title', 'status', 'parentId', 'deadline', 'tags'].filter(field => task[field] !== undefined && task[field] !== '')
      .map(field => [field, task[field]])
  );
  const changed = outcomes.flatMap(item => item.changes || []).map(change => change.fields
    ? { taskId: change.id, changedFields: change.fields, valuesBefore: shown(change.before), valuesAfter: shown(change.after) }
    : { taskId: change.id, operation: change.before ? 'removed' : 'created', task: shown(change.after || change.before) });
  const root = journal.rootFor(corrected.id) || corrected;
  return {
    userText: entry.text,
    // What the person was touching when they objected: a task or an action. «Здесь» and «это»
    // mean that element and nothing else.
    element: elementOf(entry.context.elementId, graph, journal),
    // The id rollbackAction needs, which is the root of the chain rather than the entry.
    actionId: root.id,
    original: {
      id: corrected.id,
      kind: corrected.kind,
      text: corrected.text || null,
      heard: corrected.command?.transcript || null,
      source: corrected.command?.source || (corrected.kind === 'text' ? 'agent' : 'ui'),
      modelContext: clone(corrected.modelContext || null),
      rawModelResponse: corrected.rawModelResponse ?? null,
      answer: corrected.answer || '',
      commands: clone(corrected.commands || (corrected.command ? [corrected.command] : []))
    },
    outcome: { status, error, target, changed }
  };
}

export class TaskAgent {
  constructor({ resolveModel = null } = {}) { this.resolveModel = resolveModel; }
  buildContext({ entry, graph, journal }) {
    const root = entry.corrects ? journal.rootFor(entry.corrects) : null;
    const history = root ? journal.chain(root.id).filter(item => item.id !== entry.id).map(item => ({
      id: item.id, corrects: item.corrects || null, kind: item.kind, text: item.text || null,
      answer: item.answer || '', commands: clone(item.commands || (item.command ? [item.command] : []))
    })) : [];
    const target = [...history].reverse().flatMap(item => item.commands || []).find(command => command.actId)?.actId
      || (entry.context.elementId.startsWith('task:') ? entry.context.elementId.slice(5) : null);
    // The branch the conversation is in. A new task born in the context of another one
    // belongs beside it, not at the root: a request to split a task into two came in while
    // that task was held, and the model answered with addItem twice — two tasks in the root,
    // far from the project they belong to. The target's own parent is what makes «beside it»
    // expressible, so it is handed over rather than searched for among a hundred tasks.
    const targetParent = target ? (graph.items.find(item => item.id === target)?.parentId ?? null) : null;
    return {
      text: entry.text, target, targetParent, context: clone(entry.context), corrects: entry.corrects || null,
      // Not just what was asked for, but how it ended: a correction is usually a reaction to
      // the outcome, and the model was being handed the request without the result of it.
      action: root ? {
        id: root.id, text: root.text, answer: root.answer || '', commands: clone(root.commands || []),
        ...(() => { const { status, error, target: acted } = journal.action(root); return { status, error, target: acted }; })()
      } : null,
      correction: correctionOf({ entry, graph, journal }),
      history, tasks: clone(graph.items), graphRevision: graph.revision, today: new Date().toISOString().slice(0, 10)
    };
  }
  async invoke(modelContext) {
    if (this.resolveModel) return this.resolveModel({ ...clone(modelContext), modelContext: clone(modelContext) });
    const parsed = parseCommand(modelContext.text.trim(), modelContext.target);
    if (parsed.kind === 'one') {
      const command = toCommand(parsed.hypothesis, modelContext.target);
      if (command?.command === 'setParent') {
        const candidates = findCandidates(parsed.hypothesis.tail, adaptSnapshot(modelContext.tasks));
        if (candidates.length !== 1) return JSON.stringify({ answer: 'Уточните, в какую задачу перенести.', commands: [] });
        command.payload.parentId = candidates[0].id;
      }
      if (command) return JSON.stringify({ answer: '', commands: [command] });
    }
    return JSON.stringify({ answer: 'Не удалось однозначно понять команду. Уточните действие.', commands: [] });
  }
  parse(raw, modelContext = {}) {
    let value = raw;
    if (typeof raw === 'string') { try { value = JSON.parse(raw); } catch { fail('INVALID_DECISION', 'Модель вернула некорректный ответ'); } }
    if (!value || typeof value !== 'object' || Array.isArray(value)) fail('INVALID_DECISION', 'Модель вернула некорректный ответ');
    const answer = value.answer ?? value.reply ?? '';
    let commands = value.commands ?? [];
    if (typeof answer !== 'string' || !Array.isArray(commands) || commands.length > 10 || commands.some(command => !command || typeof command.command !== 'string')) fail('INVALID_DECISION', 'Модель вернула некорректный ответ');
    if (!commands.length && typeof modelContext.text === 'string') {
      const deterministic = parseCommand(modelContext.text.trim(), modelContext.target);
      if (deterministic.kind === 'one') {
        const command = toCommand(deterministic.hypothesis, modelContext.target);
        if (command?.command === 'setParent') {
          const candidates = findCandidates(deterministic.hypothesis.tail, adaptSnapshot(modelContext.tasks || []));
          if (candidates.length === 1) command.payload.parentId = candidates[0].id;
          else commands = [];
        }
        if (command?.command !== 'setParent' || command.payload.parentId) commands = command ? [command] : [];
      }
    }
    const taskIds = new Set((modelContext.tasks || []).map(task => task.id));
    const targetCommands = new Set(['addChild', 'editItem', 'setStatus', 'setParent', 'setTags', 'setDeadline', 'toggleCollapse', 'deleteItem']);
    const parsedCommands = clone(commands).map(command => {
      if (targetCommands.has(command.command) && !taskIds.has(command.actId) && taskIds.has(modelContext.target)) command.actId = modelContext.target;
      // A task created while another one is in hand stays in that branch. The prompt asks for
      // this, and this is what holds when the model answers with addItem anyway: the request
      // was about a task, so the root is the one place the answer cannot belong.
      if (command.command === 'addItem' && taskIds.has(modelContext.target)) {
        command.command = 'addChild';
        command.actType = 'task';
        command.actId = modelContext.targetParent && taskIds.has(modelContext.targetParent)
          ? modelContext.targetParent
          : modelContext.target;
      }
      if (command.command === 'setStatus' && typeof command.payload === 'string') command.payload = { status: command.payload };
      return command;
    });
    return { answer, commands: parsedCommands };
  }
}
