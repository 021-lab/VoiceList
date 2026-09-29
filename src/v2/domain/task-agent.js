import { parseCommand, toCommand } from '../../command-resolver.js';
import { findCandidates } from '../../resolver.js';
import { adaptSnapshot } from '../../snapshot-adapter.js';
import { clone, fail } from './contracts.js';

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
