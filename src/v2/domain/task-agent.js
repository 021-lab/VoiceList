import { parseCommand, toCommand } from '../../command-resolver.js';
import { findCandidates } from '../../resolver.js';
import { adaptSnapshot } from '../../snapshot-adapter.js';

export class TaskAgent {
  constructor({ resolveModel = null } = {}) { this.resolveModel = resolveModel; }
  async run({ input, graph, journal }) {
    if (input.command) return { commands: [input.command] };
    const text = input.text.trim();
    const original = input.context.actionId ? journal.actions().find(x => x.id === input.context.actionId) : null;
    const target = original?.target || (input.context.elementId.startsWith('task:') ? input.context.elementId.slice(5) : null);
    if (original && /^(отмени|откатить|откат|назад|undo)[.!\s]*$/iu.test(text)) {
      return { commands: [{ command: 'rollbackAction', actId: original.id, actType: 'action', payload: {} }] };
    }
    const parsed = parseCommand(text, target);
    if (parsed.kind === 'one') {
      const command = toCommand(parsed.hypothesis, target);
      if (command?.command === 'setParent') {
        const candidates = findCandidates(parsed.hypothesis.tail, adaptSnapshot(graph.items));
        if (candidates.length !== 1) return { reply: 'Уточните, в какую задачу перенести.' };
        command.payload.parentId = candidates[0].id;
      }
      if (command) return { commands: [command] };
    }
    if (this.resolveModel) return this.resolveModel({ text, target, graph, action: original, dialogue: original ? journal.dialogue(original.id) : [] });
    return { reply: 'Не удалось однозначно понять команду. Уточните действие; свободный диалог станет доступен после настройки модели.' };
  }
}
