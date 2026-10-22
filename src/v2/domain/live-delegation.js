/** What the voice hears back from a delegated request, and when.
 *  Pure — the host owns the clock and the socket — so the owner's timing rule is testable. */

/** Past this the result is late: the voice says «Думаю» meanwhile and «Кстати, …» after. */
export const QUICK_RESULT_MS = 3_000;

const STATUS_WORDS = { Open: 'открыта', Focus: 'в фокусе', Pause: 'на паузе', Done: 'готово', Archive: 'в архиве', Info: 'теперь заметка' };

/** Short beats clear: the first few words of a title are enough to recognise it by ear. */
function spoken(title) {
  const words = String(title || '').replace(/\s+/g, ' ').trim().split(' ');
  return words.length > 4 ? `${words.slice(0, 4).join(' ')}…` : words.join(' ');
}

function shortDate(value) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(value || ''));
  return match ? `${Number(match[3])}.${match[2]}` : String(value || '');
}

/** One applied command as a few words. The title is read after the change, so a rename
 *  already says the new name and a new task its own. */
export function confirmText(command, items = []) {
  const title = (id) => spoken(items.find(item => item.id === id)?.title);
  const subject = title(command?.actId);
  switch (command?.command) {
    case 'addItem':
    case 'addChild':
      return command.payload?.status === 'Info' ? 'Заметка добавлена' : `Добавил: ${spoken(command.payload?.title)}`;
    case 'setStatus': return subject ? `${subject} ${STATUS_WORDS[command.payload?.status] || 'готово'}` : 'Сделал';
    case 'setDeadline': return subject ? `${subject} до ${shortDate(command.payload?.deadline)}` : 'Срок стоит';
    case 'editItem': return `Теперь ${spoken(command.payload?.title)}`;
    case 'setParent': return subject ? `${subject} перенесена` : 'Перенёс';
    default: return 'Готово';
  }
}

/** The owner's rule. A failure is always spoken. A quick result is spoken at once. A late one
 *  is spoken as an aside unless the conversation has moved on — then it goes to the voice's
 *  silent context, where it can still answer «а что с клубникой?». */
export function deliveryFor({ elapsedMs, movedOn, failed, text }) {
  const late = elapsedMs >= QUICK_RESULT_MS;
  if (failed) return { kind: 'commentary', content: late ? `Кстати, ${lowerFirst(text)}` : text };
  if (!late) return { kind: 'commentary', content: text };
  if (movedOn) return { kind: 'thinking', content: `Выполнено, не озвучено: ${text}` };
  return { kind: 'commentary', content: `Кстати, ${lowerFirst(text)}` };
}

function lowerFirst(text) {
  const value = String(text || '');
  return value ? value[0].toLowerCase() + value.slice(1) : value;
}

/** The recent conversation as plain role messages: the backend sees what was said, nothing of
 *  the protocol around it. */
export function buildSolInput(turns) {
  return (turns || []).map(turn => ({ role: turn.role === 'assistant' ? 'assistant' : 'user', content: String(turn.text ?? '') }));
}
