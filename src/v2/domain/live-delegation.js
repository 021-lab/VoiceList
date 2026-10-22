/** What the voice hears back from a delegated request, and when.
 *  Pure — the host owns the clock and the socket — so the owner's timing rule is testable. */

/** Past this a result is late: it no longer cuts in and waits for the next question. */
export const QUICK_RESULT_MS = 3_000;

/** Held confirmations kept for the next pause. Older ones are dropped: by then the newer
 *  ones are what matters, and the owner allows a confirmation to go unspoken. */
export const MAX_HELD = 2;

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

/** The owner's rule. A failure is always spoken — as an aside when late. An answer the person
 *  waits for is spoken unless they have moved on, then the voice only keeps it in mind. A
 *  quick confirmation with no conversation going on is spoken at once; any other is held
 *  until the person asks the next thing and said in the pause before that answer. */
export function deliveryFor({ elapsedMs, active, failed, answer = false, text }) {
  const late = elapsedMs >= QUICK_RESULT_MS;
  if (failed) return { kind: 'speak', content: late ? `Кстати, ${lowerFirst(text)}` : text };
  if (answer) return active ? { kind: 'quiet', content: text } : { kind: 'speak', content: text };
  if (!late && !active) return { kind: 'speak', content: text };
  return { kind: 'hold', content: text };
}

const LEAD_WORDS = new Set(['добавил', 'теперь', 'заметка', 'срок', 'промпт', 'перенёс', 'сделал', 'готово']);

/** The task a confirmation is about, as a stem that survives Russian endings: «Голден на
 *  паузе» → «голде», matched in «Голден», «Голдена». */
function subjectStem(text) {
  const words = String(text || '').toLowerCase().replace(/[«»"':,.…]/g, ' ').split(/\s+/).filter(Boolean);
  const word = words.find(item => !LEAD_WORDS.has(item) && item.length >= 3);
  return word ? word.slice(0, 5) : null;
}

/** Held confirmations the voice has not already covered in its own words. A newer message
 *  that names the task makes the aside a repeat, and the owner allows dropping it. */
export function stillUnsaid(held = [], voiceSince = []) {
  const said = voiceSince.join(' ').toLowerCase();
  return held.filter(text => { const stem = subjectStem(text); return !stem || !said.includes(stem); });
}

/** Held confirmations fill the gap after the next question: «Думаю, кстати, клубника на паузе»
 *  while the backend works on it, or «Кстати, …» after the voice has answered by itself. */
export function asideFor(held = [], { thinking = true } = {}) {
  if (!held.length) return null;
  const list = held.map(lowerFirst).join(', ');
  return thinking ? `Думаю, кстати, ${list}.` : `Кстати, ${list}.`;
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
