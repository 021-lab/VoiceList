/** Past conversations as the voice models can read them: what the person said, what the voice
 *  answered and what was done, by session, oldest first. A voice session starts with no memory
 *  of the one before it; asked «что ты сделал?» it answered that there had been nothing. The
 *  journal has all of it. Pure: entries in, lines out. */

const SOURCES = { 'gpt-live': 'GPT-Live', 'gemini-live': 'Gemini', agent: 'текст', user: 'текст', schedule: 'расписание' };
const VERBS = {
  addItem: 'добавить', addChild: 'добавить внутрь', setStatus: 'статус', setDeadline: 'срок', editItem: 'переименовать',
  setParent: 'перенести', setTags: 'метка', deleteItem: 'удалить', reorderItems: 'переставить', undo: 'отменить последнее',
  rollbackAction: 'откатить действие'
};
/** Button presses, MCP calls and benchmarks are not part of a conversation. */
const QUIET_SOURCES = new Set(['ui', 'mcp', 'bench']);

export const HISTORY_LINES = 80;

const minute = (at) => String(at || '').replace('T', ' ').slice(0, 16);
const short = (text, size = 200) => { const value = String(text || '').replace(/\s+/g, ' ').trim(); return value.length > size ? `${value.slice(0, size)}…` : value; };

/** The session a journal entry belongs to, read from its request key: «gemini:gm_1:call» and
 *  «gemini-live:gm_1:speech» are the same conversation. */
function sessionOf(entry) {
  // Typed requests come from the page under its own random key: they are one stream, «текст».
  if (entry.kind === 'text') return { id: '', name: SOURCES.user };
  const [kind, id] = String(entry.key?.clientKey || '').split(':');
  const source = entry.source || entry.command?.source || (kind === 'gemini' ? 'gemini-live' : kind);
  return { id: id || '', name: SOURCES[source] || source || 'интерфейс' };
}

/** The title a task had when the entry was made: a rename since would otherwise make the line
 *  name the task by what it became, and a rename line say «Кефир → Кефир». */
function titleThen(id, entry, technical, titles) {
  const changes = (technical?.executor?.[entry.id]?.outcomes || []).flatMap(outcome => outcome.changes || []);
  return changes.find(change => change.id === id)?.before?.title || titles.get(id) || '';
}

function describe(command, titles, entry, technical) {
  const p = command?.payload || {};
  const name = command?.command;
  const quoted = (text, size = 80) => (text ? `«${short(text, size)}»` : '');
  const target = quoted(titleThen(command?.actId, entry, technical, titles)) || command?.actId || '';
  const verb = VERBS[name] || name || 'действие';
  if (name === 'undo' || name === 'rollbackAction') return verb;
  if (name === 'addItem' || name === 'addChild') return [verb, quoted(p.title), p.status === 'Info' ? '(заметка)' : '', name === 'addChild' ? `в ${target}` : ''].filter(Boolean).join(' ');
  if (name === 'addTree') return `добавить ${quoted(p.tree?.title)} с подзадачами ${(p.tree?.children || []).map(child => quoted(child.title, 60)).join(', ')}${command.actId ? ` в ${target}` : ''}`;
  if (name === 'editItem') return `${verb} ${target} → ${quoted(p.title)}`;
  if (name === 'setParent') return `${verb} ${target} ${p.parentId == null ? 'в корень' : `под ${quoted(titles.get(p.parentId), 60) || 'удалённую задачу'}`}`;
  if (name === 'setDeadline') return `${verb} ${target} ${p.deadline || 'снят'}`;
  if (name === 'setTags') return `${verb} ${target} ${p.tag || (p.tags || []).join(', ')}`.trim();
  return [verb, target, p.status].filter(Boolean).join(' ');
}

function outcome(entry, technical) {
  const ledger = technical?.executor?.[entry.id] || {};
  if (ledger.error) return `не вышло: ${short(ledger.error.message || ledger.error, 120)}`;
  if (['pending', 'waiting'].includes(ledger.status)) return 'выполняется';
  if (technical?.undoneEntries?.[entry.id]) return 'отменено';
  return 'выполнено';
}

export function formatHistory(entries = [], technical = {}, items = [], { limit = HISTORY_LINES } = {}) {
  const titles = new Map(items.map(item => [item.id, item.title]));
  const lines = [];
  let session = null;
  for (const entry of [...entries].sort((a, b) => a.cursor - b.cursor)) {
    let line = null;
    if (entry.kind === 'speech') {
      line = `${entry.role === 'assistant' ? 'Голос' : 'Человек'}: ${short(entry.text)}`;
    } else if (entry.kind === 'text') {
      const did = (entry.commands || []).map(command => describe(command, titles, entry, technical)).join('; ');
      line = `Человек (текстом): ${short(entry.text)}${entry.answer ? ` → ответ: ${short(entry.answer)}` : ''}${did ? ` → ${did}, ${outcome(entry, technical)}` : ''}`;
    } else if (entry.command && !QUIET_SOURCES.has(entry.command.source || 'ui')) {
      line = `Сделано: ${describe(entry.command, titles, entry, technical)} — ${outcome(entry, technical)}`;
    }
    if (!line) continue;
    const where = sessionOf(entry);
    if (!session || session.id !== where.id || session.name !== where.name) {
      session = where;
      lines.push(`— разговор ${where.name}${where.id ? ` ${where.id}` : ''}, ${minute(entry.at)} UTC —`);
    }
    lines.push(`${minute(entry.at).slice(11)} ${line}`);
  }
  // The newest part is what a question about «the last time» needs; the oldest goes first.
  if (!lines.length) return 'Журнал пуст: прошлых разговоров нет.';
  const kept = lines.slice(-limit);
  // A cut in the middle of a conversation still says which conversation it is.
  if (!kept[0].startsWith('—')) {
    const header = lines.slice(0, lines.length - kept.length).reverse().find(line => line.startsWith('—'));
    if (header) kept.unshift(header);
  }
  return kept.join('\n');
}
