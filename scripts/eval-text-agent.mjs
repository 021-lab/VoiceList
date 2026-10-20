// Runs the text agent's instruction against the phrase corpus and prints what it got wrong.
// Real model calls: needs OPENAI_API_KEY. Usage: node scripts/eval-text-agent.mjs [model]
import { readFileSync } from 'node:fs';
import { resolveOpenAI } from '../worker/v2/model.js';

const model = process.argv[2] || 'gpt-6-luna';
const corpus = JSON.parse(readFileSync(new URL('../docs/text-agent-corpus.json', import.meta.url), 'utf8'));
const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) { console.error('OPENAI_API_KEY не задан'); process.exit(2); }

const norm = (text) => String(text).toLowerCase().replace(/ё/g, 'е').replace(/[«»"'.,!?:;\s]+$/u, '').replace(/^[«»"'\s]+/u, '').replace(/\s+/g, ' ').trim();
// The command as the instruction writes it: addChild(inbox, купить молоко).
const notation = (command) => {
  const { command: name, actId, payload = {} } = command;
  if (name === 'addItem') return `addItem(${norm(payload.title)})`;
  if (name === 'addChild') return `${payload.status === 'Info' ? 'addInfo' : 'addChild'}(${actId}, ${norm(payload.title)})`;
  if (name === 'editItem') return `editItem(${actId}, ${norm(payload.title)})`;
  if (name === 'setStatus') return `setStatus(${actId}, ${norm(payload.status)})`;
  if (name === 'setDeadline') return `setDeadline(${actId}, ${payload.deadline})`;
  if (name === 'setParent') return `setParent(${actId}, ${payload.parentId})`;
  return name;
};
const parseExpected = (line) => { const [, name, args] = line.match(/^(\w+)\((.*)\)$/); return `${name}(${args.split(', ').map((arg, index) => index === args.split(', ').length - 1 ? norm(arg) : arg).join(', ')})`; };
const same = (got, want) => got.length === want.length && [...got].sort().join('|') === [...want].sort().join('|');

async function run(item) {
  const target = item.target;
  const targetParent = target ? corpus.tasks.find(task => task.id === target)?.parentId ?? null : null;
  const modelContext = {
    text: item.text, target, targetParent, context: { elementId: target ? `task:${target}` : 'app', view: 'list' },
    corrects: null, action: null, correction: null, history: [],
    tasks: corpus.tasks.map(task => ({ ...task, tags: [] })), today: corpus.today
  };
  const started = Date.now();
  let answer = ''; let got = []; let error = null;
  try {
    const raw = JSON.parse(await resolveOpenAI({ apiKey, model, modelContext }));
    answer = raw.answer; got = raw.commands.map(notation);
  } catch (caught) { error = caught.message; }
  const want = (item.expect || []).map(parseExpected);
  const alt = (item.alt || []).map(parseExpected);
  let ok = !error && (item.ask ? got.length === 0 && answer.trim() !== '' : same(got, want) || (alt.length > 0 && same(got, alt)));
  if (ok && item.rest) ok = norm(answer).includes(norm(item.rest));
  return { item, ok, got, want, answer, error, ms: Date.now() - started };
}

const results = [];
for (let index = 0; index < corpus.cases.length; index += 6) results.push(...await Promise.all(corpus.cases.slice(index, index + 6).map(run)));
for (const { item, ok, got, want, answer, error, ms } of results) {
  if (ok) continue;
  console.log(`✗ ${item.id} «${item.text}» [T=${item.target}]`);
  console.log(`   ожидалось: ${item.ask ? '(вопрос без вызовов)' : want.join('; ')}${item.rest ? ` + «Не выполнено: …${item.rest}…»` : ''}`);
  console.log(`   получено:  ${error ? 'ОШИБКА ' + error : got.join('; ') || '(нет вызовов)'}   ответ: ${JSON.stringify(answer)}   ${ms} мс`);
}
const passed = results.filter(result => result.ok).length;
const times = results.map(result => result.ms).sort((a, b) => a - b);
console.log(`\n${model}: ${passed}/${results.length} верно, медиана ${times[Math.floor(times.length / 2)]} мс`);
process.exit(passed === results.length ? 0 : 1);
