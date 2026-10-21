// Runs the text agent's instruction against the phrase corpus: the whole corpus, then the whole
// corpus again, --runs times. Reports accuracy of each run, whether the runs answered exactly
// alike, and response time. The request is the production one (resolveOpenAI); only where it
// goes and the sampling parameters are swapped, through fetchImpl.
//
//   node scripts/eval-text-agent.mjs --provider openai --model gpt-6-luna \
//     [--params '{"temperature":0,"seed":1}'] [--runs 3] [--concurrency 4] [--json out.json]
//
// Providers: openai (OPENAI_API_KEY), gemini (GEMINI_API_KEY), openrouter (Openrouter_api_key),
// ollama (OLLAMA_URL, e.g. http://mac-mini:11434). A null in --params removes that field from
// the production request — store and reasoning_effort are OpenAI's and others may refuse them.
import { readFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { resolveOpenAI } from '../worker/v2/model.js';

const { values: opt } = parseArgs({ options: {
  provider: { type: 'string', default: 'openai' }, model: { type: 'string', default: 'gpt-6-luna' },
  params: { type: 'string', default: '{}' }, runs: { type: 'string', default: '3' },
  concurrency: { type: 'string', default: '4' }, json: { type: 'string' }, quiet: { type: 'boolean', default: false }
} });
const PROVIDERS = {
  openai: { url: 'https://api.openai.com/v1/chat/completions', key: process.env.OPENAI_API_KEY },
  gemini: { url: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions', key: process.env.GEMINI_API_KEY },
  openrouter: { url: 'https://openrouter.ai/api/v1/chat/completions', key: process.env.Openrouter_api_key || process.env.OPENROUTER_API_KEY },
  ollama: { url: `${(process.env.OLLAMA_URL || 'http://localhost:11434').replace(/\/$/, '')}/v1/chat/completions`, key: 'ollama' }
};
const provider = PROVIDERS[opt.provider];
if (!provider?.key) { console.error(`Нет ключа или провайдера: ${opt.provider}`); process.exit(2); }
const params = JSON.parse(opt.params);
const corpus = JSON.parse(readFileSync(new URL('../docs/text-agent-corpus.json', import.meta.url), 'utf8'));

const fetchImpl = (url, options) => {
  const body = { ...JSON.parse(options.body), ...params };
  for (const [key, value] of Object.entries(body)) if (value === null) delete body[key];
  return fetch(provider.url, { ...options, signal: AbortSignal.timeout(120000), headers: { ...options.headers, Authorization: 'Bearer ' + provider.key }, body: JSON.stringify(body) });
};

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
const parseExpected = (line) => {
  const [, name, args] = line.match(/^(\w+)\((.*)\)$/);
  const parts = args.split(', ');
  return `${name}(${parts.map((arg, index) => index === parts.length - 1 ? norm(arg) : arg).join(', ')})`;
};
const same = (got, want) => got.length === want.length && [...got].sort().join('|') === [...want].sort().join('|');

async function runCase(item) {
  const target = item.target;
  const targetParent = target ? corpus.tasks.find(task => task.id === target)?.parentId ?? null : null;
  const modelContext = {
    text: item.text, target, targetParent, context: { elementId: target ? `task:${target}` : 'app', view: 'list' },
    corrects: null, action: null, correction: null, history: [],
    tasks: corpus.tasks.map(task => ({ ...task, tags: [] })), today: corpus.today
  };
  const started = performance.now();
  let answer = ''; let got = []; let error = null;
  try {
    const raw = JSON.parse(await resolveOpenAI({ apiKey: provider.key, geminiApiKey: provider.key, model: opt.model, modelContext, fetchImpl }));
    answer = raw.answer; got = raw.commands.map(notation);
  } catch (caught) { error = caught.message; }
  const ms = Math.round(performance.now() - started);
  const want = (item.expect || []).map(parseExpected);
  const alt = (item.alt || []).map(parseExpected);
  let ok = !error && (item.ask ? got.length === 0 && answer.trim() !== '' : same(got, want) || (alt.length > 0 && same(got, alt)));
  if (ok && item.rest) ok = norm(answer).includes(norm(item.rest));
  // What the run did, for comparing runs: the calls in order and whether anything was left
  // undone. The wording of a question or of «Не выполнено» is not compared — only its presence.
  const signature = error ? `ERROR ${error}` : `${got.join('; ')} | ${answer.trim() ? (got.length ? 'notDone' : 'text') : '-'}`;
  return { id: item.id, ok, got, answer, error, ms, signature };
}

async function runCorpus() {
  const results = [];
  const queue = [...corpus.cases];
  const concurrency = Number(opt.concurrency);
  await Promise.all(Array.from({ length: concurrency }, async () => {
    while (queue.length) { const item = queue.shift(); results.push(await runCase(item)); }
  }));
  return corpus.cases.map(item => results.find(result => result.id === item.id));
}

const runs = [];
for (let index = 0; index < Number(opt.runs); index++) runs.push(await runCorpus());

const percentile = (values, p) => { const sorted = [...values].sort((a, b) => a - b); return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))]; };
const times = runs.flat().filter(result => !result.error).map(result => result.ms);
const unstable = corpus.cases.filter((item, index) => new Set(runs.map(run => run[index].signature)).size > 1).map(item => item.id);
const summary = {
  provider: opt.provider, model: opt.model, params, cases: corpus.cases.length,
  correct: runs.map(run => run.filter(result => result.ok).length),
  errors: runs.flat().filter(result => result.error).length,
  identical: unstable.length === 0, unstable,
  medianMs: percentile(times, 0.5), p90Ms: percentile(times, 0.9),
  failed: [...new Set(runs.flatMap(run => run.filter(result => !result.ok).map(result => result.id)))]
};
if (!opt.quiet) {
  for (const [index, item] of corpus.cases.entries()) {
    const seen = runs.map(run => run[index]);
    if (seen.every(result => result.ok) && !unstable.includes(item.id)) continue;
    console.log(`✗ ${item.id} «${item.text}»`);
    for (const result of seen) console.log(`   ${result.ok ? '✓' : '✗'} ${result.error ? 'ОШИБКА ' + result.error : result.got.join('; ') || '(нет вызовов)'}  ${JSON.stringify(result.answer)}`);
  }
}
console.log(JSON.stringify(summary));
if (opt.json) writeFileSync(opt.json, JSON.stringify({ summary, runs }, null, 1));
process.exit(summary.identical && summary.correct.every(count => count === corpus.cases.length) ? 0 : 1);
