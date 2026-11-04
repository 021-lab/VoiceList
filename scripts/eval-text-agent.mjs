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
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { resolveOpenAI } from '../worker/v2/model.js';
import { resolveDecision } from './decision-agent.mjs';

const { values: opt } = parseArgs({ options: {
  provider: { type: 'string', default: 'openai' }, model: { type: 'string', default: 'gpt-6-luna' },
  params: { type: 'string', default: '{}' }, runs: { type: 'string', default: '3' },
  concurrency: { type: 'string', default: '4' }, json: { type: 'string' }, quiet: { type: 'boolean', default: false },
  // At most N requests started per S seconds: --rate 5/30 for the AI Gateway, whose limit is that.
  rate: { type: 'string' },
  // A saved instruction (docs/text-agent-v1/instruction.txt) in place of the current one.
  instruction: { type: 'string' },
  // Another phrase corpus in place of docs/text-agent-corpus.json.
  corpus: { type: 'string' }
} });
const PROVIDERS = {
  openai: { url: 'https://api.openai.com/v1/chat/completions', key: process.env.OPENAI_API_KEY },
  gemini: { url: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions', key: process.env.GEMINI_API_KEY },
  openrouter: { url: 'https://openrouter.ai/api/v1/chat/completions', key: process.env.Openrouter_api_key || process.env.OPENROUTER_API_KEY },
  ollama: { url: `${(process.env.OLLAMA_URL || 'http://localhost:11434').replace(/\/$/, '')}/v1/chat/completions`, key: 'ollama' },
  // A worker with the AI binding that takes this request and runs it on Workers AI (AI_BENCH_URL,
  // AI_BENCH_KEY). It reports the time spent inside env.AI.run as x-infer-ms.
  'workers-ai': { url: process.env.AI_BENCH_URL, key: process.env.AI_BENCH_KEY },
  // Clef + Clairic through the same worker (scripts/decision-agent.mjs); --model clef+clairic or
  // clef-flash+clairic, --params '{"gateway":"…"}'.
  decision: { url: process.env.AI_BENCH_URL, key: process.env.AI_BENCH_KEY }
};
const provider = PROVIDERS[opt.provider];
if (!provider?.key) { console.error(`Нет ключа или провайдера: ${opt.provider}`); process.exit(2); }
const params = JSON.parse(opt.params);
const instruction = opt.instruction ? readFileSync(opt.instruction, 'utf8') : null;
const corpus = JSON.parse(readFileSync(opt.corpus || new URL('../docs/text-agent-corpus.json', import.meta.url), 'utf8'));

const [rateCount, rateSeconds] = (opt.rate || '0/0').split('/').map(Number);
if (opt.rate && !(rateCount > 0 && rateSeconds > 0)) { console.error('--rate пишется как N/S, например 5/30'); process.exit(2); }
// The gateway counts requests, not runs: two runs started one after the other share the limit.
// So the times of the last requests are kept in a file, and every run reads it before sending.
const rateFile = join(tmpdir(), `eval-text-agent-rate-${rateCount}-${rateSeconds}.json`);
const readStarts = () => { try { return JSON.parse(readFileSync(rateFile, 'utf8')); } catch { return []; } };
async function paced() {
  if (!rateCount) return;
  for (;;) {
    const now = Date.now();
    const starts = readStarts().filter(time => now - time < rateSeconds * 1000).sort((x, y) => x - y);
    if (starts.length < rateCount) { writeFileSync(rateFile, JSON.stringify([...starts, now])); return; }
    await new Promise(resolve => setTimeout(resolve, starts[0] + rateSeconds * 1000 - now + 50));
  }
}

const fetchFor = (timing) => async (url, options) => {
  await paced();
  const body = { ...JSON.parse(options.body), ...params };
  if (instruction && body.messages?.[0]?.role === 'system') body.messages[0] = { ...body.messages[0], content: instruction };
  for (const [key, value] of Object.entries(body)) if (value === null) delete body[key];
  timing.sentAt ??= performance.now();
  const response = await fetch(provider.url, { ...options, signal: AbortSignal.timeout(120000), headers: { ...options.headers, Authorization: 'Bearer ' + provider.key }, body: JSON.stringify(body) });
  const server = response.headers.get('x-infer-ms');
  if (server) timing.serverMs = Number(server);
  if (!response.ok) { const text = await response.text(); return { ok: false, status: response.status, body: { cancel: async () => {} }, text }; }
  return response;
};

// A model that does not chat: its input goes to the worker as it is. Both calls of a request leave
// together; the request's time is the slower of the two.
const runRaw = (timing) => async (model, input) => {
  await paced();
  timing.sentAt ??= performance.now();
  const response = await fetch(provider.url, { method: 'POST', signal: AbortSignal.timeout(120000), headers: { Authorization: 'Bearer ' + provider.key, 'Content-Type': 'application/json' }, body: JSON.stringify({ model, raw: input, ...params }) });
  const server = Number(response.headers.get('x-infer-ms'));
  if (server) timing.serverMs = Math.max(timing.serverMs || 0, server);
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(`${model}: ${data.error?.message || response.status}`.slice(0, 200));
  return data;
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
  const timing = {};
  let answer = ''; let got = []; let error = null;
  try {
    const raw = opt.provider === 'decision'
      ? await resolveDecision(modelContext, runRaw(timing), { clefModel: opt.model.split('+')[0] })
      : JSON.parse(await resolveOpenAI({ apiKey: provider.key, geminiApiKey: provider.key, model: opt.model, modelContext, fetchImpl: fetchFor(timing) }));
    answer = raw.answer; got = raw.commands.map(notation);
  } catch (caught) { error = caught.message; }
  // Time from the request leaving, not from the case starting: waiting for the rate limit is not latency.
  const ms = Math.round(performance.now() - (timing.sentAt ?? started));
  const want = (item.expect || []).map(parseExpected);
  const alt = (item.alt || []).map(parseExpected);
  let ok = !error && (item.ask ? got.length === 0 && answer.trim() !== '' : same(got, want) || (alt.length > 0 && same(got, alt)));
  if (ok && item.rest) ok = norm(answer).includes(norm(item.rest));
  // What the run did, for comparing runs: the calls in order and whether anything was left
  // undone. The wording of a question or of «Не выполнено» is not compared — only its presence.
  const signature = error ? `ERROR ${error}` : `${got.join('; ')} | ${answer.trim() ? (got.length ? 'notDone' : 'text') : '-'}`;
  return { id: item.id, ok, got, answer, error, ms, serverMs: timing.serverMs, signature };
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
  provider: opt.provider, model: opt.model, params, instruction: opt.instruction || 'current', cases: corpus.cases.length,
  correct: runs.map(run => run.filter(result => result.ok).length),
  errors: runs.flat().filter(result => result.error).length,
  identical: unstable.length === 0, unstable,
  medianMs: percentile(times, 0.5), p90Ms: percentile(times, 0.9),
  ...(runs.flat().some(result => result.serverMs) ? { serverMedianMs: percentile(runs.flat().filter(result => result.serverMs).map(result => result.serverMs), 0.5), serverP90Ms: percentile(runs.flat().filter(result => result.serverMs).map(result => result.serverMs), 0.9) } : {}),
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
