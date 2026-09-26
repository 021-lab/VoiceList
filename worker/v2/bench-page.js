/** A bench that runs where the person is.
 *
 *  Latency here is geography, and geography cannot be measured from somewhere else: the same
 *  document object is nine milliseconds away in Singapore and half a second away in
 *  Washington. So the measurement has to happen on the phone that will be doing the talking.
 *
 *  Each row separates one part of the path, because the parts have different cures: the
 *  phone's own network, the distance from the edge to the object, and the work inside it. */
export const BENCH_PAGE = `<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Стенд задержек</title>
<style>
  :root { color-scheme: light dark; --bg:#fff; --fg:#1a1a2e; --muted:#6b7280; --line:#e5e7eb; --panel:#f7f8fa; --good:#1b8f4b; --warn:#b26a00; --bad:#c04040; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161a; --fg:#e8eaed; --muted:#9aa3af; --line:#2a2e35; --panel:#1b1f25; --good:#4ade80; --warn:#f0b429; --bad:#ff6b6b; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; padding:16px; }
  h1 { font-size:19px; margin:0 0 4px; }
  .where { color:var(--muted); font-size:13px; margin-bottom:16px; }
  button { font:inherit; font-weight:600; padding:14px 20px; width:100%; border-radius:12px; border:0; background:#2d6cdf; color:#fff; cursor:pointer; }
  button[disabled] { opacity:.5; }
  table { width:100%; border-collapse:collapse; margin-top:18px; }
  th { text-align:left; font-size:12px; color:var(--muted); font-weight:600; padding:6px 0; border-bottom:1px solid var(--line); }
  td { padding:11px 0; border-bottom:1px solid var(--line); vertical-align:baseline; }
  td.value { text-align:right; font-variant-numeric:tabular-nums; font-weight:700; white-space:nowrap; }
  .runs { color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }
  .hint { color:var(--muted); font-size:13px; margin-top:6px; }
  .good { color:var(--good); } .warn { color:var(--warn); } .bad { color:var(--bad); }
  .verdict { margin-top:20px; padding:14px 16px; background:var(--panel); border-radius:12px; font-size:14px; }
  .verdict b { font-size:16px; }
</style>
</head>
<body>
<h1>Стенд задержек</h1>
<div class="where" id="where">узел и объект определятся на первом прогоне</div>
<button id="run" type="button">Прогнать</button>
<table>
  <thead><tr><th>Что меряем</th><th style="text-align:right">медиана</th></tr></thead>
  <tbody id="rows"></tbody>
</table>
<div class="verdict" id="verdict" hidden></div>

<script>
const RUNS = 5;
const $ = (id) => document.getElementById(id);
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

/** Every measurement is the same shape: a name, what it isolates, and how to take it once. */
const STEPS = [
  { name: 'Сеть до узла Cloudflare', hint: 'ваш интернет, без сервера',
    take: async () => { const at = performance.now(); await fetch('/health', { cache: 'no-store' }); return performance.now() - at; } },
  { name: 'Узел → объект документа', hint: 'расстояние до данных',
    take: async () => { const r = await (await fetch('/api/v2/where', { cache: 'no-store' })).json(); note(r); return r.probes[0].ms; } },
  { name: 'Чтение документа', hint: 'измерено внутри воркера', phase: 'чтение документа' },
  { name: 'Добавить одну задачу', hint: 'измерено внутри воркера', phase: 'одна задача' },
  { name: 'Добавить две задачи', hint: 'измерено внутри воркера', phase: 'две задачи' },
  { name: 'Полный круг с телефона', hint: 'столько ждёт голосовая модель',
    take: async () => {
      const at = performance.now();
      await fetch('/api/live/gemini/tools', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ toolCall: { functionCalls: [{ id: 'bench' + Date.now(), name: 'getFrontier', args: {} }] } }) });
      return performance.now() - at;
    } }
];

function note(payload) {
  if (!payload) return;
  const edge = payload.edge || '';
  const object = payload.object || '';
  $('where').textContent = 'Входной узел: ' + edge + (object ? ' · объект: ' + object : '');
}

function paint(step, values, done) {
  let row = document.getElementById('row-' + step.name);
  if (!row) {
    row = document.createElement('tr');
    row.id = 'row-' + step.name;
    row.innerHTML = '<td><div class="label"></div><div class="hint"></div><div class="runs"></div></td><td class="value"></td>';
    $('rows').append(row);
    row.querySelector('.label').textContent = step.name;
    row.querySelector('.hint').textContent = step.hint;
  }
  const value = row.querySelector('.value');
  if (!done) { value.textContent = '…'; return; }
  const ms = Math.round(median(values));
  value.textContent = ms + ' мс';
  value.className = 'value ' + (ms < 120 ? 'good' : ms < 400 ? 'warn' : 'bad');
  row.querySelector('.runs').textContent = values.map(Math.round).join(', ');
}

/** The worker-side phases all come from one request, so the whole bench is run once per pass
 *  and its phases are handed to the rows that asked for them. */
async function benchPhases() {
  const payload = await (await fetch('/api/v2/bench', { cache: 'no-store' })).json();
  note(payload);
  return Object.fromEntries(payload.phases.map(phase => [phase.name, phase.ms]));
}

async function run() {
  const button = $('run');
  button.disabled = true;
  button.textContent = 'Меряю…';
  const collected = new Map(STEPS.map(step => [step.name, []]));
  for (const step of STEPS) paint(step, [], false);

  for (let pass = 0; pass < RUNS; pass++) {
    const phases = await benchPhases();
    for (const step of STEPS) {
      const value = step.phase ? phases[step.phase] : await step.take();
      collected.get(step.name).push(value);
      paint(step, collected.get(step.name), pass === RUNS - 1);
    }
    button.textContent = 'Меряю… ' + (pass + 1) + ' из ' + RUNS;
  }

  const distance = median(collected.get('Узел → объект документа'));
  const round = median(collected.get('Полный круг с телефона'));
  const verdict = $('verdict');
  verdict.hidden = false;
  verdict.innerHTML = '<b>' + Math.round(round) + ' мс</b> ждёт голосовая модель на один вызов инструмента.<br>' +
    'Из них ' + Math.round(distance) + ' мс — дорога от узла до объекта' +
    (distance < 50 ? ', то есть объект рядом с вами.' : ' — объект далеко, это главное, что стоит чинить.');

  button.disabled = false;
  button.textContent = 'Прогнать ещё раз';
}

$('run').addEventListener('click', run);
</script>
</body>
</html>`;
