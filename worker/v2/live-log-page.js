/** Readable view of the session log: one line per record, click to open it whole.
 *  Served as a page rather than shipped in the app bundle — it is a working tool for
 *  reading what the framework returned, not part of the list interface. */
export const LIVE_LOG_PAGE = `<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Лог голосовых сессий</title>
<style>
  :root { color-scheme: light dark; --bg:#fff; --fg:#1a1a2e; --muted:#6b7280; --line:#e5e7eb; --panel:#f7f8fa; --in:#1b8f4b; --out:#2563eb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161a; --fg:#e8eaed; --muted:#9aa3af; --line:#2a2e35; --panel:#1b1f25; --in:#4ade80; --out:#7aa7ff; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
  header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:14px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
  h1 { font-size:17px; margin:0; flex:1 1 auto; }
  select, button { font:inherit; padding:7px 11px; border-radius:9px; border:1px solid var(--line); background:var(--panel); color:var(--fg); }
  button { cursor:pointer; }
  button.danger { border-color:#c04040; color:#c04040; background:transparent; }
  #stats { color:var(--muted); font-size:13px; flex-basis:100%; }
  ol { list-style:none; margin:0; padding:0; }
  li { border-bottom:1px solid var(--line); }
  .row { display:grid; grid-template-columns:54px 16px 1fr; gap:10px; padding:9px 16px; cursor:pointer; align-items:baseline; }
  .row:hover { background:var(--panel); }
  .row.plain { cursor:default; }
  .row.plain:hover { background:transparent; }
  .row.context .gist { color:var(--muted); }
  .at { color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }
  .dir { font-weight:700; }
  .dir.in { color:var(--in); }
  .dir.out { color:var(--out); }
  .gist { min-width:0; overflow-wrap:anywhere; }
  .kind { color:var(--muted); font-size:12px; margin-left:6px; }
  pre { margin:0; padding:12px 16px 18px 80px; background:var(--panel); font-size:12px; line-height:1.4; white-space:pre-wrap; overflow-wrap:anywhere; }
  .empty { padding:40px 16px; color:var(--muted); text-align:center; }
</style>
</head>
<body>
<header>
  <h1>Лог голосовых сессий</h1>
  <select id="session"><option value="">Все сессии</option></select>
  <button id="refresh" type="button">Обновить</button>
  <button id="clear" class="danger" type="button">Очистить</button>
  <div id="stats"></div>
</header>
<ol id="rows"></ol>
<script>
const $ = (id) => document.getElementById(id);
const short = (value, limit = 160) => { const text = String(value ?? ''); return text.length > limit ? text.slice(0, limit) + '…' : text; };

/** The point of the line is what happened, not which event type carried it. */
function gist(entry) {
  const p = entry.payload || {};
  const inner = p.event || {};
  const item = inner.item || {};
  switch (entry.type) {
    case 'vl.speech': return [p.role === 'user' ? 'речь' : 'ответ', p.text || ''];
    case 'vl.delegation': {
      const context = (p.context || []).map(turn => (turn.role === 'user' ? '👤 ' : '🤖 ') + turn.text).join('  ·  ');
      return ['спросили бэкенд', context || '(без контекста)'];
    }
    case 'vl.backend_text': return ['ответил бэкенд', p.text || ''];
    case 'session.delegation.created': return ['делегирование', (p.delegation?.target || '') + (p.delegation?.offset_ms != null ? ' · ' + Math.round(p.delegation.offset_ms / 1000) + ' c' : '')];
    case 'session.started': return ['сессия началась', p.session?.model || ''];
    case 'session.closed': return ['сессия закрыта', p.reason || ''];
    case 'session.close': return ['просим закрыть сессию', ''];
    case 'session.usage.updated': return ['расход', short(JSON.stringify(p.usage || {}), 90)];
    case 'session.instructions.append': return ['правка инструкций', p.content || ''];
    case 'session.instructions.appended': return ['инструкции приняты', ''];
    case 'session.update': return ['обновление сессии', short(JSON.stringify(p.session || {}), 90)];
    case 'response.create': return ['продолжить ответ', ''];
    case 'response.item.create': return ['результат вызова', short((p.item || item).output || '', 160)];
    case 'vl.session.requested': return ['запрос сессии', 'модель ' + (p.backendModel || '')];
    case 'vl.session.created': return ['сессия создана', (p.requested?.backendModel || '') + ' · store ' + p.requested?.store];
    case 'vl.session.create_failed': return ['GPT-Live отклонил', short(p.detail || '', 140)];
    case 'vl.session.unreachable': return ['не дозвонились', p.error?.message || ''];
    case 'vl.session.stopped': return ['остановлено', p.reason || ''];
    case 'vl.sideband.closed': return ['sideband закрыт', ''];
    case 'vl.sideband.failed': return ['sideband не открылся', p.error?.message || ''];
    case 'vl.event.unparsed': return ['нечитаемый кадр', short(p.raw || '', 120)];
    case 'vl.dispatch.failed': return ['ошибка обработки', p.error?.message || ''];
  }
  if (entry.type.endsWith('response.output_item.done') && item.type === 'function_call') return ['вызов ' + item.name, short(item.arguments || '', 140)];
  if (entry.type.endsWith('response.created')) return ['ответ модели начат', ''];
  if (entry.type.endsWith('response.completed')) return ['ответ модели готов', ''];
  if (entry.type.endsWith('response.output_item.added')) return ['элемент ответа', item.type || ''];
  return [entry.type, ''];
}

let rows = [];
function render() {
  const list = $('rows');
  list.textContent = '';
  const picked = $('session').value;
  const shown = rows.filter(entry => !picked || entry.liveSessionId === picked);
  if (!shown.length) { list.innerHTML = '<li class="empty">Записей нет.</li>'; return; }
  for (const entry of shown) {
    const [kind, detail] = gist(entry);
    const speech = entry.type === 'vl.speech' || entry.type === 'vl.backend_text';
    const li = document.createElement('li');
    const row = document.createElement('div');
    row.className = speech ? 'row plain' : (entry.type === 'vl.delegation' ? 'row context' : 'row');
    row.innerHTML = '<span class="at"></span><span class="dir"></span><span class="gist"></span>';
    row.querySelector('.at').textContent = entry.at.slice(11, 19);
    const dir = row.querySelector('.dir');
    dir.textContent = entry.direction === 'out' ? '→' : '←';
    dir.classList.add(entry.direction === 'out' ? 'out' : 'in');
    const body = row.querySelector('.gist');
    body.textContent = detail ? short(detail, 400) : kind;
    if (detail) { const label = document.createElement('span'); label.className = 'kind'; label.textContent = kind; body.append(label); }
    li.append(row);
    // Speech rows hold nothing beyond their own text, so they do not open.
    if (!speech) {
      const details = document.createElement('pre');
      details.hidden = true;
      row.addEventListener('click', () => {
        if (details.hidden) details.textContent = '# seq ' + entry.seq + '  ' + entry.type + '\\n' + JSON.stringify(entry.payload, null, 2);
        details.hidden = !details.hidden;
      });
      li.append(details);
    }
    list.append(li);
  }
}

async function load() {
  const data = await (await fetch('/api/live/log?limit=1000')).json();
  rows = data.entries || [];
  $('stats').textContent = data.stats.events + ' записей · ' + Math.round(data.stats.payloadBytes / 1024) + ' КБ';
  const sessions = [...new Set(rows.map(entry => entry.liveSessionId).filter(Boolean))];
  const picker = $('session');
  const kept = picker.value;
  picker.textContent = '';
  picker.append(new Option('Все сессии', ''));
  for (const id of sessions) picker.append(new Option(id.slice(0, 26), id));
  picker.value = kept;
  render();
}

$('refresh').addEventListener('click', load);
$('session').addEventListener('change', render);
$('clear').addEventListener('click', async () => {
  if (!confirm('Удалить все записи лога? Это необратимо.')) return;
  await fetch('/api/live/log/clear', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  load();
});
load();
</script>
</body>
</html>`;
