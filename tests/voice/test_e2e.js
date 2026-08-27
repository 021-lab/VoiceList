// test_e2e.js — один сквозной тест на все граничные случаи.
// Путь: жест -> взвод -> мок захвата голоса -> транскрипция по словам ->
// кандидаты -> отпускание в конкретной зоне -> команда исполнителю.
// Голос замокан (asr-mock.js). Жесты в браузере — tests/e2e/gestures.spec.js.
//
// Запуск: node tests/voice/test_e2e.js   (или npm run test:voice)

'use strict';

import * as G from '../../src/gesture.js';
import * as CR from '../../src/command-resolver.js';
import * as A from '../../src/snapshot-adapter.js';
import { MockASR } from './asr-mock.js';
import { VoiceSession, validate } from '../../src/voice-session.js';

const C = G.C;
let fail = 0, pass = 0;

function check(name, ok, extra) {
  if (ok) { pass++; console.log('PASS', name); }
  else { fail++; console.log('FAIL', name, extra !== undefined ? JSON.stringify(extra) : ''); }
}

// ===== Граф «Ремонт» ==================================================
const T = (id, line1, parentId = null, status = 'Open') => ({ id, line1, parentId, status, order: 1 });
const TASKS = [
  T('inbox', 'Входящие'),
  T('1', 'Ремонт'),
  T('2', 'Ванная', '1'),
  T('3', 'Установить смеситель', '2'),
  T('4', 'Спальня', '1'),
  T('5', 'Купить обои', '4'),
  T('6', 'Свет', '4', 'Pause'),
  T('7', 'Купить лампочку', '6'),
  T('8', 'Покупки'),
  T('9', 'Купить смеситель', '8'),
  T('10', 'Старый список', null, 'Archive'),
  T('11', 'Прошлый ремонт', null, 'Done'),
];

// ===== Драйвер жеста ==================================================
function runGesture({ phrase, failure = null, interim = true, gesture = 'hold',
                      elementId = null, dy = 0, tasks = TASKS }) {
  const rec = new G.GestureRecognizer();
  const asr = new MockASR({ phrase, failure, interim });
  const s = new VoiceSession({ tasks, asr });

  rec.down({ t: 0, x: 200, y: 500, elementId });
  if (gesture === 'swipe') {
    rec.move({ t: 80, x: 200 + C.CONTEXT_DX_PX - 5, y: 500 });
  } else if (gesture === 'scroll') {
    rec.move({ t: 60, x: 200, y: 400 });
  } else {
    rec.tick({ t: C.LONGPRESS_MS });
  }
  if (rec.state !== 'armed') return { armed: false, state: rec.state };

  if (!s.arm(rec.contextId)) return { armed: false, denied: true, messages: s.messages };
  asr.speak();
  rec.move({ t: 800, x: 200, y: 500 - dy });
  s.move(dy);
  const out = s.release(dy);
  return { armed: true, mode: rec.mode, contextId: rec.contextId, out, session: s };
}

const cmdOf = (r) => (r.out && r.out.command) || null;
const isCmd = (r, command, extra = {}) => {
  const c = cmdOf(r);
  if (!c || c.command !== command) return false;
  return Object.entries(extra).every(([k, v]) =>
    k === 'status' || k === 'line1' || k === 'query' || k === 'parentId'
      ? c.payload[k] === v : c[k] === v);
};

// ===== 1. Жесты =======================================================
{
  const r = runGesture({ phrase: 'добавь купить молоко' });
  check('жест: удержание взводит оверлей без контекста', r.armed && r.mode === 'global' && r.contextId === null, r);
}
{
  const r = runGesture({ phrase: 'готово', gesture: 'swipe', elementId: '3' });
  check('жест: свайп влево даёт контекст', r.armed && r.mode === 'context' && r.contextId === '3', r);
}
{
  const r = runGesture({ phrase: 'добавь молоко', gesture: 'scroll' });
  check('жест: прокрутка отменяет жест, оверлей не взводится', !r.armed && r.state === 'scroll', r);
}
{
  const r = runGesture({ phrase: 'добавь молоко', gesture: 'hold', elementId: '3' });
  check('жест: удержание на элементе не даёт контекста', r.armed && r.contextId === null, r);
}
{
  const rec = new G.GestureRecognizer();
  rec.down({ t: 0, x: 10, y: 10 });
  const up = rec.up({ t: 100 });
  check('жест: короткое касание — тап', up.state === 'tap', up);
}
{
  const rec = new G.GestureRecognizer();
  rec.down({ t: 0, x: 200, y: 500, elementId: null });
  rec.tick({ t: C.LONGPRESS_MS });
  rec.move({ t: 500, x: 100, y: 500 });
  check('жест: свайп после взвода игнорируется', rec.mode === 'global' && rec.contextId === null, rec.mode);
}

// ===== 2. Геометрия оверлея ==========================================
{
  const cases = [
    ['ниже зоны отмены', G.selectAt(-C.CANCEL_ZONE_PX, 3).zone === 'cancel'],
    ['граница отмены не включена', G.selectAt(-C.CANCEL_ZONE_PX + 1, 3).zone === 'as-is'],
    ['мёртвая зона', G.selectAt(0, 3).zone === 'as-is'],
    ['верх мёртвой зоны', G.selectAt(C.DEADZONE_PX - 1, 3).zone === 'as-is'],
    ['первый кандидат', G.selectAt(C.DEADZONE_PX, 3).index === 0],
    ['второй кандидат', G.selectAt(C.DEADZONE_PX + C.ROW_H_PX, 3).index === 1],
    ['клэмп по длине стека', G.selectAt(C.DEADZONE_PX + 10 * C.ROW_H_PX, 3).index === 2],
    ['пустой стек — всегда как есть', G.selectAt(500, 0).zone === 'as-is'],
  ];
  for (const [name, ok] of cases) check('геометрия: ' + name, ok);
}

// ===== 3. Разбор команд ==============================================
const P = (text, ctx = null) => CR.parseCommand(text, ctx);
const one = (text, ctx = null) => { const p = P(text, ctx); return p.kind === 'one' ? p.hypothesis : null; };

{
  const h = one('добавь купить молоко');
  check('разбор: добавление без контекста', h && h.rule.command === 'addChild' && h.tail === 'купить молоко', h && h.tail);
}
{
  const h = one('добавь всё что нужно');
  check('разбор: глагол только в начале, фантома с хвоста нет',
    h && h.rule.command === 'addChild' && h.tail === 'все что нужно', h && h.tail);
}
{
  const h = one('нужно сделать отчёт');
  check('разбор: длинная форма бьёт короткую', h && h.n === 2 && h.tail === 'отчет', h);
}
{
  const h = one('сними фокус', '3');
  check('разбор: «сними фокус» — Open, а не «сними»', h && h.rule.status === 'Open' && h.n === 2, h && h.rule.status);
}
{
  check('разбор: «сначала …» без контекста — пусто', P('сначала купить краску').kind === 'empty');
  const h = one('сначала купить краску', '3');
  check('разбор: «сначала …» с контекстом — подзадача', h && h.rule.command === 'addChild', h);
}
{
  const a = one('перенеси в ремонт', '3'), b = one('перенеси ремонт', '3');
  check('разбор: предлог в остатке не влияет', a && b && a.tail === b.tail && a.tail === 'ремонт', [a && a.tail, b && b.tail]);
}
{
  const h = one('переименуй новый отчёт', '3');
  check('разбор: переименование', h && h.rule.command === 'editItem' && h.tail === 'новый отчет', h);
}
{
  check('разбор: «перенеси» и «переименуй» не сливаются',
    one('переименуй х', '3').rule.command === 'editItem' && one('перенеси х', '3').rule.command === 'setParent');
}
{
  check('разбор: безслотовая с остатком — пусто', P('готово молоко', '3').kind === 'empty');
  check('разбор: статус без контекста — пусто', P('готово').kind === 'empty');
  check('разбор: статус с контекстом', one('готово', '3').rule.status === 'Done');
}
{
  check('разбор: голое существительное командой не является', P('фокус', '3').kind === 'empty');
  check('разбор: «это в фокус» — статус', one('это в фокус', '3').rule.status === 'Focus');
  check('разбор: «это фокус» без предлога — статус', one('это фокус', '3').rule.status === 'Focus');
  check('разбор: «сделай это фокус» — статус', one('сделай это фокус', '3').rule.status === 'Focus');
  check('разбор: «сделай фокус» — статус', one('сделай фокус', '3').rule.status === 'Focus');
  check('разбор: «это сделано» — Done', one('это сделано', '3').rule.status === 'Done');
  check('разбор: «это пауза» — Pause', one('это пауза', '3').rule.status === 'Pause');
  check('разбор: «это архив» — Archive', one('это архив', '3').rule.status === 'Archive');
  check('разбор: «это открыто» — Open', one('это открыто', '3').rule.status === 'Open');
  check('разбор: «это информация» — Info', one('это информация', '3').rule.status === 'Info');
}
{
  check('разбор: «это на паузу»', one('это на паузу', '3').rule.status === 'Pause');
  check('разбор: «это на паузу купить молоко» — пусто', P('это на паузу купить молоко', '3').kind === 'empty');
  check('разбор: «удали» — архивирование', one('удали', '3').rule.status === 'Archive');
}
{
  const h = one('это в ремонт', '3');
  check('разбор: «это в …» — перенос', h && h.rule.command === 'setParent' && h.tail === 'ремонт', h);
}
{
  check('разбор: undo без контекста', one('отмени').rule.command === 'undo');
  check('разбор: undo с контекстом — пусто', P('отмени', '3').kind === 'empty');
}
{
  check('разбор: не команда — пусто', P('позвонить Ване', '3').kind === 'empty');
  check('разбор: одиночное «сделай» — пусто', P('сделай', '3').kind === 'empty');
}
{
  const h = one('добавьте купить молоко');
  check('разбор: морфология снимается стем-ярусом', h && h.rule.command === 'addChild', h);
}
{
  const h = one('перенесси ремонт', '3');
  check('разбор: опечатка глагола ловится', h && h.rule.command === 'setParent', h);
}

// ===== 4. Адаптер снимка =============================================
{
  const rows = A.adaptSnapshot(TASKS);
  const byId = Object.fromEntries(rows.map((r) => [r.id, r]));
  check('адаптер: line1 -> title', byId['3'].title === 'Установить смеситель');
  check('адаптер: Done -> closed', byId['11'].status === 'closed');
  check('адаптер: Archive -> superseded', byId['10'].status === 'superseded');
  check('адаптер: Pause остаётся рабочим', byId['6'].status === 'open');
}
{
  const r = runGesture({ phrase: 'найди старый список' });
  const labels = r.session.overlay.stack.map((s) => s.label);
  check('адаптер: архивные не попадают в кандидаты', !labels.includes('Старый список'), labels);
}

// ===== 5. Цели и защита от циклов ====================================
{
  const r = runGesture({ phrase: 'перенеси покупки', gesture: 'swipe', elementId: '3', dy: 0 });
  check('цель: единственная — под пальцем', isCmd(r, 'setParent', { parentId: '8' }), cmdOf(r));
}
{
  const r = runGesture({ phrase: 'перенеси смеситель', gesture: 'swipe', elementId: '5', dy: 0 });
  const n = r.session.overlay.stack.length;
  check('цель: неоднозначность даёт стек', n >= 1, { n, finger: r.session.overlay.finger.label });
}
{
  const r = runGesture({ phrase: 'перенеси ванная', gesture: 'swipe', elementId: '1', dy: 0 });
  check('цикл: потомок исключён из целей', r.out.action === 'cancel' && r.out.why === 'notFound', r.out);
}
{
  const r = runGesture({ phrase: 'перенеси несуществующее', gesture: 'swipe', elementId: '3', dy: 0 });
  check('цель: не найдено — исполнения нет', r.out.action === 'cancel', r.out);
}

// ===== 6. Отпускание в зонах =========================================
{
  const r = runGesture({ phrase: 'добавь купить молоко', dy: 0 });
  check('зона: без смещения — команда под пальцем', isCmd(r, 'addChild', { line1: 'купить молоко' }), cmdOf(r));
  check('зона: без контекста родитель — inbox', cmdOf(r).actId === 'inbox');
}
{
  const r = runGesture({ phrase: 'добавь купить молоко', gesture: 'swipe', elementId: '2', dy: 0 });
  check('зона: с контекстом родитель — контекст', cmdOf(r).actId === '2', cmdOf(r));
}
{
  const r = runGesture({ phrase: 'добавь купить смеситель', dy: 0 });
  const stack = r.session.overlay.stack;
  check('добавление: дубли уходят вверх', stack.length > 0 && stack.some((s) => s.label === 'Купить смеситель'), stack.map((s) => s.label));
}
{
  const r = runGesture({ phrase: 'добавь купить смеситель', dy: C.DEADZONE_PX });
  check('добавление: выбор дубля отменяет добавление', isCmd(r, 'viewItem'), cmdOf(r));
}
{
  const r = runGesture({ phrase: 'добавь купить молоко', dy: -C.CANCEL_ZONE_PX });
  check('зона: отмена ничего не исполняет', r.out.action === 'cancel', r.out);
}
{
  const r = runGesture({ phrase: 'позвонить Ване', dy: 0 });
  check('зона: нераспознанное уходит в фолбэк', r.out.action === 'fallback' && r.out.text === 'позвонить Ване', r.out);
}

// ===== 7. Отказы ASR =================================================
{
  const r = runGesture({ phrase: 'готово', failure: 'denied' });
  check('ASR: отказ микрофона — оверлей не взводится', !r.armed && r.denied, r);
  check('ASR: отказ микрофона — сообщение показано', r.messages.includes('Нет доступа к микрофону'), r.messages);
}
{
  const r = runGesture({ phrase: 'добавь молоко', failure: 'aborted', dy: 0 });
  check('ASR: обрыв — команда не исполняется', r.out.action === 'cancel', r.out);
  check('ASR: обрыв — сообщение показано', r.session.messages.includes('Распознавание прервано'), r.session.messages);
}
{
  const r = runGesture({ phrase: 'добавь молоко', failure: 'timeout', dy: 0 });
  check('ASR: таймаут финала — команда не исполняется', r.out.action === 'cancel', r.out);
  check('ASR: таймаут — сообщение показано', r.session.messages.includes('Не расслышал, повторите'), r.session.messages);
}
{
  const r = runGesture({ phrase: 'добавь купить молоко', interim: false, dy: 0 });
  check('ASR: без промежуточных — деградированный режим', r.out.action === 'fallback', r.out);
  check('ASR: деградированный режим объявлен', r.session.messages.includes('Выбор команды недоступен'), r.session.messages);
}

// ===== 8. Пересчёт и заморозка =======================================
{
  const r = runGesture({ phrase: 'добавь купить молоко', dy: 0 });
  check('пересчёт: по одному на слово', r.session.recomputes === 3, r.session.recomputes);
}
{
  const asr = new MockASR({ phrase: 'перенеси смеситель' });
  const s = new VoiceSession({ tasks: TASKS, asr });
  s.arm('5');
  asr.speak();
  s.move(C.FREEZE_DY_PX);
  const frozen = s.overlay.stack.map((x) => x.label);
  s.onInterim('перенеси смеситель установить');
  check('заморозка: порядок стека не меняется после сдвига',
    JSON.stringify(s.overlay.stack.map((x) => x.label)) === JSON.stringify(frozen), frozen);
}

// ===== 9. Актуальность снимка ========================================
{
  const stale = { command: 'setParent', actId: '3', payload: { parentId: 'нет-такого' } };
  check('снимок: команда с исчезнувшим id отвергается', !validate(stale, TASKS));
  check('снимок: inbox не требует наличия в снимке',
    validate({ command: 'addChild', actId: 'inbox', payload: {} }, TASKS));
}

// ===== 10. Непротиворечивость грамматики =============================
// Две формы одной длины, одной формы остатка и с одинаковым требованием
// контекста не должны сходиться по ярусам — иначе вернётся неоднозначность.
{
  const collisions = [];
  for (let i = 0; i < CR.RULES.length; i++) {
    for (let j = i + 1; j < CR.RULES.length; j++) {
      const a = CR.RULES[i], b = CR.RULES[j];
      if (a.len !== b.len || a.slot !== b.slot) continue;
      if (CR.sameCommand(a, b)) continue;
      if ((a.ctx === 'required' && b.ctx === 'forbidden') ||
          (a.ctx === 'forbidden' && b.ctx === 'required')) continue;
      if (CR.matchForm(a.tokens, b) !== null) collisions.push([a.form, b.form]);
    }
  }
  check('грамматика: пересекающихся форм нет', collisions.length === 0, collisions);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
