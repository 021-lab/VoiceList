// command-resolver.js — перевод транскрипции в команду.
// Реализация docs/COMMAND_RESOLVER.md. Зависит только от resolver.js.

'use strict';

import * as R from './resolver.js';

// ===== Параметры Θ команд =============================================
// Отдельные от Θ целей: калибруются на своём корпусе.
const STEM_PREFIX = 4;
const STEM_MAX_DELTA = 3;   // морфологический хвост короткий: «добавьте» × «добавь» — 2,
                            // а «фокус» × «фокусируй» — 5, и это уже другое слово
const TYPO_DIST = 1;
const TYPO_MINLEN = 6;      // строже, чем у целей: на пятибуквенных глаголах
                            // одна правка превращает «купи» в «купе»
const W_EXACT = 3;
const W_STEM = 2;
const W_TYPO = 1;

// Короткие служебные слова сопоставляются только точным ярусом:
// стем и опечатка на них дают ложные срабатывания.
const EXACT_ONLY = new Set(['это', 'в', 'во', 'на', 'к', 'ко', 'тут', 'мне', 'как', 'где']);

// Предлоги, снимаемые с начала остатка.
const LEAD_PREP = new Set(['в', 'во', 'на', 'к', 'ко']);

// ===== Таблица фреймов ================================================
// slot: 'required' — остаток обязателен; 'none' — остаток обязан быть пуст.
// ctx:  'required' | 'forbidden' | 'optional' | 'ignored'
// target: true — остаток разрешается в задачу через resolver.resolve

const FRAMES = [
  // --- со слотом ---
  { command: 'addChild', slot: 'required', ctx: 'optional', target: false,
    forms: ['добавь', 'добавить', 'запиши', 'записать', 'создай', 'создать',
            'новая', 'новый', 'новое', 'нужно', 'надо',
            'мне нужно', 'нужно сделать', 'надо сделать'] },

  { command: 'addChild', slot: 'required', ctx: 'required', target: false,
    forms: ['добавь подзадачу', 'добавь блокер', 'сначала', 'сначала нужно',
            'сначала надо', 'сначала сделать', 'тут сначала', 'тут сначала нужно'] },

  { command: 'setParent', slot: 'required', ctx: 'required', target: true,
    forms: ['перенеси', 'перенести', 'перемести', 'переместить', 'вложи',
            'сделай подзадачей', 'это в', 'это'] },

  { command: 'editItem', slot: 'required', ctx: 'required', target: false,
    forms: ['переименуй', 'переименовать', 'назови', 'назвать',
            'исправь', 'исправить'] },

  { command: 'showSearch', slot: 'required', ctx: 'ignored', target: true,
    forms: ['найди', 'найти', 'ищи', 'искать', 'покажи', 'показать', 'поиск', 'где'] },

  // --- без слота ---
  { command: 'setStatus', status: 'Done', slot: 'none', ctx: 'required', target: false,
    forms: ['готово', 'готова', 'сделано', 'выполнено', 'закрой', 'закрыть',
            'заверши', 'завершить', 'это готово', 'это сделано'] },

  { command: 'setStatus', status: 'Focus', slot: 'none', ctx: 'required', target: false,
    forms: ['сфокусируй', 'фокусируй', 'это в фокус', 'это фокус', 'сделай в фокус', 'сделай фокус', 'сделай это фокус'] },

  { command: 'setStatus', status: 'Pause', slot: 'none', ctx: 'required', target: false,
    forms: ['отложи', 'отложить', 'это на паузу', 'это паузу', 'это пауза', 'это потом'] },

  { command: 'setStatus', status: 'Archive', slot: 'none', ctx: 'required', target: false,
    forms: ['удали', 'удалить', 'убери', 'убрать', 'заархивируй',
            'это в архив', 'это архив'] },

  { command: 'setStatus', status: 'Open', slot: 'none', ctx: 'required', target: false,
    forms: ['сними', 'сними фокус', 'сними паузу', 'возобнови',
            'это в работу', 'это работу', 'это открыто'] },

  { command: 'setStatus', status: 'Info', slot: 'none', ctx: 'required', target: false,
    forms: ['это информация', 'это инфо'] },

  { command: 'undo', slot: 'none', ctx: 'forbidden', target: false,
    forms: ['отмени', 'отменить', 'откати', 'откатить', 'верни как было'] },
];

// Развёрнутая таблица: одна запись на форму.
const RULES = [];
for (const f of FRAMES) {
  for (const form of f.forms) {
    const tokens = form.split(' ');
    RULES.push({ ...f, form, tokens, len: tokens.length });
  }
}
const MAX_FORM_LEN = RULES.reduce((m, r) => Math.max(m, r.len), 1);

// ===== Ярусы ==========================================================
function tier(spoken, formToken) {
  if (spoken === formToken) return W_EXACT;
  if (EXACT_ONLY.has(formToken)) return 0;
  const minLen = Math.min(spoken.length, formToken.length);
  let c = 0;
  while (c < minLen && spoken[c] === formToken[c]) c++;
  // Стем: короткое слово целиком является началом длинного, и разница длин
  // не больше окончания. «сделай» × «сделано» расходятся на пятой букве —
  // это разные слова, а не одна форма.
  if (c >= STEM_PREFIX && c === minLen &&
      Math.abs(spoken.length - formToken.length) <= STEM_MAX_DELTA) return W_STEM;
  if (TYPO_DIST > 0 && spoken.length >= TYPO_MINLEN && formToken.length >= TYPO_MINLEN &&
      R.damerauLevenshtein(spoken, formToken) <= TYPO_DIST) return W_TYPO;
  return 0;
}

function matchForm(tokens, rule) {
  let w = 0;
  for (let i = 0; i < rule.len; i++) {
    const t = tier(tokens[i], rule.tokens[i]);
    if (t === 0) return null;
    w += t;
  }
  return w;
}

function stripLeadPrep(tokens) {
  return tokens.length && LEAD_PREP.has(tokens[0]) ? tokens.slice(1) : tokens;
}

function ctxOk(rule, context) {
  if (rule.ctx === 'required') return context != null;
  if (rule.ctx === 'forbidden') return context == null;
  return true;
}

function sameCommand(a, b) {
  return a.command === b.command && (a.status ?? null) === (b.status ?? null);
}

// ===== Разбор =========================================================
// parseCommand(text, context) -> {kind:'one'|'vector'|'empty', ...}
function parseCommand(text, context) {
  const T = R.tokenize(R.normalize(text));
  if (T.length === 0) return { kind: 'empty', why: 'no-input' };

  for (let len = Math.min(MAX_FORM_LEN, T.length); len >= 1; len--) {
    const H = [];
    let matchedAtThisLen = false;
    for (const rule of RULES) {
      if (rule.len !== len) continue;
      const w = matchForm(T, rule);
      if (w === null) continue;
      matchedAtThisLen = true;   // форма опознана — короче уже не смотрим
      const tail = stripLeadPrep(T.slice(len));
      if (rule.slot === 'required' && tail.length === 0) continue;
      if (rule.slot === 'none' && tail.length > 0) continue;
      if (!ctxOk(rule, context)) continue;
      H.push({ rule, n: len, w, tail: tail.join(' ') });
    }
    // Самая длинная форма выигрывает: если форма этой длины опознана, но
    // отвергнута по остатку или контексту — к более коротким не спускаемся.
    if (H.length === 0) { if (matchedAtThisLen) return { kind: 'empty', why: 'shape' }; continue; }

    H.sort((a, b) => b.n - a.n || b.w - a.w);
    const best = H[0], second = H[1];
    const tie = !!second && best.n === second.n && best.w === second.w &&
                !sameCommand(best.rule, second.rule);
    // инвариант: ничья никогда не разрешается молча
    if (tie) return { kind: 'vector', why: 'tie', hypotheses: H };
    return { kind: 'one', hypothesis: best };
  }
  return { kind: 'empty', why: 'no-verb' };
}

// Гипотеза -> команда исполнителя (цель ещё не разрешена)
function toCommand(h, context) {
  const r = h.rule;
  switch (r.command) {
    case 'addChild':
      return { command: 'addChild', actId: context ?? 'inbox', payload: { line1: h.tail } };
    case 'editItem':
      return { command: 'editItem', actId: context, payload: { line1: h.tail } };
    case 'setParent':
      return { command: 'setParent', actId: context, payload: { parentId: null } };
    case 'showSearch':
      return { command: 'showSearch', actId: null, payload: { query: h.tail } };
    case 'setStatus':
      return { command: 'setStatus', actId: context, payload: { status: r.status } };
    case 'undo':
      return { command: 'undo', actId: null, payload: {} };
    default:
      return null;
  }
}

const THETA = { STEM_PREFIX, STEM_MAX_DELTA, TYPO_DIST, TYPO_MINLEN,
                W_EXACT, W_STEM, W_TYPO };

export { FRAMES, RULES, MAX_FORM_LEN, tier, matchForm, parseCommand, toCommand,
         stripLeadPrep, sameCommand, THETA };
