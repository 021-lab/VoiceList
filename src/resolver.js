// resolver.js — поиск задачи по нечёткому голосовому вводу
// Реализация docs/TASK_RESOLVER.md. Чистый JS, без зависимостей.
// Контракт: resolve(text, rows) -> {kind:'one'|'vector'|'empty', ...}

'use strict';

// ===== Параметры Θ =====================================================
// Вся настройка точности — здесь. Общее правило: чем строже параметры,
// тем реже молчаливая запись и тем чаще оверлей выбора. Ужесточение
// любого параметра может только сузить множество молчаливых записей
// (свойство монотонности, см. TASK_RESOLVER.md §2.4).

// --- Секция 1. Что считается совпадением двух слов ---------------------
// Управляет «шириной» каждого яруса: насколько непохожие слова ещё
// склеиваются. Расширяешь — ловится больше морфологии и ошибок ASR,
// но растёт риск склеить разные слова.

const STEM_PREFIX = 4;   // Стем: минимальная длина общего начала слов.
                         // 4 склеивает «ванная»/«ванную» (общее «ванн»),
                         // но не «сон»/«сони». Меньше — агрессивнее
                         // склейка коротких слов, больше ложных пар.

const STEM_NUM = 2;      // Стем: общее начало должно покрывать долю
const STEM_DEN = 3;      // NUM/DEN короткого слова. При 2/3:
                         // «свет»/«света» склеятся (общих 4 из 5 букв),
                         // «переход»/«переезд» — нет (общих 4, а нужно
                         // 5 из 7): доля отсекает случайно общие начала
                         // у длинных слов. Больше доля — строже.

const TYPO_DIST = 1;     // Опечатки: максимум правок (вставка/удаление/
                         // замена/перестановка) между словами.
                         // 1 ловит типичную ошибку распознавания речи
                         // («сместитель»). 2 — заметно шире и опаснее.
                         // 0 — ярус опечаток выключен совсем.

const TYPO_MINLEN = 5;   // Опечатки: слова короче не сравниваются по
                         // правкам — на коротких словах одна правка
                         // превращает слово в другое слово («кот»/«код»).

// --- Секция 2. Ценность ярусов ------------------------------------------
// Веса складываются в счёт кандидата s и решают ничьи между кандидатами
// с равным числом совпавших слов. Обязателен порядок:
// W_EXACT > W_STEM > W_TYPO > 0 — точное совпадение всегда ценнее
// морфологии, морфология ценнее опечатки.

const W_EXACT = 3;       // слово совпало буква в букву
const W_STEM = 2;        // совпало по основе («купи» ~ «купить»)
const W_TYPO = 1;        // совпало с одной опечаткой

// --- Секция 3. Когда писать молча, а когда показывать оверлей ----------
// Правила решения над лучшим кандидатом (best) и вторым (second).
// Ничья best/second по (m, s) ВСЕГДА даёт оверлей — это инвариант,
// параметрами не отключается.

const MAX_MISS = 0;      // R2: сколько слов запроса может не найтись,
                         // чтобы лидер всё ещё считался уверенным.
                         // 0: «купи смеситель» должен найти оба слова.
                         // 1: «большой синий шкаф» уверенно возьмёт
                         // «Синий шкаф». Больше — либеральнее.

const COVER_NUM = 1;     // R3: минимум найденных слов запроса как доля
const COVER_DEN = 2;     // NUM/DEN. 1/2: из четырёх слов должны
                         // совпасть два. Выше доля — строже к длинным
                         // многословным запросам.

const LEAD = 1;          // R3: отрыв лидера от второго в ЦЕЛЫХ словах.
                         // 1: лидер нашёл хотя бы на слово больше.
                         // 2: нужен отрыв в два слова — почти всегда
                         // оверлей при похожих кандидатах.

// --- Секция 4. Габариты, не точность ------------------------------------
// На решения не влияют — ограничивают размеры выдачи и защищают от
// вырожденных данных.

const CAND_LIMIT = 7;    // сколько кандидатов показывает оверлей
const FOUND_CAP = 200;   // потолок найденного множества из поиска;
                         // при срабатывании — предупреждение в лог
const PART_MIN = 2;      // куски слова короче не становятся поисковыми
                         // паттернами (защита от '%ль%' по всей базе)

// ===== Нормализация (§2.1) ============================================
function normalize(s) {
  return s.toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim().replace(/\s+/g, ' ');
}
function tokenize(s) {
  return normalize(s).split(' ').filter(Boolean);
}

// ===== Преобразование запроса в паттерны (§4.1) ========================
// Recall-superset: точный/стем ловятся префиксом-4, опечатка DL<=1 —
// одной из трёх частей (правка локальна, целой остаётся хотя бы одна).
function queryPatterns(text) {
  const pats = new Set();
  for (const q of tokenize(text)) {
    if (q.length < STEM_PREFIX) { pats.add(q); continue; }
    pats.add(q.slice(0, STEM_PREFIX));
    if (q.length >= TYPO_MINLEN && TYPO_DIST > 0) {
      const a = Math.ceil(q.length / 3), b = Math.ceil((2 * q.length) / 3);
      for (const part of [q.slice(0, a), q.slice(a, b), q.slice(b)])
        if (part.length >= PART_MIN) pats.add(part);
    }
  }
  return [...pats];
}

// SQL-вариант отбора (§4.2) — если таблица в SQLite
function buildQuery(text) {
  const pats = queryPatterns(text);
  const like = pats.map(() => "title_norm LIKE '%' || ? || '%'").join(' OR ');
  return {
    sql: `SELECT id, title, title_norm, status FROM tasks
          WHERE status NOT IN ('closed','superseded')
            AND (${like}) LIMIT ${FOUND_CAP}`,
    params: pats,
  };
}

// JS-вариант отбора — если задачи уже в памяти из поллинга GET /tasks
function findCandidates(text, tasks) {
  const pats = queryPatterns(text);
  const found = [];
  for (const t of tasks) {
    if (t.status === 'closed' || t.status === 'superseded') continue;
    const tn = t.title_norm ?? normalize(t.title);
    if (pats.some((p) => tn.includes(p))) {
      found.push({ ...t, title_norm: tn });
      if (found.length >= FOUND_CAP) {
        console.warn('resolver: FOUND_CAP reached');
        break;
      }
    }
  }
  return found;
}

// ===== Ярусы (§2.2) ====================================================
// Дамерау–Левенштейн (OSA; для порога 1 совпадает с полным DL)
function damerauLevenshtein(a, b) {
  const n = a.length, m = b.length;
  if (Math.abs(n - m) > TYPO_DIST + 1) return TYPO_DIST + 1; // быстрый выход
  let prev2 = null;
  let prev = Array.from({ length: m + 1 }, (_, j) => j);
  for (let i = 1; i <= n; i++) {
    const cur = [i];
    for (let j = 1; j <= m; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1])
        v = Math.min(v, prev2[j - 2] + 1); // транспозиция
      cur.push(v);
    }
    prev2 = prev; prev = cur;
  }
  return prev[m];
}

function pairTier(q, t) {
  if (q === t) return W_EXACT;
  const minLen = Math.min(q.length, t.length);
  let c = 0;
  while (c < minLen && q[c] === t[c]) c++;
  if (c >= STEM_PREFIX && c * STEM_DEN >= STEM_NUM * minLen) return W_STEM;
  if (TYPO_DIST > 0 && q.length >= TYPO_MINLEN && t.length >= TYPO_MINLEN &&
      damerauLevenshtein(q, t) <= TYPO_DIST) return W_TYPO;
  return 0;
}

// ===== Подпись кандидата (§2.3): жадное паросочетание ==================
function signature(Q, titleTokens) {
  const pairs = [];
  Q.forEach((q, qi) => titleTokens.forEach((t, ti) => {
    const w = pairTier(q, t);
    if (w > 0) pairs.push({ qi, ti, w });
  }));
  pairs.sort((x, y) => y.w - x.w);
  const usedQ = new Set(), usedT = new Set();
  let m = 0, s = 0;
  for (const p of pairs) {
    if (usedQ.has(p.qi) || usedT.has(p.ti)) continue;
    usedQ.add(p.qi); usedT.add(p.ti);
    m += 1; s += p.w;
  }
  return { m, s, t: usedT.size };
}

// ===== Решение (§4.3, инварианты §3) ===================================
// rows: найденное множество [{id, title, title_norm?, status}]
function resolve(text, rows) {
  const qn = normalize(text);
  const Q = tokenize(text);
  if (Q.length === 0) return { kind: 'empty' };

  const withNorm = rows.map((r) => ({ ...r, title_norm: r.title_norm ?? normalize(r.title) }));

  // R1 / инвариант 1: точное равенство названия
  const exact = withNorm.filter((r) => r.title_norm === qn);
  if (exact.length === 1) return { kind: 'one', id: exact[0].id, why: 'R1' };
  if (exact.length > 1)
    return { kind: 'vector', why: 'R1-dup',
             candidates: exact.slice(0, CAND_LIMIT).map((r) => ({
               id: r.id, title: r.title,
               sig: { m: Q.length, s: Q.length * W_EXACT, t: Q.length } })) };

  // подписи над найденным множеством
  const cands = withNorm
    .map((r) => ({ id: r.id, title: r.title, sig: signature(Q, tokenize(r.title_norm)) }))
    .filter((c) => c.sig.m > 0);           // паттерн зацепил, ярусы — нет
  if (cands.length === 0) return { kind: 'empty' };  // инвариант 3

  cands.sort((a, b) =>
    b.sig.m - a.sig.m || b.sig.s - a.sig.s || b.sig.t - a.sig.t);
  const best = cands[0], second = cands[1];
  const tie = !!second && best.sig.m === second.sig.m && best.sig.s === second.sig.s;

  // инвариант 2: ничья никогда не разрешается молча
  if (!tie) {
    // R2: почти все токены запроса нашлись, лидер уникален по (m, s)
    if (best.sig.m >= Q.length - MAX_MISS)
      return { kind: 'one', id: best.id, why: 'R2' };
    // R3: отрыв в целых токенах + минимальное покрытие запроса
    if ((!second || best.sig.m - second.sig.m >= LEAD) &&
        best.sig.m * COVER_DEN >= Q.length * COVER_NUM)
      return { kind: 'one', id: best.id, why: 'R3' };
  }
  return { kind: 'vector', why: tie ? 'tie' : 'doubt',
           candidates: cands.slice(0, CAND_LIMIT) };
}

// Полный цикл: текст + все задачи (из поллинга) -> исход
function resolveOverTasks(text, tasks) {
  return resolve(text, findCandidates(text, tasks));
}

export { normalize, tokenize, queryPatterns, buildQuery, findCandidates,
         damerauLevenshtein, pairTier, signature, resolve, resolveOverTasks,
         CAND_LIMIT, FOUND_CAP };

const api = { normalize, tokenize, queryPatterns, buildQuery, findCandidates,
              damerauLevenshtein, pairTier, signature, resolve, resolveOverTasks };
export default api;
if (typeof window !== 'undefined') window.resolver = api;
