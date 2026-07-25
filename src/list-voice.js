const STATUS_PHRASES = [
  { status: 'Done', words: ['выполнено', 'выполнена', 'выполнен', 'готово', 'готова', 'сделано', 'сделана', 'закрыто', 'закрыта'] },
  { status: 'Focus', words: ['фокус', 'фокусе', 'важно', 'главное'] },
  { status: 'Pause', words: ['пауза', 'паузу', 'паузе', 'ожидание', 'отложено', 'отложена'] },
  { status: 'Archive', words: ['архив', 'архиве', 'архивировать'] },
  { status: 'Open', words: ['открыто', 'открыта', 'заново', 'вернуть', 'активно', 'активна'] }
];

const TAG_WORDS = ['важное', 'срочно', 'купить', 'дом', 'работа', 'отложить'];

const TAG_CANONICAL = {
  'важное': 'Важное',
  'срочно': 'Срочно',
  'купить': 'Купить',
  'дом': 'Дом',
  'работа': 'Работа',
  'отложить': 'Отложить'
};

const STOP_WORDS = new Set(['задачу', 'задача', 'задачи', 'пункт', 'пункта', 'элемент', 'дело', 'на', 'в', 'во', 'к', 'ко', 'для', 'с', 'со', 'из', 'по', 'и']);

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

export function normalizeSpeech(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/[.,!?;:«»"'()\-–—]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function stem(token) {
  // Very small Russian suffix trim: enough to match "яблоки" against "яблоко".
  return token.length > 4 ? token.slice(0, token.length - 1) : token;
}

function tokenize(value) {
  return normalizeSpeech(value).split(' ').filter(Boolean);
}

function meaningfulTokens(value) {
  const tokens = tokenize(value).filter((token) => !STOP_WORDS.has(token));
  return tokens.length ? tokens : tokenize(value);
}

function scoreItem(item, queryTokens) {
  const titleNormalized = normalizeSpeech(item.line1);
  const queryNormalized = queryTokens.join(' ');
  if (!queryNormalized) return 0;
  if (titleNormalized === queryNormalized) return 1;

  const titleTokens = tokenize(item.line1);
  if (!titleTokens.length) return 0;

  let hits = 0;
  for (const queryToken of queryTokens) {
    const queryStem = stem(queryToken);
    const matched = titleTokens.some((titleToken) => {
      const titleStem = stem(titleToken);
      return titleToken === queryToken || titleStem === queryStem ||
        titleToken.startsWith(queryStem) || queryToken.startsWith(titleStem);
    });
    if (matched) hits += 1;
  }

  if (!hits) return 0;
  const precision = hits / queryTokens.length;
  const recall = hits / titleTokens.length;
  return (precision * 0.75 + recall * 0.25) * 0.95;
}

export function findItemByPhrase(items, phrase) {
  const queryTokens = meaningfulTokens(phrase);
  if (!queryTokens.length) return null;

  let best = null;
  let bestScore = 0;

  for (const item of items || []) {
    const score = scoreItem(item, queryTokens);
    if (score > bestScore) {
      bestScore = score;
      best = item;
    }
  }

  return bestScore >= 0.5 ? best : null;
}

function statusFromPhrase(phrase) {
  const tokens = tokenize(phrase);
  for (const entry of STATUS_PHRASES) {
    if (tokens.some((token) => entry.words.includes(token))) return entry.status;
  }
  return null;
}

function tagFromPhrase(phrase) {
  const tokens = tokenize(phrase);
  for (const token of tokens) {
    if (TAG_WORDS.includes(token)) return TAG_CANONICAL[token];
  }
  return null;
}

function stripLeadingStopWords(phrase) {
  const tokens = tokenize(phrase);
  while (tokens.length && STOP_WORDS.has(tokens[0])) tokens.shift();
  return tokens.join(' ');
}

function splitDetails(phrase) {
  const match = phrase.match(/^(.*?)\s+(?:с подписью|подпись|описание|заметка)\s+(.+)$/);
  if (!match) return { line1: phrase.trim(), line2: '' };
  return { line1: match[1].trim(), line2: match[2].trim() };
}

function notFound(phrase) {
  return { ok: false, reason: 'not-found', feedback: `Не нашёл задачу «${phrase}»` };
}

const RULES = [
  {
    name: 'showActionLog',
    patterns: [/^(?:покажи|открой|показать)?\s*журнал(?:\s+действий)?$/],
    build: () => ({ input: { actId: 'log', actType: 'tab', command: 'showActionLog', payload: {} }, feedback: 'Открываю журнал' })
  },
  {
    name: 'showFrontier',
    patterns: [/^(?:покажи|открой|показать)?\s*фронтир(?:\s+задач)?$/],
    build: () => ({ input: { actId: 'frontier', actType: 'tab', command: 'showFrontier', payload: {} }, feedback: 'Открываю фронтир' })
  },
  {
    name: 'showList',
    patterns: [/^(?:покажи|открой|показать|вернись\s+в|назад\s+в)?\s*список$/],
    build: () => ({ input: { actId: 'list', actType: 'tab', command: 'showList', payload: {} }, feedback: 'Открываю список' })
  },
  {
    name: 'undo',
    patterns: [/^(?:отмена|отмени|отменить|верни\s+как\s+было|шаг\s+назад)$/],
    build: () => ({ input: { actId: 'list', actType: 'list', command: 'undo', payload: {} }, feedback: 'Отменяю последнее действие' })
  },
  {
    name: 'addChild',
    patterns: [
      /^(?:добавь|добавить|создай|создать)\s+подзадачу\s+(?<child>.+?)\s+(?:к|в|для)\s+(?<target>.+)$/,
      /^(?:добавь|добавить|создай|создать)\s+(?:к|в|для)\s+(?<target>.+?)\s+подзадачу\s+(?<child>.+)$/
    ],
    build: (groups, state) => {
      const target = findItemByPhrase(state.snapshot.items, groups.target);
      if (!target) return notFound(groups.target.trim());
      const details = splitDetails(stripLeadingStopWords(groups.child));
      if (!details.line1) return { ok: false, reason: 'empty', feedback: 'Не расслышал название подзадачи' };
      return {
        input: { actId: target.id, actType: 'item', command: 'addChild', payload: details },
        feedback: `Добавил подзадачу «${details.line1}» в «${target.line1}»`
      };
    }
  },
  {
    name: 'addItem',
    patterns: [
      /^(?:добавь|добавить|создай|создать|запиши|записать)\s+(?:новую\s+)?(?:задачу|пункт|дело|элемент)?\s*(?<title>.+)$/,
      /^(?:новая|новую)\s+задача?у?\s+(?<title>.+)$/
    ],
    build: (groups) => {
      const details = splitDetails(stripLeadingStopWords(groups.title));
      if (!details.line1) return { ok: false, reason: 'empty', feedback: 'Не расслышал название задачи' };
      return {
        input: { actId: 'list', actType: 'list', command: 'addItem', payload: details },
        feedback: `Добавил задачу «${details.line1}»`
      };
    }
  },
  {
    name: 'editItem',
    patterns: [/^(?:переименуй|переименовать|измени|изменить)\s+(?<target>.+?)\s+(?:в|на)\s+(?<title>.+)$/],
    build: (groups, state) => {
      const target = findItemByPhrase(state.snapshot.items, groups.target);
      if (!target) return notFound(groups.target.trim());
      const details = splitDetails(groups.title.trim());
      if (!details.line1) return { ok: false, reason: 'empty', feedback: 'Не расслышал новое название' };
      return {
        input: { actId: target.id, actType: 'item', command: 'editItem', payload: details },
        feedback: `Переименовал в «${details.line1}»`
      };
    }
  },
  {
    name: 'deleteItem',
    patterns: [/^(?:удали|удалить|убери|убрать|сотри)\s+(?<target>.+)$/],
    build: (groups, state) => {
      const target = findItemByPhrase(state.snapshot.items, groups.target);
      if (!target) return notFound(groups.target.trim());
      return {
        input: { actId: target.id, actType: 'item', command: 'deleteItem', payload: {} },
        feedback: `Удалил «${target.line1}»`
      };
    }
  },
  {
    name: 'setTags',
    patterns: [
      /^(?:тег|тэг|метка|пометь\s+тегом)\s+(?<tag>.+?)\s+(?:для|у|на)\s+(?<target>.+)$/,
      /^(?:пометь|отметь)\s+(?<target>.+?)\s+(?:как|тегом)\s+(?<tag>.+)$/
    ],
    build: (groups, state) => {
      const tag = tagFromPhrase(groups.tag);
      if (!tag) return { ok: false, reason: 'unknown-tag', feedback: `Нет тега «${groups.tag.trim()}»` };
      const target = findItemByPhrase(state.snapshot.items, groups.target);
      if (!target) return notFound(groups.target.trim());
      return {
        input: { actId: target.id, actType: 'item', command: 'setTags', payload: { tag } },
        feedback: `Переключил тег «${tag}» у «${target.line1}»`
      };
    }
  },
  {
    name: 'toggleCollapse',
    patterns: [/^(?:сверни|свернуть|разверни|развернуть)\s+(?<target>.+)$/],
    build: (groups, state) => {
      const target = findItemByPhrase(state.snapshot.items, groups.target);
      if (!target) return notFound(groups.target.trim());
      return {
        input: { actId: target.id, actType: 'item', command: 'toggleCollapse', payload: {} },
        feedback: `Переключил сворачивание «${target.line1}»`
      };
    }
  },
  {
    name: 'setStatusSuffix',
    patterns: [
      /^(?:отметь|отметить|пометь|поставь|переведи|сделай)\s+(?<target>.+?)\s+(?:как\s+|в\s+|на\s+)?(?<status>[^\s]+)$/,
      /^(?<target>.+?)\s+(?:готово|выполнено|сделано)$/
    ],
    build: (groups, state) => {
      const status = statusFromPhrase(groups.status || 'выполнено');
      if (!status) return { ok: false, reason: 'unknown-status', feedback: `Не знаю статус «${(groups.status || '').trim()}»` };
      const target = findItemByPhrase(state.snapshot.items, groups.target);
      if (!target) return notFound(groups.target.trim());
      return {
        input: { actId: target.id, actType: 'item', command: 'setStatus', payload: { status } },
        feedback: `«${target.line1}» → ${status}`
      };
    }
  },
  {
    name: 'setStatusPrefix',
    patterns: [
      /^(?:фокус|фокусируйся)\s+(?:на\s+)?(?<target>.+)$/,
      /^(?:в\s+)?(?<status>архив|паузу|пауза|фокус)\s+(?<target>.+)$/,
      /^(?:архивируй|архивировать)\s+(?<target>.+)$/,
      /^(?:верни|вернуть|открой\s+заново|активируй)\s+(?<target>.+)$/
    ],
    build: (groups, state, matchedPattern) => {
      const status = groups.status
        ? statusFromPhrase(groups.status)
        : (/фокус/.test(matchedPattern.source) ? 'Focus' :
          /архивируй/.test(matchedPattern.source) ? 'Archive' : 'Open');
      if (!status) return { ok: false, reason: 'unknown-status', feedback: 'Не понял статус' };
      const target = findItemByPhrase(state.snapshot.items, groups.target);
      if (!target) return notFound(groups.target.trim());
      return {
        input: { actId: target.id, actType: 'item', command: 'setStatus', payload: { status } },
        feedback: `«${target.line1}» → ${status}`
      };
    }
  }
];

export function createVoiceCommandParser() {
  return {
    parse(transcript, state) {
      const phrase = normalizeSpeech(transcript);
      if (!phrase) return { ok: false, reason: 'empty', feedback: 'Ничего не расслышал' };

      const safeState = state && state.snapshot ? state : { snapshot: { items: [] } };

      for (const rule of RULES) {
        for (const pattern of rule.patterns) {
          const match = phrase.match(pattern);
          if (!match) continue;

          const built = rule.build(match.groups || {}, safeState, pattern);
          if (built.ok === false) return { ...built, transcript: phrase, rule: rule.name };
          return { ok: true, transcript: phrase, rule: rule.name, ...built };
        }
      }

      return { ok: false, reason: 'unknown', feedback: `Не понял команду: «${phrase}»`, transcript: phrase };
    }
  };
}

export function isSpeechRecognitionSupported(scope = globalThis) {
  return Boolean(scope && (scope.SpeechRecognition || scope.webkitSpeechRecognition));
}

export function createRecognitionFactory(scope = globalThis) {
  const Recognition = scope.SpeechRecognition || scope.webkitSpeechRecognition;
  if (!Recognition) return null;
  return () => new Recognition();
}

export function createVoiceController({
  recognitionFactory,
  parser = createVoiceCommandParser(),
  dispatch,
  getState,
  onStatus = () => {},
  lang = 'ru-RU'
}) {
  let recognition = null;
  let listening = false;
  let undoSnapshot = null;

  function emit(status, extra = {}) {
    onStatus({ status, listening, ...extra });
  }

  function handleTranscript(rawTranscript) {
    const state = getState();
    const result = parser.parse(rawTranscript, state);

    if (!result.ok) {
      emit('rejected', { transcript: result.transcript || normalizeSpeech(rawTranscript), message: result.feedback, reason: result.reason });
      return result;
    }

    const input = { ...result.input, source: 'voice' };

    if (input.command === 'undo') {
      if (!undoSnapshot) {
        const message = 'Нечего отменять';
        emit('rejected', { transcript: result.transcript, message, reason: 'no-undo' });
        return { ok: false, reason: 'no-undo', feedback: message, transcript: result.transcript };
      }
      input.payload = { snapshot: clone(undoSnapshot) };
      undoSnapshot = null;
    } else if (!String(input.command).startsWith('show')) {
      undoSnapshot = state?.snapshot ? clone(state.snapshot) : null;
    }

    dispatch(input);
    emit('accepted', { transcript: result.transcript, message: result.feedback, command: input.command });
    return { ...result, input };
  }

  function stop() {
    if (recognition) {
      try {
        recognition.stop();
      } catch {
        // Recognition may already be stopping; nothing to recover.
      }
    }
    listening = false;
    emit('idle');
  }

  function start() {
    if (listening) return false;
    if (!recognitionFactory) {
      emit('unsupported', { message: 'Голосовой ввод не поддерживается этим браузером' });
      return false;
    }

    recognition = recognitionFactory();
    recognition.lang = lang;
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      const results = Array.from(event.results || []);
      const last = results[results.length - 1];
      if (!last) return;
      const transcript = last[0]?.transcript || '';
      if (!last.isFinal) {
        emit('interim', { transcript: normalizeSpeech(transcript) });
        return;
      }
      handleTranscript(transcript);
    };

    recognition.onerror = (event) => {
      listening = false;
      const message = event?.error === 'not-allowed'
        ? 'Нет доступа к микрофону'
        : `Ошибка распознавания: ${event?.error || 'unknown'}`;
      emit('error', { message, reason: event?.error });
    };

    recognition.onend = () => {
      listening = false;
      emit('idle');
    };

    recognition.start();
    listening = true;
    emit('listening', { message: 'Слушаю…' });
    return true;
  }

  return {
    handleTranscript,
    isListening: () => listening,
    start,
    stop,
    toggle() {
      return listening ? (stop(), false) : start();
    }
  };
}

export function createVoiceUI({ button, statusEl, transcriptEl, controller, supported = true }) {
  if (!button) return { destroy() {} };

  if (!supported) {
    button.disabled = true;
    button.title = 'Голосовой ввод не поддерживается этим браузером';
    if (statusEl) statusEl.textContent = 'Голос недоступен в этом браузере';
    return { destroy() {} };
  }

  function onClick() {
    controller.toggle();
  }

  button.addEventListener('click', onClick);

  return {
    onStatus({ status, transcript, message }) {
      button.classList.toggle('listening', status === 'listening');
      button.setAttribute('aria-pressed', String(status === 'listening'));

      if (transcriptEl && typeof transcript === 'string') transcriptEl.textContent = transcript;
      if (!statusEl) return;

      statusEl.dataset.state = status;
      if (message) statusEl.textContent = message;
      else if (status === 'idle') statusEl.textContent = 'Нажмите микрофон и скажите команду';
    },
    destroy() {
      button.removeEventListener('click', onClick);
    }
  };
}
