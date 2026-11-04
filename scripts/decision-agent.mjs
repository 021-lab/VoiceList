// The text agent built from two models that do not write text: Clef answers typed questions
// about the request (which actions, which status, which task, where a new task goes) with
// probabilities, Clairic pulls typed values out of it (titles, a date, what is left over). Code
// turns the answers into the same commands the chat models produce. For the benchmark only:
// scripts/eval-text-agent.mjs --provider decision --model clef+clairic.

const WEEKDAYS = ['воскресенье', 'понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота'];
const YES = 0.5;

function taskLine(task, byId) {
  const parent = task.parentId ? byId.get(task.parentId) : null;
  return `«${task.title}»${parent ? ` в «${parent.title}»` : ' в корне'}, статус ${task.status}`;
}

/** What both models read: the request, the task in hand, the date and the list. */
export function buildState(context) {
  const byId = new Map(context.tasks.map(task => [task.id, task]));
  const held = context.target ? byId.get(context.target) : null;
  const weekday = WEEKDAYS[new Date(context.today + 'T12:00:00Z').getUTCDay()];
  return [
    `Сегодня ${context.today}, ${weekday}.`,
    held ? `Пользователь держит задачу ${taskLine(held, byId)}.` : 'Пользователь не держит никакую задачу.',
    'Задачи в списке: ' + context.tasks.filter(task => task.id !== 'inbox').map(task => taskLine(task, byId)).join('; ') + '.',
    `Реплика пользователя: «${context.text}»`
  ].join('\n');
}

/** Clef's questions. Every choice of a task lists the tasks by title and parent, and the task in
 *  hand as its own option, so «это», «её» and a request without a name land on it. */
export function buildQuestions(context) {
  const byId = new Map(context.tasks.map(task => [task.id, task]));
  const tasks = Object.fromEntries(context.tasks.filter(task => task.id !== 'inbox').map(task => [task.id, taskLine(task, byId)]));
  const held = context.target ? { held: 'Задача, которую пользователь держит (реплика не называет другую)' } : {};
  const choice = (instructions, criteria) => ({ type: 'choice', instructions, criteria });
  const noul = (instructions) => ({ type: 'noul', instructions });
  return {
    create: noul('Просит ли реплика создать новую задачу или называет новое дело (существительным или инфинитивом), которого нет в списке?'),
    note: noul('Просит ли реплика добавить к задаче заметку — новый текст-информацию со словами «заметка», «запомни», «инфо»?'),
    rename: noul('Просит ли реплика переименовать задачу (переименуй, измени на, назови, исправь на)?'),
    status: noul('Просит ли реплика сменить статус существующей задачи: готово, сделано, в фокус, отложить, удалить или в архив, вернуть в работу, сделать заметкой?'),
    deadline: noul('Ставит ли реплика срок существующей задаче?'),
    move: noul('Просит ли реплика перенести существующую задачу под другую задачу или в корень?'),
    question: noul('Это вопрос о списке задач, а не просьба что-то изменить?'),
    ambiguous: noul('Называет ли реплика задачу так, что под это описание подходят две или больше задач из списка с одинаковым названием, а уточнения нет?'),
    statusValue: choice('Какой статус просит поставить реплика?', {
      Done: 'Готово, сделано, выполнено, дело сделано (купил, отправил)',
      Focus: 'В фокус, сфокусировать, срочно', Pause: 'Отложить, на паузу, потом',
      Archive: 'Удалить, убрать, в архив, не нужно', Open: 'Вернуть в работу, открыть снова',
      Info: 'Сделать заметкой'
    }),
    place: choice('Куда положить новую задачу?', {
      inbox: 'Во «Входящие»: пользователь не держит задачу и не сказал «в корень»',
      inside: 'Внутрь задачи, которую пользователь держит: держит задачу, или сказано «подзадача», «сначала надо»',
      beside: 'Рядом с задачей, которую держит пользователь: «рядом», «на том же уровне», «ещё задача», «раздели на»',
      root: 'В корень списка: сказано «в корень» или «новый проект»'
    }),
    statusTarget: choice('У какой задачи сменить статус?', { ...held, ...tasks }),
    renameTarget: choice('Какую задачу переименовать?', { ...held, ...tasks }),
    deadlineTarget: choice('Какой задаче поставить срок?', { ...held, ...tasks }),
    moveSubject: choice('Какую задачу перенести?', { ...held, ...tasks }),
    moveDestination: choice('Куда перенести задачу?', { root: 'В корень списка', ...tasks })
  };
}

/** Clairic's fields. A text value is an exact span of the request. */
export function buildFields() {
  return {
    title1: { type: 'text', instructions: 'Первая новая задача, которую просит создать реплика: всё её название целиком, без слов-команд («добавь», «создай», «новая задача», «раздели на», «рядом»), без места и срока' },
    title2: { type: 'text', instructions: 'Вторая новая задача целиком, если реплика просит создать две или больше (соединены «и» или запятой)' },
    title3: { type: 'text', instructions: 'Третья новая задача целиком, если реплика просит создать три' },
    newTitle: { type: 'text', instructions: 'Новое название задачи при переименовании, без слов «переименуй в», «измени на» и без срока' },
    noteText: { type: 'text', instructions: 'Текст заметки после слов «заметка», «запомни», «инфо»' },
    deadline: { type: 'date', instructions: 'Срок, названный в реплике: сегодня, завтра = сегодня + 1, день недели — первая такая дата строго после сегодня, через неделю = сегодня + 7, к концу месяца — последний день месяца' },
    leftover: { type: 'text', instructions: 'Часть реплики, которую нельзя выполнить в списке задач: просьба к ассистенту (включи музыку, закажи такси) или срок, названный для новой задачи; если такой нет — пусто' }
  };
}

/** The decision as commands, in the shape resolveOpenAI returns. */
export function assemble(context, clef, clairic) {
  const a = clef.answers; const v = clairic.answers;
  const yes = (name) => (a[name]?.noul ?? 0) >= YES;
  const value = (name) => (v[name] && (v[name].p_null ?? 0) < 0.5 ? v[name].value : null);
  const pick = (name) => { const id = a[name]?.choice; return id === 'held' ? context.target : id; };
  const titles = [...new Set(['title1', 'title2', 'title3'].map(value).filter(Boolean).map(text => String(text).trim()).filter(Boolean))];
  const leftover = [value('leftover')].filter(Boolean).map(text => String(text).trim()).filter(text => text.length > 2);
  if (yes('question') && !['create', 'rename', 'status', 'deadline', 'move', 'note'].some(yes)) return { answer: 'Это вопрос, а не команда.', commands: [] };
  if (yes('ambiguous')) return { answer: 'Под это описание подходит несколько задач — уточните, какая.', commands: [] };
  const commands = [];
  const byId = new Map(context.tasks.map(task => [task.id, task]));
  if (yes('create') && titles.length) {
    const place = a.place?.choice;
    for (const title of titles) {
      if (place === 'root' || (place === 'beside' && !context.targetParent)) commands.push({ command: 'addItem', actId: 'list', payload: { title } });
      else if (place === 'beside') commands.push({ command: 'addChild', actId: context.targetParent, payload: { title } });
      else if (place === 'inside' && context.target) commands.push({ command: 'addChild', actId: context.target, payload: { title } });
      else commands.push({ command: 'addChild', actId: context.target || 'inbox', payload: { title } });
    }
  }
  if (yes('note') && value('noteText') && context.target) commands.push({ command: 'addChild', actId: context.target, payload: { title: value('noteText'), status: 'Info' } });
  if (yes('rename') && value('newTitle') && byId.has(pick('renameTarget'))) commands.push({ command: 'editItem', actId: pick('renameTarget'), payload: { title: value('newTitle') } });
  if (yes('status') && byId.has(pick('statusTarget'))) commands.push({ command: 'setStatus', actId: pick('statusTarget'), payload: { status: a.statusValue.choice } });
  if (yes('deadline') && value('deadline') && byId.has(pick('deadlineTarget'))) commands.push({ command: 'setDeadline', actId: pick('deadlineTarget'), payload: { deadline: value('deadline') } });
  if (yes('move') && byId.has(pick('moveSubject'))) commands.push({ command: 'setParent', actId: pick('moveSubject'), payload: { parentId: a.moveDestination.choice === 'root' ? null : a.moveDestination.choice } });
  return { answer: leftover.length ? `Не выполнено: ${leftover.join('; ')}` : '', commands };
}

/** One request: Clef and Clairic in parallel, then the assembly. run(model, input) calls a model. */
export async function resolveDecision(context, run, { clefModel = 'clef' } = {}) {
  const state = buildState(context);
  const [clef, clairic] = await Promise.all([
    run(`@cf/cloudflare/${clefModel}`, { model: clefModel, state, questions: buildQuestions(context) }),
    // Clairic returns spans of its state: given the task list it quotes titles from it, so it reads
    // only the request and the date.
    run('@cf/cloudflare/clairic', { model: 'clairic', state: state.split('\n').filter(line => !line.startsWith('Задачи в списке') && !line.startsWith('Пользователь')).join('\n'), fields: buildFields() })
  ]);
  return assemble(context, clef, clairic);
}
