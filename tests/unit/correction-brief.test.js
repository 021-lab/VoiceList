import { describe, expect, it } from 'vitest';
import { AGENT_SYSTEM, buildCorrectionMessages, resolveOpenAI } from '../../worker/v2/model.js';

const task = (id, title, over = {}) => ({
  id, parentId: null, order: 10, status: 'Open', title, collapsed: false, tags: [], ...over
});
const context = () => ({
  text: 'Так неправильно, надо было в той же ветке', today: '2026-09-29', graphRevision: 31,
  target: 'sm', targetParent: 'sk',
  tasks: [
    task('sm', 'Данные сторон', { parentId: 'sk' }),
    task('sr', 'Данные сторон - продавец'),
    task('old', 'Прошлогодняя', { status: 'Archive' }),
    task('done', 'Сделанная', { status: 'Done' })
  ],
  correction: {
    userText: 'Так неправильно, надо было в той же ветке',
    element: { kind: 'action', actionId: 'e36', request: 'Разнеси на две задачи' },
    actionId: 'e36',
    original: {
      id: 'e36', kind: 'text', text: 'Разнеси на две задачи', heard: null, source: 'agent',
      modelContext: { text: 'Разнеси на две задачи', target: 'sm', tasks: [task('sm', 'Данные сторон')] },
      rawModelResponse: '{"reply":"","commands":[{"command":"addItem","payload":{"title":"Продавец"}}]}',
      answer: '', commands: [{ command: 'addItem', payload: { title: 'Продавец' } }]
    },
    outcome: {
      status: 'applied', error: null, target: 'sr',
      changed: [{ taskId: 'sr', operation: 'created', task: { title: 'Данные сторон - продавец' } }]
    }
  }
});

describe('бриф для модели коррекции', () => {
  const [system, user] = buildCorrectionMessages(context());

  it('идёт шестью блоками в согласованном порядке', () => {
    const order = [...user.content.matchAll(/<([a-z_]+)>/g)].map(match => match[1]);
    expect(order).toEqual([
      'instructions_given_to_first_model', 'context_given_to_first_model', 'answer_returned_by_first_model',
      'what_the_answer_did', 'active_tasks_now', 'what_the_user_said'
    ]);
  });

  it('цитирует инструкцию и ответ исходной модели дословно', () => {
    expect(user.content).toContain(AGENT_SYSTEM);
    expect(user.content).toContain('"reply":"","commands"');
    // Список задач того момента не едет вторым экземпляром.
    expect(user.content).toContain('<omitted: 1 tasks as of that moment');
  });

  it('в исходе несёт идентификатор действия для отката и сами изменённые строки', () => {
    expect(user.content).toContain('"actionId":"e36"');
    expect(user.content).toContain('"operation":"created"');
    expect(user.content).toContain('Данные сторон - продавец');
  });

  it('состояние задач — только активные и без служебных полей отрисовки', () => {
    const state = user.content.match(/<active_tasks_now>\n(.*)\n<\/active_tasks_now>/s)[1];
    const items = JSON.parse(state);
    expect(items.map(item => item.id)).toEqual(['sm', 'sr']);
    expect(items[0]).not.toHaveProperty('order');
    expect(items[0]).not.toHaveProperty('collapsed');
  });

  it('слова пользователя идут последними и несут нажатый элемент', () => {
    expect(user.content.indexOf('<what_the_user_said>')).toBeGreaterThan(user.content.indexOf('<active_tasks_now>'));
    expect(user.content).toContain('"elementInHand"');
    expect(user.content).toContain('"kind":"action"');
    expect(user.content).toContain('Сегодня 2026-09-29.');
  });

  it('велит архивировать, а не удалять, и даёт откат по идентификатору действия', () => {
    expect(system.content).toMatch(/управляющей списком задач/);
    expect(system.content).toMatch(/отправь в архив setStatus со статусом Archive/);
    expect(system.content).toMatch(/deleteItem применяй только когда пользователь прямо просит удалить/);
    expect(system.content).toMatch(/rollbackAction с actId, равным actionId из блока what_the_answer_did/);
  });

  it('у голосового действия честно говорит, что рассуждения модели нет', () => {
    const voice = context();
    voice.correction.original = { ...voice.correction.original, modelContext: null, rawModelResponse: null, heard: 'разнеси на две' };
    const [, body] = buildCorrectionMessages(voice);
    expect(body.content).toMatch(/недоступен: действие сделала голосовая модель/);
    expect(body.content).toContain('разнеси на две');
  });
});

describe('вызов модели', () => {
  const stubBody = (payload) => {
    const bytes = new TextEncoder().encode(JSON.stringify(payload));
    return { body: { getReader: () => { let done = false; return { read: async () => done ? { done: true } : (done = true, { done: false, value: bytes }), cancel: async () => {} }; } } };
  };
  async function call(modelContext) {
    let sent = null;
    await resolveOpenAI({
      apiKey: 'k', model: 'gpt-4.1-mini', correctionModel: 'gpt-5.6-sol', modelContext,
      fetchImpl: async (url, options) => {
        sent = JSON.parse(options.body);
        return { ok: true, ...stubBody({ choices: [{ message: { content: '{}' } }] }) };
      }
    });
    return sent;
  }

  it('коррекция уходит своей модели, обычная реплика — агентской', async () => {
    expect((await call(context())).model).toBe('gpt-5.6-sol');
    expect((await call({ text: 'добавь хлеб', tasks: [] })).model).toBe('gpt-4.1-mini');
  });

  it('температуру просит только там, где её принимают', async () => {
    // Живой отказ: «Unsupported value: temperature does not support 0 with this model».
    expect((await call(context()))).not.toHaveProperty('temperature');
    expect((await call({ text: 'добавь хлеб', tasks: [] })).temperature).toBe(0);
  });
});
