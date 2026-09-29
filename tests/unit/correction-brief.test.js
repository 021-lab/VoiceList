import { describe, expect, it } from 'vitest';
import { AGENT_SYSTEM, buildCorrectionMessages, resolveOpenAI } from '../../worker/v2/model.js';

const context = (over = {}) => ({
  text: 'Так неправильно, надо было в той же ветке',
  target: 'sm', targetParent: 'sk', today: '2026-09-29',
  tasks: [{ id: 'sm', parentId: 'sk', line1: 'Данные сторон', status: 'Open' }],
  correction: {
    userText: 'Так неправильно, надо было в той же ветке',
    original: {
      id: 'e36', kind: 'text', text: 'Разнеси на две задачи', heard: null, source: 'agent',
      modelContext: { text: 'Разнеси на две задачи', target: 'sm', tasks: [{ id: 'sm', line1: 'Данные сторон' }] },
      rawModelResponse: '{"reply":"","commands":[{"command":"addItem","payload":{"line1":"Продавец"}}]}',
      answer: '', commands: [{ command: 'addItem', payload: { line1: 'Продавец' } }]
    },
    outcome: { status: 'applied', error: null, target: 'sr' },
    ...over
  }
});

describe('бриф для модели коррекции', () => {
  it('несёт пять блоков: инструкцию, контекст, ответ, исход и слова пользователя', () => {
    const [system, user] = buildCorrectionMessages(context());
    expect(system.content).toMatch(/исправляешь работу другой модели/);
    for (const block of ['инструкция_исходной_модели', 'контекст_исходной_модели', 'ответ_исходной_модели',
      'что_получилось', 'что_сказал_пользователь', 'состояние_сейчас']) {
      expect(user.content).toContain('<' + block + '>');
    }
    // Инструкция цитируется дословно, а не пересказывается.
    expect(user.content).toContain(AGENT_SYSTEM);
    expect(user.content).toContain('"reply":"","commands"');
    expect(user.content).toContain('Так неправильно, надо было в той же ветке');
    // Состояние документа едет без самого дела о коррекции, иначе оно было бы дважды.
    expect(user.content).not.toContain('"correction"');
    // И список задач едет один раз: в контексте того момента он заменён пометкой.
    expect(user.content).toContain('<опущено: 1 задач на тот момент');
  });

  it('у голосового действия честно говорит, что рассуждения модели нет', () => {
    const voice = context();
    voice.correction.original = { ...voice.correction.original, modelContext: null, rawModelResponse: null, heard: 'разнеси на две' };
    const [, user] = buildCorrectionMessages(voice);
    expect(user.content).toMatch(/недоступен: действие сделала голосовая модель/);
    expect(user.content).toContain('разнеси на две');
  });
});

describe('вызов модели', () => {
  async function call(modelContext) {
    let sent = null;
    await resolveOpenAI({
      apiKey: 'k', model: 'gpt-4.1-mini', correctionModel: 'gpt-5.6-luna', modelContext,
      fetchImpl: async (url, options) => {
        sent = JSON.parse(options.body);
        return { ok: true, body: null, json: async () => ({}), ...stubBody({ choices: [{ message: { content: '{}' } }] }) };
      }
    });
    return sent;
  }
  const stubBody = (payload) => {
    const bytes = new TextEncoder().encode(JSON.stringify(payload));
    return { body: { getReader: () => { let done = false; return { read: async () => done ? { done: true } : (done = true, { done: false, value: bytes }), cancel: async () => {} }; } } };
  };

  it('коррекция уходит своей модели, обычная реплика — агентской', async () => {
    expect((await call(context())).model).toBe('gpt-5.6-luna');
    expect((await call({ text: 'добавь хлеб', tasks: [] })).model).toBe('gpt-4.1-mini');
  });
});
