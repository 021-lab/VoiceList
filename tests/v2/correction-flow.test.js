import { describe, expect, it } from 'vitest';
import { DocumentRuntime } from '../../src/v2/domain/document-runtime.js';
import { buildCorrectionMessages } from '../../worker/v2/model.js';

/** Живой случай с прода целиком: модель создала две задачи в корне вместо ветки, пользователь
 *  сказал «так неправильно», и коррекция должна исправить сделанное, а не переспрашивать. */
describe('коррекция действия', () => {
  it('собирает дело: инструкцию, контекст, ответ, исход и слова пользователя — и правит сделанное', async () => {
    const seen = [];
    const answers = [
      JSON.stringify({ answer: 'Создаю две задачи', commands: [
        { command: 'addChild', actId: 'inbox', actType: 'task', payload: { title: 'Продавец' } }
      ] }),
      JSON.stringify({ answer: 'Перенёс в нужную ветку', commands: [
        { command: 'setParent', actId: 'PLACEHOLDER', actType: 'task', payload: { parentId: 'milk1' } }
      ] })
    ];
    const runtime = new DocumentRuntime({
      resolveModel: async (context) => { seen.push(context.modelContext || context); return answers[seen.length - 1]; }
    });

    const first = await runtime.executeAndWait({
      key: { clientKey: 'ui', seq: 1 }, context: { elementId: 'app', view: 'list', revision: 0 },
      text: 'Разнеси на две задачи'
    });
    const created = runtime.journal.action(runtime.journal.get(first.requestId)).target;
    answers[1] = answers[1].replace('PLACEHOLDER', created);

    await runtime.executeAndWait({
      key: { clientKey: 'ui', seq: 2 },
      context: { elementId: 'action:' + first.requestId, view: 'log', revision: runtime.graph.revision },
      text: 'Так неправильно, она должна быть внутри молока'
    });

    // Модель коррекции получила дело, а не голую фразу.
    const correction = seen.at(-1).correction;
    expect(correction.userText).toContain('внутри молока');
    expect(correction.original.id).toBe(first.requestId);
    expect(correction.original.rawModelResponse).toContain('addChild');
    expect(correction.original.modelContext.text).toBe('Разнеси на две задачи');
    expect(correction.outcome.status).toBe('applied');

    // И бриф из этого дела собирается целиком.
    const [, user] = buildCorrectionMessages(seen.at(-1));
    expect(user.content).toContain('Разнеси на две задачи');
    expect(user.content).toContain('внутри молока');

    // Исправление применено к тому, что было сделано, а не создано заново.
    expect(runtime.graph.read({ id: created }).parentId).toBe('milk1');
    expect(runtime.graph.read().items.filter(item => item.title === 'Продавец')).toHaveLength(1);
  });
});
