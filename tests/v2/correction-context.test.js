import { describe, expect, it } from 'vitest';
import { DocumentRuntime } from '../../src/v2/domain/document-runtime.js';

/** Живой случай с прода: на строку журнала сказали «так неправильно, надо иначе», и ничего не
 *  произошло — модель получила фразу без действия, о котором речь, и переспросила. */
function runtime(onContext = () => {}) {
  return new DocumentRuntime({
    resolveModel: async (context) => {
      onContext(context.modelContext || context);
      return JSON.stringify({ answer: 'понял', commands: [] });
    }
  });
}
const speak = (app, text, elementId, seq) => ({
  key: { clientKey: 'ui', seq }, context: { elementId, view: 'log', revision: app.graph.revision }, text
});

describe('реплика, сказанная на действие', () => {
  it('становится его корректировкой, даже когда экран не передал actionId', async () => {
    const seen = [];
    const app = runtime(context => seen.push(context));
    const first = await app.executeAndWait({
      key: { clientKey: 'ui', seq: 1 }, context: { elementId: 'app', view: 'list', revision: 0 },
      text: 'Добавь задачу купить молоко'
    });

    await app.executeAndWait(speak(app, 'Так неправильно, надо две задачи', 'action:' + first.requestId, 2));

    const entry = app.journal.entries.at(-1);
    expect(entry.corrects).toBe(first.requestId);
    // Модель видит исправляемое действие: что просили, что ответили, чем кончилось.
    const context = seen.at(-1);
    expect(context.action).toMatchObject({ id: first.requestId, text: 'Добавь задачу купить молоко' });
    expect(context.action).toHaveProperty('status');
    expect(context.history.length).toBeGreaterThan(0);
  });

  it('обычная реплика ни к чему не привязывается', async () => {
    const seen = [];
    const app = runtime(context => seen.push(context));
    await app.executeAndWait({
      key: { clientKey: 'ui', seq: 1 }, context: { elementId: 'app', view: 'list', revision: 0 }, text: 'Добавь хлеб'
    });
    expect(app.journal.entries.at(-1).corrects).toBeUndefined();
    expect(seen.at(-1).action).toBeNull();
  });

  it('ссылка на несуществующее действие отвергается, а не молчит', async () => {
    const app = runtime();
    await expect(app.submit(speak(app, 'Поправь', 'action:e999', 5))).rejects.toMatchObject({ code: 'NOT_FOUND' });
  });
});
