import { describe, expect, it } from 'vitest';
import { formatHistory } from '../../src/v2/domain/history.js';

const speech = (cursor, role, text, source, session) => ({ id: 'e' + cursor, cursor, at: `2026-11-24T02:1${cursor}:00.000Z`, kind: 'speech', role, text, source, key: { clientKey: `${source}:${session}:speech` } });
const command = (cursor, cmd, session) => ({ id: 'e' + cursor, cursor, at: `2026-11-24T02:1${cursor}:30.000Z`, kind: 'ui', command: cmd, key: { clientKey: `gemini:${session}:call${cursor}` } });

describe('history of past conversations', () => {
  const items = [{ id: 'wu', title: 'Разблокировать аккаунт в Тинькофф банке' }];
  const entries = [
    speech(1, 'user', 'Мне нужно разблокировать аккаунт, а потом послать денег Маше.', 'gemini-live', 'gm_1'),
    command(2, { command: 'addItem', actId: null, payload: { title: 'Разблокировать аккаунт в Тинькофф банке' }, source: 'gemini-live' }, 'gm_1'),
    command(3, { command: 'setStatus', actId: 'wu', payload: { status: 'Done' }, source: 'ui' }, 'gm_1'),
    speech(4, 'assistant', 'Done.', 'gemini-live', 'gm_1'),
    speech(5, 'user', 'Что ты сделал?', 'gemini-live', 'gm_2')
  ];
  const technical = { executor: { e2: { status: 'complete', outcomes: [{ target: 'wu' }] } } };

  it('tells what was said and done, by conversation, leaving button presses out', () => {
    const text = formatHistory(entries, technical, items);
    expect(text).toContain('— разговор Gemini gm_1, 2026-11-24 02:11 UTC —');
    expect(text).toContain('02:11 Человек: Мне нужно разблокировать аккаунт, а потом послать денег Маше.');
    expect(text).toContain('Сделано: добавить «Разблокировать аккаунт в Тинькофф банке» — выполнено');
    expect(text).toContain('Голос: Done.');
    expect(text).toContain('— разговор Gemini gm_2');
    expect(text).not.toContain('статус');
  });

  it('keeps the newest lines and still names their conversation', () => {
    const text = formatHistory(entries, technical, items, { limit: 2 });
    expect(text.split('\n')[0]).toMatch(/^— разговор Gemini/);
    expect(text).toContain('Что ты сделал?');
  });

  it('names a renamed task by its old title and labels typed requests as text', () => {
    const renamed = { id: 'e9', cursor: 9, at: '2026-11-24T03:00:00.000Z', kind: 'ui', key: { clientKey: 'gpt-live:live_1' },
      command: { command: 'editItem', actId: 'wu', payload: { title: 'Кефир' }, source: 'gpt-live' } };
    const typed = { id: 'e10', cursor: 10, at: '2026-11-24T03:05:00.000Z', kind: 'text', text: 'перенеси кефир в корень', key: { clientKey: '3f2a9c1e-uuid' },
      commands: [{ command: 'setParent', actId: 'wu', payload: { parentId: null } }] };
    const tech = { executor: { e9: { outcomes: [{ changes: [{ id: 'wu', before: { title: 'Молоко' }, after: { title: 'Кефир' } }] }] } } };
    const text = formatHistory([renamed, typed], tech, [{ id: 'wu', title: 'Кефир' }]);
    expect(text).toContain('переименовать «Молоко» → «Кефир»');
    expect(text).toContain('— разговор текст,');
    expect(text).toContain('перенести «Кефир» в корень');
  });

  it('says so when there is nothing yet', () => {
    expect(formatHistory([], {}, [])).toBe('Журнал пуст: прошлых разговоров нет.');
  });
});
