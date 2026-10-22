import { describe, expect, it } from 'vitest';
import { QUICK_RESULT_MS, asideFor, confirmText, deliveryFor, stillUnsaid } from '../../src/v2/domain/live-delegation.js';

const items = [{ id: 'rt', title: 'Клубника' }, { id: 'rl', title: 'Очень длинное название задачи про отпуск' }];

describe('confirmations', () => {
  it('say the change in a few words', () => {
    expect(confirmText({ command: 'setStatus', actId: 'rt', payload: { status: 'Pause' } }, items)).toBe('Клубника на паузе');
    expect(confirmText({ command: 'setDeadline', actId: 'rt', payload: { deadline: '2026-10-23' } }, items)).toBe('Клубника до 23.10');
    expect(confirmText({ command: 'addChild', actId: 'inbox', payload: { title: 'молоко' } }, items)).toBe('Добавил: молоко');
    expect(confirmText({ command: 'editItem', actId: 'rt', payload: { title: 'Малина' } }, items)).toBe('Теперь Малина');
  });

  it('cut a long title to its first words', () => {
    expect(confirmText({ command: 'setStatus', actId: 'rl', payload: { status: 'Focus' } }, items)).toBe('Очень длинное название задачи… в фокусе');
  });
});

describe('when a result is spoken', () => {
  const quick = QUICK_RESULT_MS - 1;
  const late = QUICK_RESULT_MS + 1;

  it('speaks a quick result in a quiet moment at once', () => {
    expect(deliveryFor({ elapsedMs: quick, active: false, failed: false, text: 'Клубника на паузе' }))
      .toEqual({ kind: 'speak', content: 'Клубника на паузе' });
  });

  it('holds a late result, or one that would cut into the conversation', () => {
    expect(deliveryFor({ elapsedMs: late, active: false, failed: false, text: 'Клубника на паузе' }).kind).toBe('hold');
    expect(deliveryFor({ elapsedMs: quick, active: true, failed: false, text: 'Клубника на паузе' }).kind).toBe('hold');
  });

  it('always speaks a failure', () => {
    expect(deliveryFor({ elapsedMs: late, active: true, failed: true, text: 'Не вышло: нет задачи' }))
      .toEqual({ kind: 'speak', content: 'Кстати, не вышло: нет задачи' });
  });

  it('speaks an answer unless the person has moved on', () => {
    expect(deliveryFor({ elapsedMs: late, active: false, answer: true, text: 'В фокусе отчёт.' }).kind).toBe('speak');
    expect(deliveryFor({ elapsedMs: quick, active: true, answer: true, text: 'В фокусе отчёт.' }).kind).toBe('quiet');
  });

  it('turns held confirmations into the aside for the next pause', () => {
    expect(asideFor(['Клубника на паузе', 'Отчёт до 23.10'])).toBe('Думаю, кстати, клубника на паузе, отчёт до 23.10.');
    expect(asideFor(['Клубника на паузе'], { thinking: false })).toBe('Кстати, клубника на паузе.');
    expect(asideFor([])).toBeNull();
  });
});

describe('held confirmations the voice already covered', () => {
  it('are dropped when the voice named the task, whatever the ending', () => {
    expect(stillUnsaid(['Голден на паузе', 'Добавил: хлеб'], ['Поставила «Голдена» на паузу.'])).toEqual(['Добавил: хлеб']);
    expect(stillUnsaid(['Добавил: хлеб'], ['Хлеб добавлен.'])).toEqual([]);
  });
});
