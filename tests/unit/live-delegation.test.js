import { describe, expect, it } from 'vitest';
import { QUICK_RESULT_MS, confirmText, deliveryFor } from '../../src/v2/domain/live-delegation.js';

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

  it('speaks a quick result at once', () => {
    expect(deliveryFor({ elapsedMs: quick, movedOn: true, failed: false, text: 'Клубника на паузе' }))
      .toEqual({ kind: 'commentary', content: 'Клубника на паузе' });
  });

  it('speaks a late result as an aside', () => {
    expect(deliveryFor({ elapsedMs: late, movedOn: false, failed: false, text: 'Клубника на паузе' }))
      .toEqual({ kind: 'commentary', content: 'Кстати, клубника на паузе' });
  });

  it('keeps a late result silent once the conversation has moved on', () => {
    expect(deliveryFor({ elapsedMs: late, movedOn: true, failed: false, text: 'Клубника на паузе' }).kind).toBe('thinking');
  });

  it('always speaks a failure', () => {
    expect(deliveryFor({ elapsedMs: late, movedOn: true, failed: true, text: 'Не вышло: нет задачи' }).kind).toBe('commentary');
  });
});
