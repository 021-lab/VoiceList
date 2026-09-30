import { describe, expect, it } from 'vitest';
import { TaskAgent } from '../../src/v2/domain/task-agent.js';

/** Живой случай с прода: пользователь держал подзадачу «Данные сторон» внутри проекта и
 *  попросил разнести её на две — продавца и покупателя. Модель ответила двумя addItem, и обе
 *  задачи легли в корень, за пределы проекта. */
const tasks = [
  { id: 'inbox', parentId: null, order: 0, status: 'Open', title: 'Входящие', collapsed: false, tags: [] },
  { id: 'sj', parentId: null, order: 10, status: 'Open', title: 'Договор продажи квартиры', collapsed: false, tags: [] },
  { id: 'sk', parentId: 'sj', order: 10, status: 'Open', title: 'Черновик договора', collapsed: false, tags: [] },
  { id: 'sm', parentId: 'sk', order: 20, status: 'Open', title: 'Данные сторон', collapsed: false, tags: [] }
];
const agent = new TaskAgent();
const decision = (commands) => JSON.stringify({ answer: '', commands });
const context = (target, targetParent) => ({ tasks, target, targetParent, text: 'разнеси на две' });

describe('ветка разговора', () => {
  it('равноправная задача встаёт рядом с той, что в руках, а не в корень', () => {
    const { commands } = agent.parse(decision([
      { command: 'addItem', actId: 'new1', actType: 'addItem', payload: { title: 'Данные продавца' } },
      { command: 'addItem', actId: 'new2', actType: 'addItem', payload: { title: 'Данные покупателя' } }
    ]), context('sm', 'sk'));

    expect(commands.map(c => [c.command, c.actId])).toEqual([['addChild', 'sk'], ['addChild', 'sk']]);
    expect(commands[0].payload.title).toBe('Данные продавца');
  });

  it('у корневой цели ветка — это она сама', () => {
    const { commands } = agent.parse(decision([{ command: 'addItem', actId: 'new1', payload: { title: 'Подзадача' } }]), context('sj', null));
    expect(commands[0]).toMatchObject({ command: 'addChild', actId: 'sj' });
  });

  it('без цели задача по-прежнему создаётся в корне', () => {
    const { commands } = agent.parse(decision([{ command: 'addItem', actId: 'list', payload: { title: 'Купить молоко' } }]), context(null, null));
    expect(commands[0]).toMatchObject({ command: 'addItem', actId: 'list' });
  });

  it('подзадача, названная моделью явно, остаётся там, куда её положили', () => {
    const { commands } = agent.parse(decision([{ command: 'addChild', actId: 'sm', payload: { title: 'Паспорт' } }]), context('sm', 'sk'));
    expect(commands[0]).toMatchObject({ command: 'addChild', actId: 'sm' });
  });
});
