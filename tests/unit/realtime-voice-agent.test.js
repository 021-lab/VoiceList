import { describe, expect, test } from 'vitest';

import {
  createDialogueRepository,
  DIALOGUES_STORAGE_KEY,
  taskContextFromState,
  taskTreeFromState,
  taskInputFromToolCall
} from '../../src/realtime-voice-agent.js';

function createMemoryStorage() {
  const entries = new Map();
  return {
    getItem(key) { return entries.get(key) || null; },
    setItem(key, value) { entries.set(key, String(value)); }
  };
}

describe('Realtime voice agent', () => {
  test('persists separate dialogue transcripts across repository instances', () => {
    const storage = createMemoryStorage();
    let tick = 0;
    const now = () => new Date(`2026-08-16T10:00:0${tick++}.000Z`);
    let id = 0;
    const createId = () => `id-${++id}`;
    const repository = createDialogueRepository({ storage, now, createId });
    const dialogueId = repository.start();
    repository.append(dialogueId, { role: 'user', text: 'Добавь задачу' });
    repository.append(dialogueId, { role: 'assistant', text: 'Готово' });
    repository.finish(dialogueId);

    const reloaded = createDialogueRepository({ storage, now, createId }).list();
    expect(reloaded).toHaveLength(1);
    expect(reloaded[0].messages.map((message) => [message.role, message.text])).toEqual([
      ['user', 'Добавь задачу'],
      ['assistant', 'Готово']
    ]);
    expect(reloaded[0].endedAt).toBeTruthy();
    expect(JSON.parse(storage.getItem(DIALOGUES_STORAGE_KEY))).toHaveLength(1);
  });

  test('maps exactly the documented task operations to existing command envelopes', () => {
    expect(taskInputFromToolCall('addItem', '{"line1":"Позвонить"}')).toEqual({
      actId: 'list',
      actType: 'list',
      command: 'addItem',
      payload: { line1: 'Позвонить' },
      source: 'openai-realtime'
    });
    expect(taskInputFromToolCall('addInfo', { parentId: 'apple', line1: 'Сезонные дешевле в сентябре' })).toEqual({
      actId: 'apple',
      actType: 'task',
      command: 'addChild',
      payload: { line1: 'Сезонные дешевле в сентябре', status: 'Info' },
      source: 'openai-realtime'
    });
    expect(taskInputFromToolCall('setStatus', { taskId: 'milk1', status: 'Done' })).toMatchObject({
      actId: 'milk1',
      command: 'setStatus',
      payload: { status: 'Done' }
    });
    expect(() => taskInputFromToolCall('deleteItem', { taskId: 'milk1' })).toThrow(/Unsupported task operation/);
  });

  test('attaches the latest user transcript to Realtime tool commands for the server log', () => {
    expect(taskInputFromToolCall(
      'setParent',
      { taskId: 'fudji', parentId: 'goldn' },
      { transcript: 'Фуджи перенеси под голден' }
    )).toMatchObject({
      actId: 'fudji',
      command: 'setParent',
      payload: { parentId: 'goldn' },
      source: 'openai-realtime',
      transcript: 'Фуджи перенеси под голден'
    });
  });

  test('builds agent context from task data but excludes the action log', () => {
    const context = taskContextFromState({
      snapshot: {
        items: [{ id: 'one', line1: 'One', status: 'Focus', privateExtra: 'omit' }]
      },
      actionLog: [{ transcript: 'do not send' }]
    });
    expect(context).toEqual([{
      id: 'one',
      parentId: null,
      order: 0,
      status: 'Focus',
      line1: 'One',
      line2: '',
      tags: []
    }]);
    expect(JSON.stringify(context)).not.toContain('do not send');
  });

  test('builds the nested task tree context with only id, title, status, and children', () => {
    const tree = taskTreeFromState({
      snapshot: {
        items: [
          { id: 'apple', parentId: null, line1: 'Яблоки', line2: 'secret details', status: 'Focus', order: 1 },
          { id: 'goldn', parentId: 'apple', line1: 'Голден', status: 'Open', order: 2 },
          { id: 'fudji', parentId: 'apple', line1: 'Фуджи', status: 'Pause', tags: ['omit'], order: 3 }
        ]
      },
      actionLog: [{ transcript: 'omit' }]
    });
    expect(tree).toEqual([{
      id: 'apple',
      title: 'Яблоки',
      status: 'Focus',
      children: [
        { id: 'goldn', title: 'Голден', status: 'Open', children: [] },
        { id: 'fudji', title: 'Фуджи', status: 'Pause', children: [] }
      ]
    }]);
    expect(JSON.stringify(tree)).not.toContain('secret details');
    expect(JSON.stringify(tree)).not.toContain('omit');
  });
});
