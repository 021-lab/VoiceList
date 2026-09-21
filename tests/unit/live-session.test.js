import { describe, expect, it } from 'vitest';
import {
  DEFAULT_BACKEND_MODEL, DEFAULT_VOICE_PROMPT, LIVE_MODEL, LIVE_TOOL_NAMES, SNAPSHOT_HEADER,
  buildLiveSessionConfig, formatSnapshotDeltas, formatTaskSnapshot, isLoggableEvent, readFunctionCall
} from '../../src/v2/domain/live-session.js';

const task = (id, parentId, status, line1, extra = {}) => ({ id, parentId, order: 10, status, line1, line2: '', collapsed: false, tags: [], ...extra });

const sample = [
  task('rs', null, 'Open', 'Яблоки'),
  task('rt', 'rs', 'Open', 'Голден'),
  task('ru', 'rs', 'Pause', 'Фуджи'),
  task('rv', 'ru', 'Focus', 'Первый позад'),
  task('rw', null, 'Done', 'Купленное'),
  task('rx', null, 'Archive', 'Старое')
];

describe('task snapshot', () => {
  it('writes a header and one row per task in tree order', () => {
    expect(formatTaskSnapshot(sample)).toBe([
      SNAPSHOT_HEADER,
      'rs\t-\tO\tЯблоки',
      'rt\trs\tO\tГолден',
      'ru\trs\tP\tФуджи',
      'rv\tru\tF\tПервый позад'
    ].join('\n'));
  });

  it('leaves Done and Archive out', () => {
    const text = formatTaskSnapshot(sample);
    expect(text).not.toContain('Купленное');
    expect(text).not.toContain('Старое');
  });

  it('collapses whitespace so a title cannot forge a column or a row', () => {
    const text = formatTaskSnapshot([task('rs', null, 'Open', 'Хитрая\trz\t-\tO\tПодделка')]);
    expect(text.split('\n')).toHaveLength(2);
    expect(text).toContain('rs\t-\tO\tХитрая rz - O Подделка');
  });

  it('keeps a task whose parent is hidden, attached to the root', () => {
    const rows = formatTaskSnapshot([task('rw', null, 'Done', 'Купленное'), task('ry', 'rw', 'Open', 'Осталось')]);
    expect(rows).toContain('ry\t-\tO\tОсталось');
  });
});

describe('snapshot deltas', () => {
  const before = task('ru', 'rs', 'Pause', 'Фуджи');

  it('reports a created task in the same language as the snapshot', () => {
    expect(formatSnapshotDeltas([{ id: 'rz', before: null, after: task('rz', 'rs', 'Open', 'Гренни Смит'), fields: null }]))
      .toEqual(['+ rz rs O Гренни Смит']);
  });

  it('reports rename, status and move separately', () => {
    expect(formatSnapshotDeltas([
      { id: 'ru', before, after: { ...before, line1: 'Фуджи красные' }, fields: ['line1'] },
      { id: 'ru', before, after: { ...before, status: 'Focus' }, fields: ['status'] },
      { id: 'ru', before, after: { ...before, parentId: 'rw' }, fields: ['parentId'] }
    ])).toEqual(['~ ru Фуджи красные', '* ru F', '> ru rw']);
  });

  it('treats a move into Done as leaving the snapshot', () => {
    expect(formatSnapshotDeltas([{ id: 'ru', before, after: { ...before, status: 'Done' }, fields: ['status'] }]))
      .toEqual(['- ru']);
  });

  it('treats a move out of Done as entering the snapshot', () => {
    const done = { ...before, status: 'Done' };
    expect(formatSnapshotDeltas([{ id: 'ru', before: done, after: before, fields: ['status'] }]))
      .toEqual(['+ ru rs P Фуджи']);
  });
});

describe('session config', () => {
  const config = buildLiveSessionConfig({ items: sample, voicePrompt: 'Голос.', backendPrompt: 'Бэкенд.', backendModel: 'gpt-5.6-terra' });

  it('runs gpt-live-1 with responses delegation', () => {
    expect(config.model).toBe(LIVE_MODEL);
    expect(config.delegation.type).toBe('responses');
    expect(config.delegation.responses.model).toBe('gpt-5.6-terra');
    expect(config.delegation.responses.instructions).toBe('Бэкенд.');
  });

  it('puts the tools on the backend, not on the session', () => {
    expect(config.tools).toBeUndefined();
    expect(config.delegation.responses.tools.map(tool => tool.name)).toEqual(LIVE_TOOL_NAMES);
  });

  it('carries the task snapshot in the voice instructions', () => {
    expect(config.instructions).toContain('Голос.');
    expect(config.instructions).toContain('rt\trs\tO\tГолден');
  });

  it('falls back to the default backend model', () => {
    expect(buildLiveSessionConfig({}).delegation.responses.model).toBe(DEFAULT_BACKEND_MODEL);
  });
});

describe('function calls', () => {
  it('reads a call nested inside an item', () => {
    expect(readFunctionCall({ type: 'response.output_item.done', item: { type: 'function_call', call_id: 'call_1', name: 'setStatus', arguments: '{"taskId":"rs","status":"Focus"}' } }))
      .toEqual({ callId: 'call_1', name: 'setStatus', arguments: { taskId: 'rs', status: 'Focus' } });
  });

  it('reads a call carried on the event itself', () => {
    expect(readFunctionCall({ call_id: 'call_2', name: 'getFrontier', arguments: '{}' }))
      .toEqual({ callId: 'call_2', name: 'getFrontier', arguments: {} });
  });

  it('survives arguments that are not valid JSON', () => {
    expect(readFunctionCall({ call_id: 'call_3', name: 'addItem', arguments: '{oops' }).arguments).toEqual({});
  });

  it('ignores an output item that is not a function call', () => {
    expect(readFunctionCall({ item: { type: 'message', call_id: 'call_4', name: 'nope' } })).toBeNull();
  });
});

describe('loggable events', () => {
  it('keeps transcripts, delegations and finished items', () => {
    for (const type of ['session.started', 'session.delegation.created', 'response.event', 'session.closed', 'session.usage.updated']) {
      expect(isLoggableEvent(type)).toBe(true);
    }
  });

  it('drops audio bytes and every streamed delta, transcripts included', () => {
    for (const type of ['session.output_audio.delta', 'session.input_audio.append', 'response.output_text.delta', 'session.input_transcript.delta', 'session.output_transcript.delta']) {
      expect(isLoggableEvent(type)).toBe(false);
    }
  });
});

describe('the voice prompt', () => {
  it('makes the choice between several matching tasks a dialogue that precedes delegation', () => {
    expect(DEFAULT_VOICE_PROMPT).toContain('Подошло несколько');
    expect(DEFAULT_VOICE_PROMPT).toContain('Дождись ответа');
    expect(DEFAULT_VOICE_PROMPT).toContain('к бэкенду не обращайся');
    expect(DEFAULT_VOICE_PROMPT).toContain('повтори выбранную задачу вслух вместе с идентификатором');
    // A single match must still go through without a question, or every change costs a turn.
    expect(DEFAULT_VOICE_PROMPT).toContain('Подошла ровно одна — не переспрашивай');
  });
});
