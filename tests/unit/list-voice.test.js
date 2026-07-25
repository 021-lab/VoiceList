import { describe, expect, test, vi } from 'vitest';
import {
  createVoiceCommandParser,
  createVoiceController,
  createVoiceUI,
  findItemByPhrase,
  isSpeechRecognitionSupported,
  normalizeSpeech
} from '../../src/list-voice.js';

function createState() {
  return {
    snapshot: {
      items: [
        { id: 'milk1', parentId: null, order: 10, status: 'Open', line1: 'Молоко 3.2%', line2: '', collapsed: false, tags: [] },
        { id: 'bread', parentId: null, order: 20, status: 'Open', line1: 'Хлеб ржаной', line2: '', collapsed: false, tags: [] },
        { id: 'apple', parentId: null, order: 30, status: 'Focus', line1: 'Яблоки', line2: '', collapsed: false, tags: ['Купить'] },
        { id: 'shamp', parentId: null, order: 40, status: 'Archive', line1: 'Шампунь', line2: '', collapsed: false, tags: [] }
      ]
    },
    actionLog: []
  };
}

const parser = createVoiceCommandParser();

describe('speech normalization and item lookup', () => {
  test('normalizes case, punctuation and ё', () => {
    expect(normalizeSpeech('  Отмени,  Всё!  ')).toBe('отмени все');
  });

  test('matches an item by an inflected spoken form', () => {
    const items = createState().snapshot.items;
    expect(findItemByPhrase(items, 'яблоко')?.id).toBe('apple');
    expect(findItemByPhrase(items, 'хлеб')?.id).toBe('bread');
    expect(findItemByPhrase(items, 'молоко')?.id).toBe('milk1');
  });

  test('returns null when nothing is close enough', () => {
    expect(findItemByPhrase(createState().snapshot.items, 'вертолёт')).toBeNull();
  });
});

describe('voice command parser', () => {
  test('parses a top-level add command', () => {
    const result = parser.parse('Добавь задачу купить батарейки', createState());
    expect(result.ok).toBe(true);
    expect(result.input.command).toBe('addItem');
    expect(result.input.payload.line1).toBe('купить батарейки');
  });

  test('splits an optional second line', () => {
    const result = parser.parse('добавь задачу кофе с подписью арабика 250 грамм', createState());
    expect(result.input.payload).toEqual({ line1: 'кофе', line2: 'арабика 250 грамм' });
  });

  test('parses a child add command against an existing item', () => {
    const result = parser.parse('добавь подзадачу голден к яблоки', createState());
    expect(result.ok).toBe(true);
    expect(result.input).toMatchObject({ command: 'addChild', actId: 'apple' });
    expect(result.input.payload.line1).toBe('голден');
  });

  test('parses status commands in several phrasings', () => {
    const state = createState();
    expect(parser.parse('отметь хлеб выполнено', state).input).toMatchObject({ command: 'setStatus', actId: 'bread', payload: { status: 'Done' } });
    expect(parser.parse('фокус на молоко', state).input).toMatchObject({ command: 'setStatus', actId: 'milk1', payload: { status: 'Focus' } });
    expect(parser.parse('в архив хлеб', state).input).toMatchObject({ command: 'setStatus', actId: 'bread', payload: { status: 'Archive' } });
    expect(parser.parse('верни шампунь', state).input).toMatchObject({ command: 'setStatus', actId: 'shamp', payload: { status: 'Open' } });
  });

  test('parses rename, delete, tag and collapse commands', () => {
    const state = createState();
    expect(parser.parse('переименуй хлеб в багет', state).input).toMatchObject({ command: 'editItem', actId: 'bread', payload: { line1: 'багет' } });
    expect(parser.parse('удали шампунь', state).input).toMatchObject({ command: 'deleteItem', actId: 'shamp' });
    expect(parser.parse('пометь молоко как срочно', state).input).toMatchObject({ command: 'setTags', actId: 'milk1', payload: { tag: 'Срочно' } });
    expect(parser.parse('сверни яблоки', state).input).toMatchObject({ command: 'toggleCollapse', actId: 'apple' });
  });

  test('parses view switches and undo', () => {
    const state = createState();
    expect(parser.parse('покажи журнал', state).input.command).toBe('showActionLog');
    expect(parser.parse('покажи фронтир', state).input.command).toBe('showFrontier');
    expect(parser.parse('покажи список', state).input.command).toBe('showList');
    expect(parser.parse('отмена', state).input.command).toBe('undo');
  });

  test('reports an unmatched target instead of guessing', () => {
    const result = parser.parse('удали вертолёт', createState());
    expect(result.ok).toBe(false);
    expect(result.reason).toBe('not-found');
  });

  test('reports an unknown command', () => {
    const result = parser.parse('сыграй музыку', createState());
    expect(result.ok).toBe(false);
    expect(result.reason).toBe('unknown');
  });

  test('rejects an unsupported tag', () => {
    const result = parser.parse('пометь молоко как ерунда', createState());
    expect(result.ok).toBe(false);
    expect(result.reason).toBe('unknown-tag');
  });
});

describe('voice controller', () => {
  function createControllerFixture() {
    const dispatched = [];
    const events = [];
    let state = createState();
    const controller = createVoiceController({
      recognitionFactory: null,
      dispatch: (input) => dispatched.push(input),
      getState: () => state,
      onStatus: (event) => events.push(event)
    });
    return { controller, dispatched, events, setState: (next) => { state = next; } };
  }

  test('dispatches a parsed command tagged with the voice source', () => {
    const { controller, dispatched, events } = createControllerFixture();
    controller.handleTranscript('Добавь задачу купить батарейки');

    expect(dispatched).toHaveLength(1);
    expect(dispatched[0]).toMatchObject({ command: 'addItem', source: 'voice' });
    expect(events.at(-1)).toMatchObject({ status: 'accepted', command: 'addItem' });
  });

  test('does not dispatch when the phrase is not understood', () => {
    const { controller, dispatched, events } = createControllerFixture();
    controller.handleTranscript('сыграй музыку');

    expect(dispatched).toHaveLength(0);
    expect(events.at(-1)).toMatchObject({ status: 'rejected', reason: 'unknown' });
  });

  test('refuses undo before any voice mutation happened', () => {
    const { controller, dispatched, events } = createControllerFixture();
    controller.handleTranscript('отмена');

    expect(dispatched).toHaveLength(0);
    expect(events.at(-1)).toMatchObject({ status: 'rejected', reason: 'no-undo' });
  });

  test('undo replays the snapshot captured before the last voice mutation', () => {
    const { controller, dispatched } = createControllerFixture();
    controller.handleTranscript('удали шампунь');
    controller.handleTranscript('отмена');

    expect(dispatched).toHaveLength(2);
    expect(dispatched[1].command).toBe('undo');
    expect(dispatched[1].payload.snapshot.items.map((item) => item.id)).toContain('shamp');
  });

  test('view commands do not consume the undo snapshot', () => {
    const { controller, dispatched } = createControllerFixture();
    controller.handleTranscript('удали шампунь');
    controller.handleTranscript('покажи журнал');
    controller.handleTranscript('отмена');

    expect(dispatched.map((input) => input.command)).toEqual(['deleteItem', 'showActionLog', 'undo']);
  });

  test('reports unsupported recognition instead of throwing', () => {
    const { controller, events } = createControllerFixture();
    expect(controller.start()).toBe(false);
    expect(events.at(-1)).toMatchObject({ status: 'unsupported' });
  });

  test('drives a fake recognition session end to end', () => {
    const dispatched = [];
    const events = [];
    const recognition = { start: vi.fn(), stop: vi.fn(), lang: '' };
    const controller = createVoiceController({
      recognitionFactory: () => recognition,
      dispatch: (input) => dispatched.push(input),
      getState: () => createState(),
      onStatus: (event) => events.push(event)
    });

    expect(controller.toggle()).toBe(true);
    expect(recognition.start).toHaveBeenCalled();
    expect(recognition.lang).toBe('ru-RU');
    expect(controller.isListening()).toBe(true);

    recognition.onresult({ results: [[{ transcript: 'фокус на' }]] });
    expect(events.at(-1)).toMatchObject({ status: 'interim', transcript: 'фокус на' });

    recognition.onresult({ results: [Object.assign([{ transcript: 'фокус на молоко' }], { isFinal: true })] });
    expect(dispatched.at(-1)).toMatchObject({ command: 'setStatus', actId: 'milk1', payload: { status: 'Focus' } });

    recognition.onerror({ error: 'not-allowed' });
    expect(events.at(-1)).toMatchObject({ status: 'error', message: 'Нет доступа к микрофону' });

    controller.stop();
    expect(recognition.stop).toHaveBeenCalled();
  });
});

describe('voice UI binding', () => {
  test('toggles the controller from the button and reflects status', () => {
    document.body.innerHTML = '<button id="voice-btn"></button><p id="voice-status"></p><p id="voice-transcript"></p>';
    const button = document.getElementById('voice-btn');
    const statusEl = document.getElementById('voice-status');
    const transcriptEl = document.getElementById('voice-transcript');
    const controller = { toggle: vi.fn() };

    const voiceUI = createVoiceUI({ button, statusEl, transcriptEl, controller });
    button.click();
    expect(controller.toggle).toHaveBeenCalledTimes(1);

    voiceUI.onStatus({ status: 'listening', message: 'Слушаю…' });
    expect(button.classList.contains('listening')).toBe(true);
    expect(button.getAttribute('aria-pressed')).toBe('true');
    expect(statusEl.dataset.state).toBe('listening');

    voiceUI.onStatus({ status: 'accepted', message: 'Добавил задачу «чай»', transcript: 'добавь задачу чай' });
    expect(button.classList.contains('listening')).toBe(false);
    expect(transcriptEl.textContent).toBe('добавь задачу чай');
  });

  test('disables the button when recognition is unsupported', () => {
    document.body.innerHTML = '<button id="voice-btn"></button><p id="voice-status"></p>';
    const button = document.getElementById('voice-btn');
    const statusEl = document.getElementById('voice-status');

    createVoiceUI({ button, statusEl, controller: { toggle: vi.fn() }, supported: false });
    expect(button.disabled).toBe(true);
    expect(statusEl.textContent).toContain('недоступен');
    expect(button.title).toContain('не поддерживается');
  });

  test('detects recognition support from the given scope', () => {
    expect(isSpeechRecognitionSupported({})).toBe(false);
    expect(isSpeechRecognitionSupported({ webkitSpeechRecognition: function Fake() {} })).toBe(true);
  });
});
