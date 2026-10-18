import { describe, expect, it } from 'vitest';
import { isPromptEcho, transcriptionPrompt, transcribeOpenAI } from '../../worker/v2/transcribe.js';

const task = (id, title, over = {}) => ({ id, parentId: null, status: 'Open', title, ...over });
const items = [
  task('inbox', 'Входящие'),
  task('tp', 'Сделать договор продажи квартиры'),
  task('tt', 'Написать предложение для парень', { parentId: 'tp' }),
  task('ts', 'Написать предложение для сестры', { parentId: 'tp' }),
  task('sa', 'Провести рисёрч по сауне'),
  task('old', 'Старая архивная', { status: 'Archive' })
];

describe('подсказка распознавания', () => {
  it('начинается с задачи в руках, её родителя и соседей, архив не берёт', () => {
    const prompt = transcriptionPrompt(items, 'tt');
    expect(prompt.indexOf('Написать предложение для парень')).toBeLessThan(prompt.indexOf('Сделать договор продажи квартиры'));
    expect(prompt).toContain('Написать предложение для сестры');
    expect(prompt).toContain('Провести рисёрч по сауне');
    expect(prompt).not.toContain('Старая архивная');
    expect(prompt).not.toContain('Входящие');
  });
  it('ограничена по длине', () => {
    const many = Array.from({ length: 500 }, (_, i) => task('t' + i, 'Очень длинное название задачи номер ' + i));
    expect(transcriptionPrompt(many, null).length).toBeLessThanOrEqual(800);
  });
});

describe('вызов расшифровки', () => {
  it('шлёт звук в gpt-4o-transcribe на русском с подсказкой и возвращает текст', async () => {
    let sent;
    const text = await transcribeOpenAI({
      apiKey: 'k', audio: new Uint8Array([1, 2, 3]).buffer, contentType: 'audio/webm;codecs=opus', prompt: 'Задачи: сауна',
      fetchImpl: async (url, options) => { sent = { url, options }; return { ok: true, json: async () => ({ text: ' переименуй в Вари ' }) }; }
    });
    expect(text).toBe('переименуй в Вари');
    expect(sent.url).toBe('https://api.openai.com/v1/audio/transcriptions');
    expect(sent.options.headers.Authorization).toBe('Bearer k');
    const form = sent.options.body;
    expect(form.get('model')).toBe('gpt-4o-transcribe');
    expect(form.get('language')).toBe('ru');
    expect(form.get('prompt')).toBe('Задачи: сауна');
    expect(form.get('file').name).toBe('speech.webm');
    expect(form.get('file').size).toBe(3);
  });
  it('звук Safari уходит как mp4, неизвестный формат и пустой ключ отклоняются', async () => {
    let name;
    await transcribeOpenAI({ apiKey: 'k', audio: new ArrayBuffer(1), contentType: 'audio/mp4', prompt: '',
      fetchImpl: async (url, options) => { name = options.body.get('file').name; return { ok: true, json: async () => ({ text: 'да' }) }; } });
    expect(name).toBe('speech.mp4');
    await expect(transcribeOpenAI({ apiKey: 'k', audio: new ArrayBuffer(1), contentType: 'text/plain', prompt: '' })).rejects.toThrow('Неподдерживаемый формат звука');
    await expect(transcribeOpenAI({ apiKey: '', audio: new ArrayBuffer(1), contentType: 'audio/webm', prompt: '' })).rejects.toThrow('ключ OpenAI');
  });
  it('отказ сервиса превращается в понятную ошибку', async () => {
    await expect(transcribeOpenAI({ apiKey: 'k', audio: new ArrayBuffer(1), contentType: 'audio/webm', prompt: '',
      fetchImpl: async () => ({ ok: false, status: 500, body: { cancel: async () => {} } }) })).rejects.toThrow('Распознавание недоступно (500)');
  });
});

describe('эхо подсказки', () => {
  const prompt = transcriptionPrompt(items, 'tt');
  it('на тишине модель повторяет подсказку — это пустой результат (случай с прода)', () => {
    expect(isPromptEcho(prompt, prompt)).toBe(true);
    expect(isPromptEcho('Написать предложение для парень. Написать предложение для сестры. Провести рисёрч по сауне', prompt)).toBe(true);
  });
  it('на тишине модель отвечает первой строкой подсказки — названием задачи в руках (случай с прода)', () => {
    expect(isPromptEcho('Написать предложение для парень', prompt, 'Написать предложение для парень')).toBe(true);
    expect(isPromptEcho('написать предложение для парень.', prompt, 'Написать предложение для парень')).toBe(true);
    expect(isPromptEcho('Написать предложение для сестры', prompt, 'Написать предложение для парень')).toBe(false);
  });
  it('сказанное название одной задачи или фраза с ним — не эхо', () => {
    expect(isPromptEcho('Провести рисёрч по сауне', prompt)).toBe(false);
    expect(isPromptEcho('переименуй в написать предложение для Вари', prompt)).toBe(false);
    expect(isPromptEcho('', prompt)).toBe(false);
  });
  it('вложенные и повторяющиеся названия, точка в названии — не ложное эхо', () => {
    const list = [task('a', 'Купить хлеб'), task('b', 'Купить хлеб и молоко'), task('c', 'Релиз v1.2 подготовить'), task('d', 'купить хлеб!')];
    const p = transcriptionPrompt(list, null);
    expect(isPromptEcho('Купить хлеб и молоко', p)).toBe(false);
    expect(isPromptEcho('купить хлеб', p)).toBe(false);
    expect(isPromptEcho('Релиз v1.2 подготовить', p)).toBe(false);
    expect(isPromptEcho('Купить хлеб и молоко. Релиз v1.2 подготовить', p)).toBe(true);
  });
});
