import { describe, expect, it } from 'vitest';
import {
  DEFAULT_GEMINI_PROMPT, GEMINI_MODEL, buildClientSetup, buildGeminiSetup, buildTokenRequest,
  buildToolResponse, geminiFunctionDeclarations, isLoggableFrame, readToolCalls, stripAudio
} from '../../src/v2/domain/gemini-live.js';

const task = (id, parentId, status, line1) => ({ id, parentId, order: 10, status, line1, line2: '', collapsed: false, tags: [] });
const sample = [task('rs', null, 'Open', 'Яблоки'), task('rt', 'rs', 'Open', 'Голден')];

describe('gemini tool declarations', () => {
  const declarations = geminiFunctionDeclarations();
  const byName = new Map(declarations.map(entry => [entry.name, entry]));

  it('leaves prompt editing out and keeps the task tools', () => {
    expect(byName.has('setStatus')).toBe(true);
    expect(byName.has('getFrontier')).toBe(true);
    expect(byName.has('setVoicePrompt')).toBe(false);
    expect(byName.has('setBackendPrompt')).toBe(false);
  });

  it('blocks on a change and lets a read run alongside speech', () => {
    expect(byName.get('editItem').behavior).toBe('BLOCKING');
    expect(byName.get('getFrontier').behavior).toBe('NON_BLOCKING');
  });

  it('converts the schema to what the API takes: no additionalProperties, no union types', () => {
    const setParent = byName.get('setParent').parameters;
    expect('additionalProperties' in setParent).toBe(false);
    expect(setParent.properties.parentId).toEqual({ type: 'string', nullable: true, description: expect.any(String) });
    expect(setParent.required).toEqual(['taskId', 'parentId']);
  });
});

describe('the connect config', () => {
  it('carries the prompt and the task table, and asks for both transcripts', () => {
    const config = buildGeminiSetup({ items: sample });
    const instruction = config.systemInstruction.parts[0].text;
    expect(instruction).toContain('rt\trs\tO\tГолден');
    expect(instruction).toContain('Выбор задачи');
    // Recognition is pinned to one language: left open, whole turns came back as Spanish.
    expect(config.inputAudioTranscription).toEqual({ languageCodes: ['ru-RU'] });
    expect(config.outputAudioTranscription).toEqual({});
    expect(config.generationConfig.responseModalities).toEqual(['AUDIO']);
    expect(config.generationConfig.speechConfig.languageCode).toBe('ru-RU');
  });

  it('keeps a saved prompt instead of the built-in one', () => {
    const config = buildGeminiSetup({ items: sample, prompt: 'Своя редакция' });
    expect(config.systemInstruction.parts[0].text).toContain('Своя редакция');
    expect(config.systemInstruction.parts[0].text).not.toContain(DEFAULT_GEMINI_PROMPT);
  });
});

describe('the token request', () => {
  it('seals the session into the token, one use, with the start window inside the session', () => {
    const now = () => new Date('2026-09-24T10:00:00.000Z');
    const request = buildTokenRequest({ items: sample, now });
    expect(request.uses).toBe(1);
    // The API knows this field and not liveConnectConstraints; it answers "Cannot find field"
    // to the latter, so the name is pinned here rather than trusted to the guide.
    expect(Object.keys(request)).toContain('bidiGenerateContentSetup');
    expect(request.bidiGenerateContentSetup.model).toBe(`models/${GEMINI_MODEL}`);
    expect(request.bidiGenerateContentSetup.tools[0].functionDeclarations.length).toBeGreaterThan(0);
    expect(new Date(request.newSessionExpireTime) < new Date(request.expireTime)).toBe(true);
  });

  it('names the model the settings chose', () => {
    const request = buildTokenRequest({ items: sample, model: 'gemini-3.8-live-extended-thinking' });
    expect(request.bidiGenerateContentSetup.model).toBe('models/gemini-3.8-live-extended-thinking');
  });
});

describe('frames', () => {
  it('reads tool calls and answers each by its id, wrapped in a result', () => {
    const calls = readToolCalls({ toolCall: { functionCalls: [{ id: 'c1', name: 'setStatus', args: { taskId: 'rt', status: 'Focus' } }] } });
    expect(calls).toEqual([{ id: 'c1', name: 'setStatus', arguments: { taskId: 'rt', status: 'Focus' } }]);
    const answer = buildToolResponse([{ id: 'c1', name: 'setStatus', response: { status: 'applied' } }]);
    expect(answer.toolResponse.functionResponses[0]).toEqual({ id: 'c1', name: 'setStatus', response: { result: { status: 'applied' } } });
  });

  it('ignores a frame that carries no calls', () => {
    expect(readToolCalls({ serverContent: {} })).toEqual([]);
  });

  it('keeps what can be read and drops the audio', () => {
    expect(isLoggableFrame({ serverContent: { modelTurn: { parts: [{ inlineData: { data: 'AAAA' } }] } } })).toBe(false);
    // The reply's transcript rides along with its audio; dropping the frame lost every answer.
    const spoken = { serverContent: { modelTurn: { parts: [{ inlineData: { data: 'AAAA' } }] }, outputTranscription: { text: 'Готово' } } };
    expect(isLoggableFrame(spoken)).toBe(true);
    expect(stripAudio(spoken).serverContent.modelTurn.parts).toEqual([]);
    expect(stripAudio(spoken).serverContent.outputTranscription.text).toBe('Готово');
    expect(isLoggableFrame({ serverContent: { outputTranscription: { text: 'Готово' } } })).toBe(true);
    expect(isLoggableFrame({ toolCall: { functionCalls: [] } })).toBe(true);
  });

  it('names the model in the first client frame', () => {
    expect(buildClientSetup('gemini-3.8-live')).toEqual({ setup: { model: 'models/gemini-3.8-live' } });
  });
});
