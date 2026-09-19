import { describe, expect, it } from 'vitest';
import { LiveSettings, promptVersion } from '../../src/v2/domain/live-settings.js';
import { DEFAULT_BACKEND_MODEL, DEFAULT_VOICE_PROMPT } from '../../src/v2/domain/live-session.js';
import { buildLogEntry } from '../../src/v2/domain/live-log-entry.js';

const memory = () => {
  const map = new Map();
  return { map, async get(key) { return map.get(key); }, async put(key, value) { map.set(key, value); } };
};

describe('live settings', () => {
  it('serves the built-in default while nothing is stored', async () => {
    const settings = new LiveSettings(memory());
    const read = await settings.read();
    expect(read.voicePrompt).toBe(DEFAULT_VOICE_PROMPT);
    expect(read.backendModel).toBe(DEFAULT_BACKEND_MODEL);
    expect(read.defaults).toEqual({ voicePrompt: true, backendPrompt: true, backendModel: true });
  });

  it('appends a rule without losing the text it had', async () => {
    const settings = new LiveSettings(memory());
    const result = await settings.writePrompt('voice', { mode: 'append', text: 'Отвечай короче.' });
    expect(result.prompt.startsWith(DEFAULT_VOICE_PROMPT)).toBe(true);
    expect(result.prompt.endsWith('Отвечай короче.')).toBe(true);
  });

  it('replaces the text wholesale when asked', async () => {
    const settings = new LiveSettings(memory());
    await settings.writePrompt('voice', { mode: 'replace', text: 'Совсем новый промпт.' });
    expect(await settings.prompt('voice')).toBe('Совсем новый промпт.');
  });

  it('keeps both texts of every edit so it can be read and undone', async () => {
    const settings = new LiveSettings(memory());
    await settings.writePrompt('voice', { mode: 'replace', text: 'Первая редакция.', source: 'model' });
    await settings.writePrompt('voice', { mode: 'replace', text: 'Вторая редакция.', source: 'model' });
    const history = await settings.history();
    expect(history).toHaveLength(2);
    expect(history[0]).toMatchObject({ target: 'voice', mode: 'replace', source: 'model', before: 'Первая редакция.', after: 'Вторая редакция.' });
    expect(history[1].before).toBe(DEFAULT_VOICE_PROMPT);
  });

  it('restores an earlier version', async () => {
    const settings = new LiveSettings(memory());
    await settings.writePrompt('voice', { mode: 'replace', text: 'Первая редакция.' });
    await settings.writePrompt('voice', { mode: 'replace', text: 'Испорченная редакция.' });
    const [latest] = await settings.history();
    await settings.restorePrompt('voice', latest.at);
    expect(await settings.prompt('voice')).toBe('Первая редакция.');
  });

  it('returns to the built-in default by clearing the field', async () => {
    const storage = memory();
    const settings = new LiveSettings(storage);
    await settings.writePrompt('voice', { mode: 'replace', text: 'Испорченная редакция.' });
    const reset = await settings.resetPrompt('voice');
    expect(reset.usingDefault).toBe(true);
    expect(await settings.prompt('voice')).toBe(DEFAULT_VOICE_PROMPT);
    expect((await settings.read()).defaults.voicePrompt).toBe(true);
  });

  it('rejects an oversized prompt and an implausible model name', async () => {
    const settings = new LiveSettings(memory());
    await expect(settings.writePrompt('voice', { mode: 'replace', text: 'a'.repeat(16_001) })).rejects.toThrow();
    await expect(settings.setBackendModel('не модель!')).rejects.toThrow();
    expect(await settings.setBackendModel('gpt-5.6-terra')).toEqual({ backendModel: 'gpt-5.6-terra', usingDefault: false });
  });

  it('marks a prompt version so an old log entry stays readable', () => {
    expect(promptVersion('один')).toBe(promptVersion('один'));
    expect(promptVersion('один')).not.toBe(promptVersion('другой'));
  });
});

describe('log entries', () => {
  it('indexes a delegation envelope by its inner type', () => {
    const row = buildLogEntry({
      type: 'response.event', delegation_id: 'item_1',
      event: { type: 'response.output_item.done', item: { type: 'function_call', call_id: 'call_1', name: 'setStatus' } }
    }, { sessionId: 'live_1' });
    expect(row.type).toBe('response.event/response.output_item.done');
    expect(row.delegationId).toBe('item_1');
    expect(row.callId).toBe('call_1');
    expect(row.direction).toBe('in');
  });

  it('carries the timeline offset of a delegation', () => {
    const row = buildLogEntry({ type: 'session.delegation.created', delegation: { id: 'item_2', target: 'responses', offset_ms: 8120 } }, { sessionId: 'live_1' });
    expect(row).toMatchObject({ type: 'session.delegation.created', delegationId: 'item_2', offsetMs: 8120 });
  });

  it('marks what we send ourselves as outgoing', () => {
    const row = buildLogEntry({ type: 'response.create' }, { sessionId: 'live_1', direction: 'out' });
    expect(row.direction).toBe('out');
  });

  it('says so in the row when a payload had to be cut', () => {
    const row = buildLogEntry({ type: 'session.started', blob: 'x'.repeat(70_000) }, { sessionId: 'live_1' });
    expect(JSON.parse(row.payload).truncated).toBe(true);
  });
});
