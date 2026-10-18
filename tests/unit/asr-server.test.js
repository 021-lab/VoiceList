import { afterEach, describe, expect, it, vi } from 'vitest';
import { ServerASR } from '../../src/asr-server.js';

class FakeRecorder {
  static isTypeSupported(type) { return type === 'audio/webm;codecs=opus'; }
  constructor(stream, { mimeType }) { this.mimeType = mimeType; FakeRecorder.last = this; }
  start() { this.state = 'recording'; }
  stop() { this.state = 'inactive'; this.ondataavailable({ data: new Blob([new Uint8Array([1, 2, 3])], { type: this.mimeType }) }); this.onstop(); }
}
const track = () => ({ stopped: false, stop() { this.stopped = true; } });
function setup({ levels = [0.1], getUserMedia, reply = { ok: true, json: async () => ({ text: 'переименуй в Вари' }) } } = {}) {
  const tracks = [track()];
  const stream = { getTracks: () => tracks };
  const posts = [];
  let i = 0;
  const asr = new ServerASR({
    target: 'tt', Recorder: FakeRecorder,
    mediaDevices: { getUserMedia: getUserMedia || (async () => stream) },
    meter: () => ({ level: () => levels[Math.min(i++, levels.length - 1)], close() {} }),
    fetchImpl: async (url, options) => { posts.push({ url, options }); return reply; }
  });
  const seen = { final: [], errors: [] };
  const handlers = { onInterim() {}, onFinal: (t) => seen.final.push(t), onError: (c) => seen.errors.push(c) };
  return { asr, handlers, seen, posts, tracks };
}
afterEach(() => vi.useRealTimers());

describe('серверное распознавание', () => {
  it('шлёт записанную речь на сервер с задачей в руках и отдаёт текст', async () => {
    vi.useFakeTimers();
    const { asr, handlers, seen, posts, tracks } = setup();
    asr.start(handlers); await vi.advanceTimersByTimeAsync(600);
    const stopping = asr.stop(); await vi.runAllTimersAsync(); await stopping;
    expect(posts).toHaveLength(1);
    expect(posts[0].url).toBe('/api/transcribe?target=tt');
    expect(posts[0].options.headers['Content-Type']).toBe('audio/webm;codecs=opus');
    expect(posts[0].options.body.size).toBe(3);
    expect(seen).toEqual({ final: ['переименуй в Вари'], errors: [] });
    expect(tracks[0].stopped).toBe(true);
  });
  it('тишину не отправляет: на ней модель выдумывает название задачи', async () => {
    vi.useFakeTimers();
    const { asr, handlers, seen, posts } = setup({ levels: [0.001] });
    asr.start(handlers); await vi.advanceTimersByTimeAsync(1000);
    const stopping = asr.stop(); await vi.runAllTimersAsync(); await stopping;
    expect(posts).toHaveLength(0); expect(seen.errors).toEqual(['silence']);
  });
  it('если громкость измерить нельзя, запись всё равно уходит', async () => {
    vi.useFakeTimers();
    const { asr, handlers, seen, posts } = setup({ levels: [null] });
    asr.start(handlers); await vi.advanceTimersByTimeAsync(600);
    const stopping = asr.stop(); await vi.runAllTimersAsync(); await stopping;
    expect(posts).toHaveLength(1); expect(seen.final).toEqual(['переименуй в Вари']);
  });
  it('запрет и неисправный микрофон, отказ сервера и пустой ответ дают свои коды', async () => {
    for (const [name, code] of [['NotAllowedError', 'denied'], ['NotFoundError', 'microphone'], ['NotReadableError', 'microphone']]) {
      const { asr, handlers, seen } = setup({ getUserMedia: async () => { throw Object.assign(new Error(name), { name }); } });
      asr.start(handlers); await asr.stop();
      expect(seen.errors).toEqual([code]);
    }
    for (const [reply, code] of [[{ ok: false, status: 400, json: async () => ({}) }, 'server'], [{ ok: true, json: async () => ({ text: '' }) }, 'silence']]) {
      vi.useFakeTimers();
      const { asr, handlers, seen } = setup({ reply });
      asr.start(handlers); await vi.advanceTimersByTimeAsync(600);
      const stopping = asr.stop(); await vi.runAllTimersAsync(); await stopping;
      expect(seen.errors).toEqual([code]);
      vi.useRealTimers();
    }
  });
  it('если запись не стартовала, микрофон закрывается и приходит ошибка', async () => {
    const { asr, handlers, seen, tracks } = setup();
    asr.Recorder = class { static isTypeSupported() { return false; } constructor() { throw new Error('no recorder'); } };
    asr.start(handlers); await asr.stop();
    expect(seen.errors).toEqual(['microphone']); expect(tracks[0].stopped).toBe(true);
  });
  it('доступен, только если браузер умеет записывать звук', () => {
    expect(ServerASR.supported({ MediaRecorder: FakeRecorder, navigator: { mediaDevices: { getUserMedia() {} } } })).toBe(true);
    expect(ServerASR.supported({ navigator: { mediaDevices: { getUserMedia() {} } } })).toBe(false);
    expect(ServerASR.supported({ MediaRecorder: FakeRecorder, navigator: {} })).toBe(false);
  });
});
