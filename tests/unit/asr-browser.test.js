import { expect, it } from 'vitest';
import { BrowserASR } from '../../src/asr-browser.js';

it('сообщает причину ошибки распознавателя своим кодом', () => {
  const cases = { 'not-allowed': 'denied', 'service-not-allowed': 'denied', 'audio-capture': 'microphone', network: 'network', 'no-speech': 'silence', aborted: 'aborted' };
  for (const [error, code] of Object.entries(cases)) {
    const recognition = { start() {}, stop() {} };
    const seen = [];
    new BrowserASR({ recognitionFactory: () => recognition }).start({ onInterim() {}, onFinal() {}, onError: (c) => seen.push(c) });
    recognition.onerror({ error });
    expect(seen).toEqual([code]);
  }
});
