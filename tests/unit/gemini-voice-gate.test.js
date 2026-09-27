import { describe, expect, it } from 'vitest';
import { modelSpeaking } from '../../src/v2/client/gemini-voice.js';

/** The streamer schedules the model's audio ahead on the output clock; scheduledTime is when
 *  the last buffer it holds finishes sounding. */
const streamer = (scheduledTime, currentTime) => ({ scheduledTime, context: { currentTime } });

describe('the microphone gate', () => {
  it('is closed while the model still has audio to play', () => {
    expect(modelSpeaking(streamer(12.4, 10))).toBe(true);
  });

  it('stays closed for a moment after the last sound, for the room to go quiet', () => {
    expect(modelSpeaking(streamer(10, 10.2))).toBe(true);
    expect(modelSpeaking(streamer(10, 10.8))).toBe(false);
  });

  it('is open before anything has been played and when there is no streamer', () => {
    expect(modelSpeaking(streamer(0, 30))).toBe(false);
    expect(modelSpeaking(null)).toBe(false);
    expect(modelSpeaking({})).toBe(false);
  });
});
