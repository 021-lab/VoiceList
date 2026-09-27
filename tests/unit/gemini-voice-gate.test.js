import { describe, expect, it } from 'vitest';
import { modelSpeaking } from '../../src/v2/client/gemini-voice.js';

/** The streamer schedules the model's audio ahead on the output clock; scheduledTime is when
 *  the last buffer it holds finishes sounding. */
const streamer = (scheduledTime, currentTime) => ({ scheduledTime, context: { currentTime } });
const idle = { active: false, lastAudioAt: 0 };

describe('the microphone gate', () => {
  it('is closed while the model still has audio to play', () => {
    expect(modelSpeaking(streamer(12.4, 10), idle)).toBe(true);
  });

  it('stays closed for a moment after the last sound, for the room to go quiet', () => {
    expect(modelSpeaking(streamer(10, 10.2), idle)).toBe(true);
    expect(modelSpeaking(streamer(10, 10.8), idle)).toBe(false);
  });

  it('stays closed in the pause between two bursts of one turn', () => {
    // Live: the model said «Статус задачи », paused, and its own «Забронировать» came back
    // as «Abono», after which it started the sentence over.
    const drained = streamer(10, 11);
    expect(modelSpeaking(drained, { active: true, lastAudioAt: 1_000 }, 1_900)).toBe(true);
    expect(modelSpeaking(drained, { active: false, lastAudioAt: 1_000 }, 1_900)).toBe(false);
  });

  it('opens again when a turn goes quiet for long enough, even if its end never arrived', () => {
    expect(modelSpeaking(streamer(10, 11), { active: true, lastAudioAt: 1_000 }, 3_100)).toBe(false);
  });

  it('is open before anything has been played and when there is no streamer', () => {
    expect(modelSpeaking(streamer(0, 30), idle)).toBe(false);
    expect(modelSpeaking(null, idle)).toBe(false);
    expect(modelSpeaking({}, idle)).toBe(false);
  });
});
