import { afterEach, describe, expect, it, vi } from 'vitest';
import { createLiveVoice } from '../../src/v2/client/live-voice.js';

class FakePeer {
  constructor() { this.listeners = {}; }
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
  emit(type, event) { for (const handler of this.listeners[type] || []) handler(event); }
  addTrack() {}
  createDataChannel() { return { addEventListener() {}, close() {} }; }
  async createOffer() { return { type: 'offer', sdp: 'offer' }; }
  async setLocalDescription() {}
  async setRemoteDescription() {}
  close() {}
}

function setup({ playRejects = false } = {}) {
  document.body.innerHTML = '<button id="voice"></button><div id="status"></div>';
  const order = [];
  const play = vi.fn(() => { order.push('play'); return playRejects ? Promise.reject(new Error('NotAllowedError')) : Promise.resolve(); });
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(play);
  const peer = new FakePeer();
  const media = { getTracks: () => [] };
  const voice = createLiveVoice({
    voiceButton: document.getElementById('voice'), voiceStatus: document.getElementById('status'),
    fetchImpl: async () => ({ ok: true, json: async () => ({ sessionId: 'live_1', sdp: 'answer' }) }),
    peerFactory: () => peer,
    mediaDevices: { getUserMedia: async () => { order.push('microphone'); return media; } }
  });
  return { voice, peer, play, order };
}

describe('GPT-Live playback', () => {
  afterEach(() => { vi.restoreAllMocks(); });

  it('starts playback inside the tap, before the microphone prompt, so Safari lets it sound', async () => {
    const { voice, order } = setup();
    await voice.start();
    expect(order.slice(0, 2)).toEqual(['play', 'microphone']);
    expect(document.querySelector('audio').playsInline).toBe(true);
  });

  it('plays the model\'s audio as soon as it arrives', async () => {
    const { voice, peer, play } = setup();
    await voice.start();
    const stream = ({ id: "remote" });
    peer.emit('track', { streams: [stream] });
    expect(document.querySelector('audio').srcObject).toBe(stream);
    expect(play).toHaveBeenCalledTimes(2);
  });

  it('does not cry wolf when only the early unlock is refused', async () => {
    const { voice } = setup({ playRejects: true });
    await voice.start();
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(document.getElementById('status').textContent).toBe('Слушаю');
  });

  it('says so when the browser keeps the sound off', async () => {
    const { voice, peer } = setup({ playRejects: true });
    await voice.start();
    peer.emit('track', { streams: [({ id: "remote" })] });
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(document.getElementById('status').textContent).toBe('Браузер не включил звук. Нажмите кнопку ещё раз.');
  });
});
