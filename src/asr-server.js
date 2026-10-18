// Server ASR adapter: records the held phrase and has the worker transcribe it.
// It exposes the same start/stop/hasInterim shape as BrowserASR. Decision: speech from a held
// task is recognised on the server (docs/DECISIONS.md).

'use strict';

const TYPES = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/webm', 'audio/ogg;codecs=opus'];
const SAMPLE_MS = 50;
const SPEECH_LEVEL = 0.02;   // RMS of a float sample; room noise stays well under it
const SPEECH_MS = 200;       // less voice than this is a held breath, not a phrase

/** RMS level of the microphone, or null when the browser will not run the audio graph
 *  (iOS keeps a context created outside a gesture suspended, and then it reads silence). */
function defaultMeter(stream, win = globalThis) {
  const Context = win.AudioContext || win.webkitAudioContext;
  if (!Context) return { level: () => null, close() {} };
  const context = new Context();
  context.resume?.().catch?.(() => {});
  const analyser = context.createAnalyser();
  analyser.fftSize = 1024;
  context.createMediaStreamSource(stream).connect(analyser);
  const samples = new Float32Array(analyser.fftSize);
  return {
    level() {
      if (context.state !== 'running') return null;
      analyser.getFloatTimeDomainData(samples);
      let sum = 0;
      for (const value of samples) sum += value * value;
      return Math.sqrt(sum / samples.length);
    },
    close() { context.close?.().catch?.(() => {}); }
  };
}

class ServerASR {
  static supported(win = globalThis) {
    return Boolean(win.MediaRecorder && win.navigator?.mediaDevices?.getUserMedia);
  }

  constructor({ target = null, endpoint = '/api/transcribe', mediaDevices = globalThis.navigator?.mediaDevices,
    Recorder = globalThis.MediaRecorder, meter = defaultMeter, fetchImpl = (...args) => globalThis.fetch(...args) } = {}) {
    Object.assign(this, { target, endpoint, mediaDevices, Recorder, meter, fetchImpl });
  }

  start(handlers) {
    this.handlers = handlers;
    this.chunks = [];
    this.speechMs = 0;
    this.measured = true;
    this.opening = this.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      this.stream = stream;
      try { return this.record(stream); } catch {
        clearInterval(this.timer);
        if (this.recorder?.state === 'recording') this.recorder.stop();
        for (const track of stream.getTracks()) track.stop();
        handlers.onError('microphone');
        return false;
      }
    }, (error) => {
      handlers.onError(error?.name === 'NotAllowedError' || error?.name === 'SecurityError' ? 'denied' : 'microphone');
      return false;
    });
    return true;
  }

  record(stream) {
    const mimeType = TYPES.find(type => this.Recorder.isTypeSupported?.(type)) || '';
    this.recorder = new this.Recorder(stream, mimeType ? { mimeType } : {});
    this.mimeType = mimeType || this.recorder.mimeType || 'audio/webm';
    this.recorder.ondataavailable = (event) => { if (event.data?.size) this.chunks.push(event.data); };
    this.recorder.start();
    this.level = this.meter(stream);
    this.timer = setInterval(() => {
      const level = this.level.level();
      if (level === null) this.measured = false;
      else if (level > SPEECH_LEVEL) this.speechMs += SAMPLE_MS;
    }, SAMPLE_MS);
    return true;
  }

  async stop() {
    if (!this.opening) return;
    const opened = await this.opening;
    this.opening = null;
    if (!opened) return;
    clearInterval(this.timer);
    await new Promise((resolve) => {
      if (this.recorder.state === 'inactive') { resolve(); return; }
      this.recorder.onstop = resolve;
      this.recorder.stop();
    });
    for (const track of this.stream.getTracks()) track.stop();
    this.level.close();
    if (this.measured && this.speechMs < SPEECH_MS) { this.handlers.onError('silence'); return; }
    const audio = new Blob(this.chunks, { type: this.mimeType });
    const query = this.target ? `?target=${encodeURIComponent(this.target)}` : '';
    try {
      const response = await this.fetchImpl(this.endpoint + query, { method: 'POST', headers: { 'Content-Type': this.mimeType }, body: audio });
      if (!response.ok) { this.handlers.onError('server'); return; }
      const text = String((await response.json()).text || '').trim();
      if (text) this.handlers.onFinal(text); else this.handlers.onError('silence');
    } catch {
      this.handlers.onError('network');
    }
  }

  hasInterim() { return false; }
}

export { ServerASR };
