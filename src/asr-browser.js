// Browser ASR adapter for INPUT_HANDLING_STANDARD.md.
// It exposes the same start/stop/hasInterim shape as MockASR.

'use strict';

class BrowserASR {
  constructor({ recognitionFactory = null, finalTimeoutMs = 1500 } = {}) {
    this.recognitionFactory = recognitionFactory;
    this.finalTimeoutMs = finalTimeoutMs;
    this.finalText = '';
    this.interimText = '';
    this.started = false;
  }

  createRecognition() {
    if (this.recognitionFactory) return this.recognitionFactory();
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    return Recognition ? new Recognition() : null;
  }

  start(handlers) {
    this.handlers = handlers;
    const recognition = this.createRecognition();
    if (!recognition) {
      handlers.onError('denied');
      return false;
    }

    this.recognition = recognition;
    recognition.lang = 'ru-RU';
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.onresult = (event) => {
      let interim = '';
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const text = event.results[index][0]?.transcript?.trim() || '';
        if (!text) continue;
        if (event.results[index].isFinal) {
          this.finalText = `${this.finalText} ${text}`.trim();
          handlers.onFinal(this.finalText);
        } else {
          interim = `${interim} ${text}`.trim();
        }
      }
      if (interim) {
        this.interimText = interim;
        handlers.onInterim(interim);
      }
    };
    recognition.onerror = (event) => {
      const code = event.error === 'not-allowed' || event.error === 'service-not-allowed' ? 'denied' : 'aborted';
      handlers.onError(code);
    };

    try {
      recognition.start();
      this.started = true;
      return true;
    } catch {
      handlers.onError('denied');
      return false;
    }
  }

  stop() {
    if (!this.started) return Promise.resolve();
    this.started = false;

    return new Promise((resolve) => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve();
      };
      const timer = setTimeout(() => {
        if (!this.finalText) this.handlers.onError('timeout');
        finish();
      }, this.finalTimeoutMs);

      this.recognition.onend = finish;
      try {
        this.recognition.stop();
      } catch {
        finish();
      }
    });
  }

  hasInterim() { return true; }
}

export { BrowserASR };
