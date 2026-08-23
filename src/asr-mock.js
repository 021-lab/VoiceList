// asr-mock.js — мок захвата голоса и транскрибации.
// Реализует контракт ASR-адаптера из INPUT_HANDLING_STANDARD.md §2,
// но вместо микрофона проигрывает заранее заданную фразу.
// Тот же интерфейс, что у настоящего адаптера: три вида событий.

'use strict';

// failure: null | 'denied' | 'aborted' | 'timeout'
// interim: false — движок не отдаёт промежуточных результатов
//          (деградированный режим)
class MockASR {
  constructor({ phrase = '', failure = null, interim = true } = {}) {
    this.phrase = phrase;
    this.failure = failure;
    this.interim = interim;
    this.started = false;
  }

  // handlers: { onInterim(text), onFinal(text), onError(code) }
  start(handlers) {
    this.handlers = handlers;
    if (this.failure === 'denied') {
      handlers.onError('denied');
      return false;
    }
    this.started = true;
    return true;
  }

  // Проиграть речь: слово за словом, как отдаёт живой движок.
  speak() {
    if (!this.started) return;
    if (this.failure === 'aborted') {
      this.handlers.onError('aborted');
      this.started = false;
      return;
    }
    if (!this.interim) return;  // промежуточных нет — стек будет пуст
    const words = this.phrase.split(' ').filter(Boolean);
    const acc = [];
    for (const w of words) {
      acc.push(w);
      this.handlers.onInterim(acc.join(' '));
    }
  }

  // Отпускание пальца: движок отдаёт финал либо молчит до таймаута.
  stop() {
    if (!this.started) return;
    this.started = false;
    if (this.failure === 'timeout') {
      this.handlers.onError('timeout');
      return;
    }
    this.handlers.onFinal(this.phrase);
  }

  hasInterim() { return this.interim; }
}

export { MockASR };
