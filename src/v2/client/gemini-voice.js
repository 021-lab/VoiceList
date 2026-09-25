/** Gemini Live in the browser: the page holds the socket, the worker holds everything else.
 *
 *  The page is deliberately thin. It does not know the prompt, the task table or the tool
 *  list — all three are sealed into the ephemeral token the worker mints — and it does not
 *  decide what a tool call does: it relays the call to the worker and sends back the answer
 *  it gets. What it owns is the microphone, the speaker and the socket.
 *
 *  Audio capture and playback come from Google's own Live API console (see vendor/). */
import { AudioRecorder } from './vendor/live-api-web-console/audio-recorder.js';
import { AudioStreamer } from './vendor/live-api-web-console/audio-streamer.js';
import { base64ToArrayBuffer } from './vendor/live-api-web-console/audio-context.js';
// The protocol module only: importing the session module would pull the server domain, and
// zod with it, into the page.
import {
  GEMINI_WS_URL, GEMINI_INPUT_SAMPLE_RATE, GEMINI_OUTPUT_SAMPLE_RATE,
  buildClientSetup, buildToolResponse, stripAudio
} from '../domain/gemini-protocol.js';

/** Frames are mirrored to the log in batches: one request per frame would outnumber the
 *  conversation itself. */
const MIRROR_INTERVAL_MS = 2_000;
const MIRROR_LIMIT = 100;

/** The output context, made synchronously and kept.
 *
 *  The vendored helper probes autoplay with an Audio element and awaits it, which spends the
 *  user's gesture before the context exists — the one thing Safari will not forgive. Browsers
 *  also cap how many contexts a page may open, so it is created once and reused. */
const contexts = { input: null, output: null };
function openContext(which, sampleRate) {
  const Ctor = window.AudioContext || window.webkitAudioContext;
  if (!Ctor) throw new Error('Браузер не поддерживает Web Audio.');
  if (!contexts[which] || contexts[which].state === 'closed') {
    try {
      contexts[which] = new Ctor({ sampleRate });
    } catch (error) {
      throw new Error(`Браузер не открыл аудио на ${sampleRate} Гц: ${error?.message || 'отказ'}`);
    }
  }
  return contexts[which];
}

/** Asks for the microphone here rather than inside the recorder, so the browser's own reason
 *  survives. The two that actually happen are a refusal and an embedded web view: an in-app
 *  browser opened from a messenger usually has no microphone at all, and the page cannot tell
 *  the user that unless it looks. */
async function requestMicrophone() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Этот браузер не даёт странице микрофон. Откройте адрес в Safari или Chrome, а не внутри другого приложения.');
  }
  try {
    return await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    const name = error?.name || '';
    if (name === 'NotAllowedError' || name === 'SecurityError') {
      throw new Error('Доступ к микрофону запрещён. Разрешите его для сайта, либо откройте адрес в Safari или Chrome, а не внутри другого приложения.');
    }
    if (name === 'NotFoundError' || name === 'OverconstrainedError') throw new Error('Микрофон не найден.');
    if (name === 'NotReadableError') throw new Error('Микрофон занят другим приложением.');
    throw new Error(`Микрофон недоступен: ${error?.message || name || 'причина неизвестна'}`);
  }
}

const audioPartsOf = (frame) =>
  (frame?.serverContent?.modelTurn?.parts || [])
    .map(part => part?.inlineData?.data)
    .filter(Boolean);

export function createGeminiVoice({
  button, status, fetchImpl = (...args) => fetch(...args), WebSocketCtor = WebSocket, onTranscript = () => {}
} = {}) {
  let session = null;

  const setStatus = (text, state = '') => {
    if (!status) return;
    status.textContent = text || '';
    status.hidden = !text;
    status.dataset.state = state;
  };

  async function post(path, body) {
    const response = await fetchImpl(path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {})
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error?.message || payload.error || `HTTP ${response.status}`);
    return payload;
  }

  /** Everything the page saw, reported for the log. It is a report and not a record: only
   *  what went through the worker actually changed anything. */
  function mirror(current) {
    if (!current.frames.length) return;
    const frames = current.frames.splice(0, MIRROR_LIMIT);
    post('/api/live/gemini/frames', { frames }).catch(() => {});
  }

  function remember(current, frame, direction) {
    current.frames.push({ direction, frame: stripAudio(frame) });
    if (current.frames.length >= MIRROR_LIMIT) mirror(current);
  }

  function send(current, message, { log = true } = {}) {
    if (current.socket?.readyState !== WebSocket.OPEN) return false;
    current.socket.send(JSON.stringify(message));
    if (log) remember(current, message, 'out');
    return true;
  }

  async function handleFrame(current, frame) {
    if (frame.setupComplete) {
      current.ready = true;
      clearTimeout(current.watchdog);
      setStatus('Слушаю', 'active');
      return;
    }

    const audio = audioPartsOf(frame);
    for (const chunk of audio) current.streamer.addPCM16(new Uint8Array(base64ToArrayBuffer(chunk)));

    const input = frame.serverContent?.inputTranscription?.text;
    const output = frame.serverContent?.outputTranscription?.text;
    if (input) onTranscript({ role: 'user', text: input });
    if (output) onTranscript({ role: 'assistant', text: output });

    // The model stopped because the user spoke over it; what is already queued is stale.
    if (frame.serverContent?.interrupted) current.streamer.stop();

    if (frame.toolCall) {
      const { results } = await post('/api/live/gemini/tools', frame);
      send(current, buildToolResponse(results));
    }

    // The session is ending on the server's terms; say so rather than dying silently.
    if (frame.goAway) setStatus('Сессия истекает, начните заново', 'warn');
  }

  /** A promise that cannot hang forever. Two of the calls below are known to never settle
   *  when the browser decides audio is not allowed, and a hung promise looks exactly like a
   *  slow network from the outside. */
  const within = (promise, ms, message) => Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(message)), ms))
  ]);

  /** A failure the page cannot show to itself: put it in the status line and in the server
   *  log, so it is readable without a browser console. */
  function report(current, message) {
    setStatus(message, 'error');
    try { post('/api/live/gemini/frames', { frames: [{ direction: 'out', frame: { clientError: message } }] }).catch(() => {}); } catch { /* nothing left to do */ }
    if (session === current) stop('error');
  }

  async function start() {
    if (session) return;
    const current = { socket: null, recorder: null, streamer: null, frames: [], timer: null, watchdog: null, ready: false };
    session = current;

    try {
      // Audio first, and before any await on the network.
      //
      // Safari and iOS only let a page open an AudioContext or take the microphone while the
      // user's gesture is still in effect. Asking the worker for a token first spends that
      // gesture on a network round trip, and resume() then never settles — no error, no
      // sound, the status line stuck on "connecting" forever.
      // Both contexts are opened and started here, before anything that can take time.
      // Asking for the microphone shows a permission sheet, and after it the gesture is
      // spent: a context resumed later stays suspended, and loading a worklet into a
      // suspended context never finishes on iOS.
      setStatus('Готовлю звук…');
      const input = openContext('input', GEMINI_INPUT_SAMPLE_RATE);
      current.streamer = new AudioStreamer(openContext('output', GEMINI_OUTPUT_SAMPLE_RATE));
      await within(current.streamer.resume(), 5_000, 'Браузер не включил воспроизведение. Нажмите кнопку ещё раз.');
      await within(input.resume(), 5_000, 'Браузер не включил запись. Нажмите кнопку ещё раз.');

      setStatus('Микрофон…');
      const stream = await within(requestMicrophone(), 30_000, 'Браузер не ответил на запрос микрофона.');
      current.recorder = new AudioRecorder(GEMINI_INPUT_SAMPLE_RATE);
      current.recorder.onData = (data) => {
        // Audio frames are the bulk of the traffic and say nothing a transcript does not.
        // Before the socket is open they are dropped: a second of lost silence costs nothing.
        send(current, { realtimeInput: { audio: { data, mimeType: `audio/pcm;rate=${GEMINI_INPUT_SAMPLE_RATE}` } } }, { log: false });
      };
      setStatus('Запускаю обработку…');
      await within(current.recorder.start(stream, input), 15_000,
        `Обработка звука не запустилась (контекст ${input.state}, ${Math.round(input.sampleRate)} Гц).`);

      setStatus('Открываю сессию…');
      const { token, model } = await post('/api/live/gemini/session');

      setStatus('Соединяюсь с Gemini…');
      current.socket = new WebSocketCtor(`${GEMINI_WS_URL}?access_token=${encodeURIComponent(token)}`);
      current.socket.addEventListener('open', () => send(current, buildClientSetup(model)));
      current.socket.addEventListener('message', async (event) => {
        const raw = typeof event.data === 'string' ? event.data : await event.data.text();
        let frame;
        try { frame = JSON.parse(raw); } catch { return; }
        remember(current, frame, 'in');
        try { await handleFrame(current, frame); }
        catch (error) { report(current, error.message || 'Ошибка обработки ответа'); }
      });
      // A socket that opens and then says nothing is the failure that used to be invisible.
      current.watchdog = setTimeout(() => {
        if (session === current && !current.ready) report(current, 'Gemini не ответил на настройку сессии.');
      }, 15_000);
      current.socket.addEventListener('close', (event) => {
        if (session !== current) return;
        if (current.ready) { stop('closed'); return; }
        report(current, `Gemini закрыл соединение до начала разговора (код ${event.code}${event.reason ? `: ${event.reason}` : ''}).`);
      });
      current.socket.addEventListener('error', () => {
        if (session === current && !current.ready) report(current, 'Не удалось соединиться с Gemini.');
      });

      current.timer = setInterval(() => mirror(current), MIRROR_INTERVAL_MS);
      if (button) button.dataset.active = 'true';
    } catch (error) {
      report(current, error.message || 'Не удалось начать разговор');
    }
  }

  function stop(reason = 'client') {
    const current = session;
    session = null;
    if (!current) return;
    clearInterval(current.timer);
    clearTimeout(current.watchdog);
    current.recorder?.stop();
    current.streamer?.stop();
    try { current.socket?.close(); } catch { /* already gone */ }
    mirror(current);
    post('/api/live/gemini/session/stop', { reason }).catch(() => {});
    if (button) delete button.dataset.active;
    // An error already put its own text in the status line; do not talk over it.
    if (reason !== 'error') setStatus(reason === 'client' ? '' : 'Разговор завершён');
  }

  button?.addEventListener('click', () => (session ? stop() : start()));

  return { start, stop, get active() { return Boolean(session); } };
}
