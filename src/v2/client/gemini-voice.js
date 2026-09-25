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
import { audioContext, base64ToArrayBuffer } from './vendor/live-api-web-console/audio-context.js';
// The protocol module only: importing the session module would pull the server domain, and
// zod with it, into the page.
import {
  GEMINI_WS_URL, GEMINI_INPUT_SAMPLE_RATE, GEMINI_OUTPUT_SAMPLE_RATE,
  buildClientSetup, buildToolResponse
} from '../domain/gemini-protocol.js';

/** Frames are mirrored to the log in batches: one request per frame would outnumber the
 *  conversation itself. */
const MIRROR_INTERVAL_MS = 2_000;
const MIRROR_LIMIT = 100;

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
    current.frames.push({ direction, frame });
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
      setStatus('Слушаю', 'active');
      await current.recorder.start();
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

  async function start() {
    if (session) return;
    setStatus('Подключаюсь…');
    const current = { socket: null, recorder: null, streamer: null, frames: [], timer: null };
    session = current;

    try {
      const { token, model } = await post('/api/live/gemini/session');
      const context = await audioContext({ id: 'gemini-output', sampleRate: GEMINI_OUTPUT_SAMPLE_RATE });
      current.streamer = new AudioStreamer(context);
      await current.streamer.resume();

      current.recorder = new AudioRecorder(GEMINI_INPUT_SAMPLE_RATE);
      current.recorder.onData = (data) => {
        // Audio frames are the bulk of the traffic and say nothing a transcript does not.
        send(current, { realtimeInput: { audio: { data, mimeType: `audio/pcm;rate=${GEMINI_INPUT_SAMPLE_RATE}` } } }, { log: false });
      };

      current.socket = new WebSocketCtor(`${GEMINI_WS_URL}?access_token=${encodeURIComponent(token)}`);
      current.socket.addEventListener('open', () => send(current, buildClientSetup(model)));
      current.socket.addEventListener('message', async (event) => {
        const raw = typeof event.data === 'string' ? event.data : await event.data.text();
        let frame;
        try { frame = JSON.parse(raw); } catch { return; }
        remember(current, frame, 'in');
        try { await handleFrame(current, frame); }
        catch (error) { setStatus(error.message || 'Ошибка обработки', 'error'); }
      });
      current.socket.addEventListener('close', () => { if (session === current) stop('closed'); });
      current.socket.addEventListener('error', () => setStatus('Соединение с Gemini оборвалось', 'error'));

      current.timer = setInterval(() => mirror(current), MIRROR_INTERVAL_MS);
      if (button) button.dataset.active = 'true';
    } catch (error) {
      session = null;
      setStatus(error.message || 'Не удалось начать разговор', 'error');
      if (button) delete button.dataset.active;
    }
  }

  function stop(reason = 'client') {
    const current = session;
    session = null;
    if (!current) return;
    clearInterval(current.timer);
    current.recorder?.stop();
    current.streamer?.stop();
    try { current.socket?.close(); } catch { /* already gone */ }
    mirror(current);
    post('/api/live/gemini/session/stop', { reason }).catch(() => {});
    if (button) delete button.dataset.active;
    setStatus(reason === 'client' ? '' : 'Разговор завершён');
  }

  button?.addEventListener('click', () => (session ? stop() : start()));

  return { start, stop, get active() { return Boolean(session); } };
}
