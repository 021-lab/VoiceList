export const LIVE_SESSION_ENDPOINT = '/api/live/session';
export const LIVE_SESSION_STOP_ENDPOINT = '/api/live/session/stop';
export const LIVE_KEY_STATUS_ENDPOINT = '/api/live/key/status';
export const LIVE_KEY_ENDPOINT = '/api/live/key';
export const LIVE_SETTINGS_ENDPOINT = '/api/live/settings';

const MAX_VISIBLE_TURNS = 240;

/** Browser side of the GPT-Live voice button.
 *
 *  Deliberately thin: audio runs straight between the browser and the model, while tool
 *  calls, the log and the task snapshot all live on the server behind the sideband
 *  connection. The page only opens the media path and reflects what it hears. */
export function createLiveVoice({
  voiceButton, voiceStatus, dialoguesButton, dialoguesPanel, dialoguesList, dialoguesClose,
  rootPanel, onCloseDialogues, settingsElements = {}, fetchImpl = fetch,
  documentRef = document, windowRef = window,
  peerFactory = (config) => new RTCPeerConnection(config), mediaDevices = navigator.mediaDevices
} = {}) {
  let session = null;
  let turns = [];

  const setStatus = (text, state) => {
    if (voiceStatus) { voiceStatus.textContent = text || ''; voiceStatus.hidden = !text; }
    if (voiceButton) { voiceButton.dataset.state = state; voiceButton.setAttribute('aria-pressed', String(state === 'live')); }
  };

  function renderTurns() {
    if (!dialoguesList) return;
    dialoguesList.textContent = '';
    if (!turns.length) {
      const empty = documentRef.createElement('p');
      empty.className = 'dialogues-empty';
      empty.textContent = 'Реплики появятся во время разговора. Полная запись хранится на сервере.';
      dialoguesList.append(empty);
      return;
    }
    for (const turn of turns) {
      const row = documentRef.createElement('div');
      row.className = `dialogue-message dialogue-${turn.role}`;
      row.textContent = turn.text;
      dialoguesList.append(row);
    }
    dialoguesList.scrollTop = dialoguesList.scrollHeight;
  }

  /** Transcript fragments follow audio cadence, not turns, so they are stitched by speaker
   *  until the other side starts. The authoritative record is the server's log. */
  function appendFragment(role, delta) {
    const text = String(delta || '');
    if (!text.trim()) return;
    const last = turns.at(-1);
    if (last && last.role === role) last.text = `${last.text}${text}`;
    else turns = [...turns, { role, text }].slice(-MAX_VISIBLE_TURNS);
    renderTurns();
  }

  function handleEvent(raw) {
    let event;
    try { event = JSON.parse(raw); } catch { return; }
    if (event.type === 'session.input_transcript.delta') appendFragment('user', event.delta);
    else if (event.type === 'session.output_transcript.delta') appendFragment('assistant', event.delta);
    else if (event.type === 'session.started') setStatus('Слушаю', 'live');
    else if (event.type === 'session.closed') stop();
  }

  async function start() {
    if (session) return;
    setStatus('Подключение…', 'connecting');
    let media;
    try { media = await mediaDevices.getUserMedia({ audio: true }); }
    catch { setStatus('Нет доступа к микрофону', 'idle'); return; }

    const peer = peerFactory({});
    const audio = documentRef.createElement('audio');
    audio.autoplay = true;
    audio.hidden = true;
    documentRef.body.append(audio);
    session = { peer, media, audio, channel: null };

    peer.addEventListener('track', event => { audio.srcObject = event.streams[0]; });
    for (const track of media.getTracks()) peer.addTrack(track, media);
    const channel = peer.createDataChannel('oai-events');
    session.channel = channel;
    channel.addEventListener('message', event => handleEvent(event.data));
    channel.addEventListener('close', () => stop());

    try {
      const offer = await peer.createOffer();
      await peer.setLocalDescription(offer);
      const response = await fetchImpl(LIVE_SESSION_ENDPOINT, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sdp: offer.sdp })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.error?.message || 'Не удалось начать сессию');
      await peer.setRemoteDescription({ type: 'answer', sdp: payload.sdp });
      session.id = payload.sessionId;
      setStatus('Слушаю', 'live');
    } catch (error) {
      teardown();
      setStatus(error.message || 'Не удалось начать сессию', 'idle');
      return;
    }
    turns = [];
    renderTurns();
  }

  function teardown() {
    if (!session) return;
    try { session.channel?.close(); } catch { /* already gone */ }
    try { session.peer.close(); } catch { /* already gone */ }
    for (const track of session.media?.getTracks() || []) { try { track.stop(); } catch { /* already gone */ } }
    try { session.audio.remove(); } catch { /* already gone */ }
    session = null;
  }

  function stop() {
    if (!session) return;
    const wasLive = Boolean(session.id);
    teardown();
    setStatus('', 'idle');
    // Closing the session server side also releases the sideband and finalises the recording.
    if (wasLive) fetchImpl(LIVE_SESSION_STOP_ENDPOINT, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }).catch(() => {});
  }

  function toggle() { if (session) stop(); else start(); }

  // The view itself is owned by the client's navigation; this only fills and reveals it.
  function openDialogues() {
    if (!dialoguesPanel) return;
    dialoguesPanel.hidden = false;
    if (rootPanel) rootPanel.dataset.dialoguesOpen = 'true';
    dialoguesButton?.classList.add('active');
    dialoguesButton?.setAttribute('aria-pressed', 'true');
    renderTurns();
  }
  function closeDialogues() {
    if (!dialoguesPanel) return;
    dialoguesPanel.hidden = true;
    if (rootPanel) rootPanel.dataset.dialoguesOpen = 'false';
    dialoguesButton?.classList.remove('active');
    dialoguesButton?.setAttribute('aria-pressed', 'false');
  }

  voiceButton?.addEventListener('click', toggle);
  dialoguesClose?.addEventListener('click', () => { closeDialogues(); onCloseDialogues?.(); });
  // A backgrounded mobile tab can otherwise keep the microphone and the billing clock running.
  documentRef.addEventListener('visibilitychange', () => { if (documentRef.visibilityState === 'hidden') stop(); });
  windowRef.addEventListener('pagehide', () => stop());

  const settings = createSettingsPanel({ ...settingsElements, fetchImpl });
  return { start, stop, toggle, openDialogues, closeDialogues, settings, get active() { return Boolean(session); }, get turns() { return turns.slice(); } };
}

/** Prompts and the backend model are edited here and stored on the server; the task table is
 *  appended by the server at session start and is deliberately not part of the editable text. */
export function createSettingsPanel({
  keyInput, keyField, keySave, keyStatus,
  voicePromptInput, voicePromptSave, voicePromptReset, voicePromptStatus,
  backendPromptInput, backendPromptSave, backendPromptReset, backendPromptStatus,
  backendModelInput, backendModelSave, backendModelStatus,
  fetchImpl = fetch
} = {}) {
  const say = (element, text) => { if (element) element.textContent = text; };

  async function call(url, options) {
    const response = await fetchImpl(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload?.error?.message || payload?.error || 'Не удалось сохранить');
    return payload;
  }

  async function load() {
    try {
      const [key, settings] = await Promise.all([
        call(LIVE_KEY_STATUS_ENDPOINT, {}), call(LIVE_SETTINGS_ENDPOINT, {})
      ]);
      if (keyField) keyField.hidden = key.configured;
      if (keySave) keySave.hidden = key.configured;
      say(keyStatus, key.configured ? 'Ключ настроен.' : 'Ключ не настроен.');
      if (voicePromptInput) voicePromptInput.value = settings.voicePrompt;
      if (backendPromptInput) backendPromptInput.value = settings.backendPrompt;
      if (backendModelInput) backendModelInput.value = settings.backendModel;
      say(voicePromptStatus, settings.defaults.voicePrompt ? 'Используется встроенный промпт.' : 'Сохранена своя редакция.');
      say(backendPromptStatus, settings.defaults.backendPrompt ? 'Используется встроенный промпт.' : 'Сохранена своя редакция.');
      say(backendModelStatus, settings.defaults.backendModel ? 'Используется модель по умолчанию.' : 'Задана своя модель.');
      return settings;
    } catch (error) { say(keyStatus, error.message); return null; }
  }

  const put = (body) => call(LIVE_SETTINGS_ENDPOINT, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

  async function savePrompt(target, input, status) {
    try {
      await put({ target, mode: 'replace', text: input?.value || '' });
      say(status, 'Сохранено. Применится со следующей сессии.');
    } catch (error) { say(status, error.message); }
  }
  async function resetPrompt(target, input, status) {
    try {
      const result = await put({ target, action: 'reset' });
      if (input) input.value = result.prompt;
      say(status, 'Возвращён встроенный промпт.');
    } catch (error) { say(status, error.message); }
  }

  keySave?.addEventListener('click', async () => {
    try {
      await call(LIVE_KEY_ENDPOINT, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ apiKey: keyInput?.value || '' }) });
      say(keyStatus, 'Ключ сохранён.');
      if (keyInput) keyInput.value = '';
      await load();
    } catch (error) { say(keyStatus, error.message); }
  });
  voicePromptSave?.addEventListener('click', () => savePrompt('voice', voicePromptInput, voicePromptStatus));
  voicePromptReset?.addEventListener('click', () => resetPrompt('voice', voicePromptInput, voicePromptStatus));
  backendPromptSave?.addEventListener('click', () => savePrompt('backend', backendPromptInput, backendPromptStatus));
  backendPromptReset?.addEventListener('click', () => resetPrompt('backend', backendPromptInput, backendPromptStatus));
  backendModelSave?.addEventListener('click', async () => {
    try {
      const result = await put({ backendModel: backendModelInput?.value || '' });
      say(backendModelStatus, result.usingDefault ? 'Используется модель по умолчанию.' : `Модель: ${result.backendModel}. Применится со следующей сессии.`);
    } catch (error) { say(backendModelStatus, error.message); }
  });

  return { load };
}
