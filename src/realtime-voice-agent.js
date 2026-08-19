export const DIALOGUES_STORAGE_KEY = 'voicelist.realtime-dialogues.v1';
export const OPENAI_KEY_STATUS_ENDPOINT = '/api/realtime/key/status';
export const OPENAI_KEY_SETUP_ENDPOINT = '/api/realtime/key';

const MAX_DIALOGUES = 30;
const MAX_MESSAGES_PER_DIALOGUE = 240;
const TOOL_RESULT_REPLY_INSTRUCTIONS = [
  'The task operation result is now available in the function_call_output.',
  'Do not continue any previous sentence.',
  'Answer only from that result, in one short Russian sentence.'
].join(' ');
const ALLOWED_STATUSES = new Set(['Open', 'Focus', 'Pause', 'Done', 'Archive', 'Info']);

function parseStoredDialogues(storage) {
  try {
    const value = JSON.parse(storage.getItem(DIALOGUES_STORAGE_KEY) || '[]');
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

function isoNow(now) {
  const value = now();
  return (value instanceof Date ? value : new Date(value)).toISOString();
}

function persistDialogues(storage, dialogues) {
  storage.setItem(DIALOGUES_STORAGE_KEY, JSON.stringify(dialogues.slice(0, MAX_DIALOGUES)));
}

export function createDialogueRepository({
  storage,
  now = () => new Date(),
  createId = () => crypto.randomUUID()
}) {
  let dialogues = parseStoredDialogues(storage);

  function update(id, updater) {
    dialogues = dialogues.map((dialogue) => dialogue.id === id ? updater(dialogue) : dialogue);
    persistDialogues(storage, dialogues);
  }

  return {
    start() {
      const dialogue = {
        id: createId(),
        startedAt: isoNow(now),
        endedAt: null,
        messages: []
      };
      dialogues = [dialogue, ...dialogues].slice(0, MAX_DIALOGUES);
      persistDialogues(storage, dialogues);
      return dialogue.id;
    },
    append(dialogueId, { role, text, eventId = null }) {
      const cleanText = String(text || '').trim();
      if (!cleanText) return;
      const messageAt = isoNow(now);
      update(dialogueId, (dialogue) => {
        if (eventId && dialogue.messages.some((message) => message.eventId === eventId)) return dialogue;
        const lastMessage = dialogue.messages.at(-1);
        if (lastMessage?.role === role && lastMessage.text === cleanText &&
            Math.abs(new Date(lastMessage.at).getTime() - new Date(messageAt).getTime()) < 2_000) {
          return dialogue;
        }
        return {
          ...dialogue,
          messages: [...dialogue.messages, {
            id: createId(),
            eventId,
            at: messageAt,
            role,
            text: cleanText
          }].slice(-MAX_MESSAGES_PER_DIALOGUE)
        };
      });
    },
    finish(dialogueId) {
      update(dialogueId, (dialogue) => ({ ...dialogue, endedAt: dialogue.endedAt || isoNow(now) }));
    },
    list() {
      return structuredClone(dialogues);
    }
  };
}

export function taskContextFromState(state) {
  const items = state?.snapshot?.items;
  if (!Array.isArray(items)) return [];
  return items.map((item) => ({
    id: item.id,
    parentId: item.parentId ?? null,
    order: item.order ?? 0,
    status: item.status || 'Open',
    line1: item.line1 || '',
    line2: item.line2 || '',
    tags: Array.isArray(item.tags) ? item.tags : []
  }));
}

export function taskTreeFromState(state) {
  const items = state?.snapshot?.items;
  if (!Array.isArray(items)) return [];

  const nodes = new Map();
  for (const item of items) {
    const id = String(item?.id || '');
    if (!id) continue;
    nodes.set(id, {
      id,
      title: String(item?.line1 || ''),
      status: String(item?.status || 'Open'),
      children: []
    });
  }

  const roots = [];
  for (const item of items) {
    const id = String(item?.id || '');
    const node = nodes.get(id);
    if (!node) continue;
    const parent = item?.parentId == null ? null : nodes.get(String(item.parentId));
    if (parent) parent.children.push(node);
    else roots.push(node);
  }
  return roots;
}

function requireText(value, label) {
  const text = String(value || '').trim();
  if (!text) throw new Error(`${label} is required`);
  return text;
}

function withTranscript(input, transcript) {
  const cleanTranscript = String(transcript || '').trim();
  return cleanTranscript ? { ...input, transcript: cleanTranscript } : input;
}

export function taskInputFromToolCall(name, rawArguments, { transcript = '' } = {}) {
  let args;
  try {
    args = typeof rawArguments === 'string' ? JSON.parse(rawArguments || '{}') : (rawArguments || {});
  } catch {
    throw new Error('Tool arguments are not valid JSON');
  }

  if (name === 'addItem') {
    return withTranscript({
      actId: 'list',
      actType: 'list',
      command: 'addItem',
      payload: { line1: requireText(args.line1, 'line1') },
      source: 'openai-realtime'
    }, transcript);
  }
  if (name === 'addChild') {
    return withTranscript({
      actId: requireText(args.parentId, 'parentId'),
      actType: 'task',
      command: 'addChild',
      payload: { line1: requireText(args.line1, 'line1') },
      source: 'openai-realtime'
    }, transcript);
  }
  if (name === 'addInfo') {
    return withTranscript({
      actId: requireText(args.parentId, 'parentId'),
      actType: 'task',
      command: 'addChild',
      payload: { line1: requireText(args.line1, 'line1'), status: 'Info' },
      source: 'openai-realtime'
    }, transcript);
  }
  if (name === 'setStatus') {
    const status = String(args.status || '');
    if (!ALLOWED_STATUSES.has(status)) throw new Error('Unsupported task status');
    return withTranscript({
      actId: requireText(args.taskId, 'taskId'),
      actType: 'task',
      command: 'setStatus',
      payload: { status },
      source: 'openai-realtime'
    }, transcript);
  }
  if (name === 'editItem') {
    return withTranscript({
      actId: requireText(args.taskId, 'taskId'),
      actType: 'task',
      command: 'editItem',
      payload: { line1: requireText(args.line1, 'line1') },
      source: 'openai-realtime'
    }, transcript);
  }
  if (name === 'setParent') {
    return withTranscript({
      actId: requireText(args.taskId, 'taskId'),
      actType: 'task',
      command: 'setParent',
      payload: { parentId: args.parentId == null ? null : requireText(args.parentId, 'parentId') },
      source: 'openai-realtime'
    }, transcript);
  }
  throw new Error(`Unsupported task operation: ${name}`);
}

function operationLabel(input) {
  if (input.command === 'addItem') return `Добавлена задача: ${input.payload.line1}`;
  if (input.command === 'addChild' && input.payload.status === 'Info') return `Добавлена информация: ${input.payload.line1}`;
  if (input.command === 'addChild') return `Добавлена подзадача: ${input.payload.line1}`;
  if (input.command === 'setStatus') return `Статус ${input.actId}: ${input.payload.status}`;
  if (input.command === 'editItem') return `Изменена задача ${input.actId}: ${input.payload.line1}`;
  if (input.command === 'setParent') return `Перемещена задача ${input.actId}`;
  return input.command;
}

function roleLabel(role) {
  if (role === 'user') return 'Вы';
  if (role === 'assistant') return 'Агент';
  if (role === 'tool') return 'Изменение';
  return 'Система';
}

function formatStartedAt(value) {
  return new Intl.DateTimeFormat('ru-RU', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit'
  }).format(new Date(value));
}

export function openAISetupTokenFromHash(hash) {
  return new URLSearchParams(String(hash || '').replace(/^#/, '')).get('openai-setup') || '';
}

export function createRealtimeVoiceAgent({
  voiceButton,
  voiceStatus,
  dialoguesButton,
  dialoguesPanel,
  dialoguesList,
  dialoguesClose,
  openAIKeyInput,
  openAIKeyField,
  openAIKeySaveButton,
  openAIKeyStatus,
  settingsOverlay,
  rootPanel,
  navigationButtons = [],
  getTaskState,
  executeTaskCommand,
  endpoint = '/api/realtime/session',
  keyStatusEndpoint = OPENAI_KEY_STATUS_ENDPOINT,
  keySetupEndpoint = OPENAI_KEY_SETUP_ENDPOINT,
  fetchImpl = fetch,
  mediaDevices = navigator.mediaDevices,
  RTCPeerConnectionCtor = globalThis.RTCPeerConnection,
  documentLike = document,
  windowLike = globalThis.window,
  storage = localStorage,
  locationLike = globalThis.location,
  historyLike = globalThis.history,
  now = () => new Date(),
  createId = () => crypto.randomUUID(),
  AbortControllerCtor = globalThis.AbortController
}) {
  const repository = createDialogueRepository({ storage, now, createId });
  let active = null;
  let keyConfigured = null;
  let setupAvailable = false;
  let setupToken = openAISetupTokenFromHash(locationLike?.hash);

  function setKeyStatus(message, tone = '') {
    if (!openAIKeyStatus) return;
    openAIKeyStatus.textContent = message;
    openAIKeyStatus.classList.toggle('success', tone === 'success');
    openAIKeyStatus.classList.toggle('error', tone === 'error');
  }

  function renderKeySettings() {
    if (!openAIKeyInput || !openAIKeySaveButton) return;
    const locked = keyConfigured === true;
    if (openAIKeyField) openAIKeyField.hidden = locked;
    openAIKeySaveButton.hidden = locked;
    openAIKeySaveButton.disabled = !setupToken || !setupAvailable;
    if (locked) setKeyStatus('Ключ сохранён на сервере и готов к работе.', 'success');
    else if (setupToken && setupAvailable) setKeyStatus('Вставьте ключ OpenAI. Ссылка настройки сработает один раз.');
    else if (keyConfigured === false) setKeyStatus('Откройте одноразовую ссылку настройки ключа.', 'error');
  }

  function openKeySettings() {
    settingsOverlay?.classList.add('open');
    settingsOverlay?.setAttribute('aria-hidden', 'false');
    renderKeySettings();
    if (keyConfigured !== true) openAIKeyInput?.focus();
  }

  function removeSetupTokenFromAddress() {
    if (!locationLike || !historyLike?.replaceState) return;
    const params = new URLSearchParams(String(locationLike.hash || '').replace(/^#/, ''));
    params.delete('openai-setup');
    const hash = params.toString();
    historyLike.replaceState(null, '', `${locationLike.pathname || '/'}${locationLike.search || ''}${hash ? `#${hash}` : ''}`);
  }

  async function refreshKeyStatus() {
    try {
      const response = await fetchImpl(keyStatusEndpoint, { headers: { Accept: 'application/json' } });
      if (!response.ok) return;
      const status = await response.json();
      keyConfigured = Boolean(status.configured);
      setupAvailable = Boolean(status.setupAvailable);
      renderKeySettings();
      if (setupToken && !keyConfigured) openKeySettings();
    } catch {
      // Static/local preview can run without the Worker status endpoint.
    }
  }

  async function saveOpenAIKey() {
    const apiKey = String(openAIKeyInput?.value || '').trim();
    if (!apiKey.startsWith('sk-') || apiKey.length < 20 || /\s/.test(apiKey)) {
      setKeyStatus('Проверьте ключ: он должен начинаться с sk-.', 'error');
      return;
    }
    if (!setupToken || !setupAvailable) {
      setKeyStatus('Ссылка настройки недействительна или уже использована.', 'error');
      return;
    }

    openAIKeySaveButton.disabled = true;
    setKeyStatus('Сохраняю ключ на сервере…');
    try {
      const response = await fetchImpl(keySetupEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ apiKey, setupToken })
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
      openAIKeyInput.value = '';
      keyConfigured = true;
      setupAvailable = false;
      setupToken = '';
      removeSetupTokenFromAddress();
      renderKeySettings();
    } catch (error) {
      setKeyStatus(error.message || 'Не удалось сохранить ключ.', 'error');
      openAIKeySaveButton.disabled = false;
    }
  }

  function renderDialogues() {
    dialoguesList.replaceChildren();
    const dialogues = repository.list();
    if (!dialogues.length) {
      const empty = documentLike.createElement('p');
      empty.className = 'dialogues-empty';
      empty.textContent = 'Здесь появятся расшифровки голосовых сессий.';
      dialoguesList.append(empty);
      return;
    }

    for (const dialogue of dialogues) {
      const card = documentLike.createElement('article');
      card.className = 'dialogue-card';
      const heading = documentLike.createElement('h3');
      heading.textContent = formatStartedAt(dialogue.startedAt);
      card.append(heading);

      if (!dialogue.messages.length) {
        const empty = documentLike.createElement('p');
        empty.className = 'dialogue-message system';
        empty.textContent = 'Нет распознанных реплик.';
        card.append(empty);
      }

      for (const message of dialogue.messages) {
        const row = documentLike.createElement('p');
        row.className = `dialogue-message ${message.role}`;
        const label = documentLike.createElement('strong');
        label.textContent = `${roleLabel(message.role)}: `;
        row.append(label, documentLike.createTextNode(message.text));
        card.append(row);
      }
      dialoguesList.append(card);
    }
  }

  function setDialoguesOpen(open) {
    rootPanel.dataset.dialoguesOpen = open ? 'true' : 'false';
    dialoguesPanel.hidden = !open;
    dialoguesButton.classList.toggle('active', open);
    dialoguesButton.setAttribute('aria-pressed', String(open));
    if (open) renderDialogues();
  }

  function setVoiceState(state, message) {
    voiceButton.dataset.state = state;
    voiceButton.setAttribute('aria-pressed', String(state === 'active' || state === 'connecting'));
    voiceButton.setAttribute('aria-label', state === 'idle' ? 'Начать голосовой диалог' : 'Остановить голосовой диалог');
    voiceStatus.textContent = message || '';
    voiceStatus.hidden = !message;
  }

  function append(role, text, eventId = null) {
    if (!active) return;
    repository.append(active.dialogueId, { role, text, eventId });
    if (rootPanel.dataset.dialoguesOpen === 'true') renderDialogues();
  }

  function clientEventId(prefix) {
    return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  }

  function sendRealtimeEvent(session, event, { ignoreError = false } = {}) {
    if (session.channel?.readyState !== 'open') return false;
    if (ignoreError && event.event_id) session.ignoredErrorEventIds.add(event.event_id);
    session.channel.send(JSON.stringify(event));
    return true;
  }

  function cancelRealtimeSpeech(session) {
    // The task ACK is now known, so interrupt stale speech and answer from the result.
    sendRealtimeEvent(session, { type: 'response.cancel', event_id: clientEventId('cancel-tool') }, { ignoreError: true });
    sendRealtimeEvent(session, { type: 'output_audio_buffer.clear', event_id: clientEventId('clear-tool-audio') }, { ignoreError: true });
  }

  async function handleToolCall(item) {
    const session = active;
    if (!session || session.toolCalls.has(item.call_id)) return;
    session.toolCalls.add(item.call_id);
    let output;
    let applied = false;
    try {
      const input = taskInputFromToolCall(item.name, item.arguments, {
        transcript: session.latestUserTranscript
      });
      const ack = await executeTaskCommand(input);
      if (ack?.status && ack.status !== 'applied') throw new Error(ack.reason || 'Task operation was rejected');
      repository.append(session.dialogueId, {
        role: 'tool',
        text: `Tool ${item.name}(${item.arguments || '{}'}): ${operationLabel(input)}`,
        eventId: `tool-${item.call_id}`
      });
      output = { status: 'applied', operation: input.command, target: ack?.newTarget || input.actId };
      applied = true;
    } catch (error) {
      repository.append(session.dialogueId, {
        role: 'system',
        text: `Ошибка изменения задачи: ${error.message}`,
        eventId: `tool-error-${item.call_id}`
      });
      output = { status: 'rejected', reason: error.message };
    }

    if (rootPanel.dataset.dialoguesOpen === 'true') renderDialogues();
    if (active !== session || session.channel?.readyState !== 'open') return;
    if (applied) cancelRealtimeSpeech(session);
    sendRealtimeEvent(session, {
      type: 'conversation.item.create',
      item: {
        type: 'function_call_output',
        call_id: item.call_id,
        output: JSON.stringify(output)
      }
    });
    sendRealtimeEvent(session, {
      type: 'response.create',
      response: {
        instructions: TOOL_RESULT_REPLY_INSTRUCTIONS
      }
    });
  }

  function transcriptFromAssistantItem(item) {
    if (item?.type !== 'message' || item?.role !== 'assistant' || !Array.isArray(item.content)) return '';
    return item.content.map((part) => part.transcript || part.text || '').filter(Boolean).join(' ').trim();
  }

  async function handleRealtimeEvent(event) {
    if (!active || !event?.type) return;

    if (event.type === 'conversation.item.input_audio_transcription.completed') {
      active.latestUserTranscript = String(event.transcript || '').trim();
      append('user', event.transcript, event.event_id || `user-${event.item_id || event.content_index || ''}`);
      return;
    }
    if (event.type === 'response.output_audio_transcript.done' ||
        event.type === 'response.audio_transcript.done' ||
        event.type === 'response.output_text.done') {
      append('assistant', event.transcript || event.text, event.event_id || `assistant-${event.response_id || ''}-${event.output_index || 0}`);
      return;
    }
    if (event.type === 'response.output_item.done') {
      if (event.item?.type === 'function_call') {
        await handleToolCall(event.item);
        return;
      }
      const transcript = transcriptFromAssistantItem(event.item);
      append('assistant', transcript, event.event_id || `assistant-item-${event.item?.id || ''}`);
      return;
    }
    if (event.type === 'error') {
      const clientEventId = event.error?.event_id || event.event_id || null;
      if (clientEventId && active.ignoredErrorEventIds.delete(clientEventId)) return;
      append('system', event.error?.message || 'Ошибка OpenAI Realtime', event.event_id || null);
    }
  }

  function cleanupVoiceSession(session) {
    session.abortController?.abort();
    try { session.channel?.close(); } catch {}
    try { session.peerConnection?.close(); } catch {}
    for (const track of session.mediaStream?.getTracks?.() || []) track.stop();
    if (session.audioElement) {
      session.audioElement.srcObject = null;
      session.audioElement.remove();
    }
  }

  function stopVoice() {
    const session = active;
    if (!session) return;
    active = null;
    cleanupVoiceSession(session);
    repository.finish(session.dialogueId);
    renderDialogues();
    setVoiceState('idle', '');
  }

  async function startVoice() {
    const dialogueId = repository.start();
    const session = {
      dialogueId,
      abortController: new AbortControllerCtor(),
      peerConnection: null,
      mediaStream: null,
      audioElement: null,
      channel: null,
      ignoredErrorEventIds: new Set(),
      toolCalls: new Set(),
      latestUserTranscript: ''
    };
    active = session;
    renderDialogues();
    setVoiceState('connecting', 'Подключение…');

    try {
      if (!RTCPeerConnectionCtor || !mediaDevices?.getUserMedia) throw new Error('Голосовой режим не поддерживается браузером');
      session.mediaStream = await mediaDevices.getUserMedia({ audio: true });
      if (active !== session) return cleanupVoiceSession(session);

      const pc = new RTCPeerConnectionCtor();
      session.peerConnection = pc;
      const audio = documentLike.createElement('audio');
      audio.autoplay = true;
      audio.playsInline = true;
      audio.hidden = true;
      documentLike.body.append(audio);
      session.audioElement = audio;
      pc.ontrack = (event) => { audio.srcObject = event.streams[0]; };

      for (const track of session.mediaStream.getTracks()) pc.addTrack(track, session.mediaStream);
      const channel = pc.createDataChannel('oai-events');
      session.channel = channel;
      channel.addEventListener('open', () => {
        if (active === session) setVoiceState('active', 'Слушаю');
      });
      channel.addEventListener('message', (message) => {
        try {
          void handleRealtimeEvent(JSON.parse(message.data));
        } catch {
          append('system', 'Получено некорректное событие Realtime');
        }
      });
      channel.addEventListener('close', () => {
        if (active === session) stopVoice();
      });

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const response = await fetchImpl(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: offer.sdp,
          taskTree: taskTreeFromState(getTaskState())
        }),
        signal: session.abortController.signal
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.error || `Realtime HTTP ${response.status}`);
      }
      await pc.setRemoteDescription({ type: 'answer', sdp: await response.text() });
      if (active === session && channel.readyState === 'open') setVoiceState('active', 'Слушаю');
    } catch (error) {
      if (active !== session || error.name === 'AbortError') return;
      append('system', error.message || 'Не удалось начать голосовой диалог');
      stopVoice();
      setVoiceState('error', 'Не удалось подключиться');
    }
  }

  voiceButton.addEventListener('click', () => {
    if (active) stopVoice();
    else if (keyConfigured === false) openKeySettings();
    else void startVoice();
  });
  openAIKeySaveButton?.addEventListener('click', () => { void saveOpenAIKey(); });
  openAIKeyInput?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      void saveOpenAIKey();
    }
  });
  dialoguesButton.addEventListener('click', () => {
    setDialoguesOpen(rootPanel.dataset.dialoguesOpen !== 'true');
  });
  dialoguesClose.addEventListener('click', () => setDialoguesOpen(false));
  for (const button of navigationButtons) button?.addEventListener('click', () => setDialoguesOpen(false));
  documentLike.addEventListener('visibilitychange', () => {
    // Mobile browsers can keep WebRTC alive while the tab/app is backgrounded;
    // close our Realtime transport explicitly so the session does not keep listening.
    if (documentLike.visibilityState === 'hidden') stopVoice();
  });
  windowLike?.addEventListener('pagehide', () => {
    // Page teardown is a separate lifecycle path from visibilitychange in some browsers.
    stopVoice();
  });

  setVoiceState('idle', '');
  setDialoguesOpen(false);
  renderDialogues();
  void refreshKeyStatus();

  return {
    renderDialogues,
    start: startVoice,
    stop: stopVoice,
    refreshKeyStatus,
    openDialogues() { setDialoguesOpen(true); }
  };
}
