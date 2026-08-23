function createClientKey(sessionStorage) {
  const existing = sessionStorage.getItem('voicelist.clientKey');
  if (existing) return existing;
  const value = `tab-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  sessionStorage.setItem('voicelist.clientKey', value);
  return value;
}

function readJson(storage, key, fallback) {
  const raw = storage.getItem(key);
  if (!raw) return fallback;
  try {
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function writeJson(storage, key, value) {
  storage.setItem(key, JSON.stringify(value));
}

function wsUrl(locationLike) {
  const protocol = locationLike.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${locationLike.host}/ws`;
}

export function createCloudflareDocumentClient({
  WebSocketCtor = WebSocket,
  locationLike = window.location,
  sessionStorage = window.sessionStorage,
  queueStorage = window.localStorage,
  setTimeoutFn = globalThis.setTimeout,
  clearTimeoutFn = globalThis.clearTimeout,
  reconnectDelayMs = 500
} = {}) {
  const clientKey = createClientKey(sessionStorage);
  const pendingKey = `voicelist.pending.${clientKey}`;
  const nextSeqKey = `voicelist.nextSeq.${clientKey}`;
  const stateHandlers = new Set();
  const ackHandlers = new Set();
  const ackWaiters = new Map();
  let socket = null;
  let knownRev = 0;
  let connectPromise = null;
  let firstStateResolve = null;
  let reconnectTimer = null;

  function readPending() {
    return readJson(queueStorage, pendingKey, []);
  }

  function writePending(messages) {
    writeJson(queueStorage, pendingKey, messages);
  }

  function nextSeq() {
    const value = Number(queueStorage.getItem(nextSeqKey) || '1');
    queueStorage.setItem(nextSeqKey, String(value + 1));
    return value;
  }

  function sendRaw(message) {
    if (socket?.readyState === WebSocketCtor.OPEN) {
      socket.send(JSON.stringify(message));
      return true;
    }
    scheduleReconnect(0);
    return false;
  }

  function flushPending() {
    for (const message of readPending()) sendRaw(message);
  }

  function removePending(seq) {
    writePending(readPending().filter((message) => message.seq !== seq));
  }

  function handleMessage(event) {
    const message = JSON.parse(event.data);
    if (message.type === 'state' && message.state) {
      knownRev = message.state.rev ?? knownRev;
      for (const handler of stateHandlers) handler(message.state);
      firstStateResolve?.(message.state);
      firstStateResolve = null;
    } else if (message.type === 'ack' && message.ack) {
      if (message.ack.status === 'applied') removePending(message.ack.seq);
      const waiter = ackWaiters.get(message.ack.seq);
      if (waiter) {
        clearTimeoutFn(waiter.timeoutId);
        ackWaiters.delete(message.ack.seq);
        waiter.resolve(message.ack);
      }
      for (const handler of ackHandlers) handler(message.ack);
    }
  }

  function clearReconnect() {
    if (!reconnectTimer) return;
    clearTimeoutFn(reconnectTimer);
    reconnectTimer = null;
  }

  function isSocketConnectingOrOpen() {
    return socket?.readyState === WebSocketCtor.CONNECTING || socket?.readyState === WebSocketCtor.OPEN;
  }

  function openSocket() {
    if (isSocketConnectingOrOpen()) return;
    clearReconnect();

    const nextSocket = new WebSocketCtor(wsUrl(locationLike));
    socket = nextSocket;

    nextSocket.addEventListener('open', () => {
      if (socket !== nextSocket) return;
      const pending = readPending();
      sendRaw({
        type: 'hello',
        clientKey,
        knownRev,
        pendingSeq: pending.map((message) => message.seq)
      });
      flushPending();
    });
    nextSocket.addEventListener('message', handleMessage);
    nextSocket.addEventListener('close', () => {
      if (socket === nextSocket) socket = null;
      scheduleReconnect(reconnectDelayMs);
    });
    nextSocket.addEventListener('error', () => {
      if (socket === nextSocket && nextSocket.readyState !== WebSocketCtor.OPEN) socket = null;
      scheduleReconnect(reconnectDelayMs);
    });
  }

  function scheduleReconnect(delayMs) {
    if (reconnectTimer || isSocketConnectingOrOpen()) return;
    reconnectTimer = setTimeoutFn(() => {
      reconnectTimer = null;
      openSocket();
    }, delayMs);
  }

  function connect() {
    if (!connectPromise) {
      connectPromise = new Promise((resolve) => {
        firstStateResolve = resolve;
        openSocket();
      });
    } else {
      openSocket();
    }
    return connectPromise;
  }

  function queueCommand(input) {
    const seq = nextSeq();
    const message = {
      type: 'command',
      clientKey,
      seq,
      input
    };
    writePending([...readPending(), message]);
    return message;
  }

  function sendCommand(input) {
    const message = queueCommand(input);
    sendRaw(message);
    return message.seq;
  }

  function sendCommandAndWait(input, { timeoutMs = 10_000 } = {}) {
    const message = queueCommand(input);
    const result = new Promise((resolve, reject) => {
      const timeoutId = setTimeoutFn(() => {
        ackWaiters.delete(message.seq);
        reject(new Error('Timed out waiting for task update'));
      }, timeoutMs);
      ackWaiters.set(message.seq, { resolve, reject, timeoutId });
    });
    sendRaw(message);
    return result;
  }

  async function sendUtterance({ target, transcript }) {
    const message = {
      type: 'utterance',
      clientKey,
      seq: nextSeq(),
      target,
      transcript
    };
    writePending([...readPending(), message]);
    sendRaw(message);
  }

  return {
    connect,
    onAck(handler) {
      ackHandlers.add(handler);
      return () => ackHandlers.delete(handler);
    },
    onState(handler) {
      stateHandlers.add(handler);
      return () => stateHandlers.delete(handler);
    },
    sendCommand,
    sendCommandAndWait,
    sendUtterance
  };
}
