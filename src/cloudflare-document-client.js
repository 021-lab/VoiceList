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
  queueStorage = window.localStorage
} = {}) {
  const clientKey = createClientKey(sessionStorage);
  const pendingKey = `voicelist.pending.${clientKey}`;
  const nextSeqKey = `voicelist.nextSeq.${clientKey}`;
  const stateHandlers = new Set();
  const ackHandlers = new Set();
  let socket = null;
  let knownRev = 0;
  let firstStateResolve = null;

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
      for (const handler of ackHandlers) handler(message.ack);
    }
  }

  function connect() {
    return new Promise((resolve, reject) => {
      firstStateResolve = resolve;
      socket = new WebSocketCtor(wsUrl(locationLike));
      socket.addEventListener('open', () => {
        const pending = readPending();
        sendRaw({
          type: 'hello',
          clientKey,
          knownRev,
          pendingSeq: pending.map((message) => message.seq)
        });
        flushPending();
      });
      socket.addEventListener('message', handleMessage);
      socket.addEventListener('error', reject);
    });
  }

  async function sendCommand(input) {
    const message = {
      type: 'command',
      clientKey,
      seq: nextSeq(),
      input
    };
    writePending([...readPending(), message]);
    sendRaw(message);
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
    sendUtterance
  };
}
