import { describe, expect, test, vi } from 'vitest';

import { createCloudflareDocumentClient } from '../../src/cloudflare-document-client.js';

class FakeWebSocket {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = FakeWebSocket.CONNECTING;
    this.listeners = {};
    this.sent = [];
    FakeWebSocket.instances.push(this);
  }

  addEventListener(type, handler) {
    this.listeners[type] ||= [];
    this.listeners[type].push(handler);
  }

  send(payload) {
    this.sent.push(JSON.parse(payload));
  }

  emit(type, event = {}) {
    for (const handler of this.listeners[type] || []) handler(event);
  }

  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.emit('open');
  }

  message(payload) {
    this.emit('message', { data: JSON.stringify(payload) });
  }
}

FakeWebSocket.CONNECTING = 0;
FakeWebSocket.OPEN = 1;

function createMemoryStorage() {
  const entries = new Map();
  return {
    getItem(key) {
      return entries.has(key) ? entries.get(key) : null;
    },
    setItem(key, value) {
      entries.set(key, String(value));
    },
    removeItem(key) {
      entries.delete(key);
    }
  };
}

describe('Cloudflare document client', () => {
  test('opens Worker WebSocket, sends hello, and sends commands with client seq', async () => {
    FakeWebSocket.instances = [];
    const onState = vi.fn();
    const client = createCloudflareDocumentClient({
      WebSocketCtor: FakeWebSocket,
      locationLike: { protocol: 'https:', host: 'vlist-cloudflare-backend.smileme.ai' },
      sessionStorage: createMemoryStorage(),
      queueStorage: createMemoryStorage()
    });
    client.onState(onState);

    const connected = client.connect();
    const socket = FakeWebSocket.instances[0];
    socket.open();
    expect(socket.url).toBe('wss://vlist-cloudflare-backend.smileme.ai/ws');
    expect(socket.sent[0]).toMatchObject({
      type: 'hello',
      knownRev: 0,
      pendingSeq: []
    });

    socket.message({
      type: 'state',
      state: {
        rev: 0,
        content: { snapshot: { items: [] }, actionLog: [] }
      }
    });
    await expect(connected).resolves.toEqual({
      rev: 0,
      content: { snapshot: { items: [] }, actionLog: [] }
    });
    expect(onState).toHaveBeenCalledOnce();

    await client.sendCommand({
      actId: 'list',
      actType: 'list',
      command: 'addItem',
      payload: { line1: 'Backend task', line2: '' },
      source: 'unit-test'
    });

    expect(socket.sent[1]).toMatchObject({
      type: 'command',
      seq: 1,
      input: {
        actId: 'list',
        actType: 'list',
        command: 'addItem',
        payload: { line1: 'Backend task', line2: '' },
        source: 'unit-test'
      }
    });
    expect(socket.sent[1].clientKey).toMatch(/^tab-/);
  });
});
