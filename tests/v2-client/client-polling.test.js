import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { Client } from '../../src/v2/client/client.js';

const node = (type, id, props = {}, children = []) => ({ type, id, props, children });
const documentTree = () => ({ schemaVersion: 1, revision: 3, cursor: 0, view: 'list',
  root: node('application', 'app', {}, [node('toolbar', 'toolbar'), node('task-list', 'screen:list', {}, [])]) });
const empty = { events: [], actions: [], uiEffects: [], nextCursor: 0, revision: 3 };
const response = (data) => ({ ok: true, status: 200, json: async () => data });
const serve = async (path) => response(path.startsWith('/api/v2/updates') ? empty
  : path.startsWith('/api/v2/input') ? { requestId: 'r1', status: 'accepted', actions: [], error: null }
  : documentTree());

let client;
beforeEach(() => {
  const template = readFileSync('list-manager.template.html', 'utf8');
  document.body.innerHTML = template.match(/<body>([\s\S]*)<script>/)[1];
  sessionStorage.clear();
  vi.stubGlobal('scrollTo', vi.fn());
});
afterEach(() => { client?.disconnect(); vi.restoreAllMocks(); });

/** Сокет, которым можно управлять из теста: открыть, уронить, прислать сообщение. */
class FakeSocket {
  static last = null;
  constructor(url) { this.url = url; this.readyState = 0; this.listeners = new Map(); FakeSocket.last = this; }
  addEventListener(type, handler) { this.listeners.set(type, [...(this.listeners.get(type) || []), handler]); }
  emit(type, event) { for (const handler of this.listeners.get(type) || []) handler(event); }
  open() { this.readyState = 1; this.emit('open', {}); }
  push(message) { this.emit('message', { data: JSON.stringify(message) }); }
  close() { this.readyState = 3; this.emit('close', {}); }
}

const create = (options = {}) => {
  client = new Client({ document, storage: sessionStorage, pollMs: 1200, idlePollMs: 15000, watchPollMs: 120000, fetch: vi.fn(serve), ...options });
  client.mount();
  client.render(documentTree());
  client.connected = true;
  return client;
};

/** Каждый опрос будит объект, а объект оплачивается по числу обращений: вкладка, открытая
 *  на сутки, тратит дневной лимит впустую. */
describe('частота опроса', () => {
  it('замедляется, когда подряд приходят пустые ответы', async () => {
    create();
    expect(client.pollDelay()).toBe(1200);
    for (let index = 0; index < 5; index += 1) await client.resume();
    expect(client.quietPolls).toBe(5);
    expect(client.pollDelay()).toBe(15000);
  });

  it('возвращается к обычному ритму, как только пользователь что-то сделал', async () => {
    create();
    for (let index = 0; index < 5; index += 1) await client.resume();
    await client.submit({ command: { command: 'showList', actId: 'menu:list', actType: 'tab', payload: {} } });
    expect(client.pollDelay()).toBe(1200);
  });

  it('скрытая вкладка не опрашивает вовсе', () => {
    create();
    Object.defineProperty(document, 'visibilityState', { value: 'hidden', configurable: true });
    client.schedulePoll();
    expect(client.pollTimer).toBeFalsy();
    Object.defineProperty(document, 'visibilityState', { value: 'visible', configurable: true });
    client.schedulePoll();
    expect(client.pollTimer).toBeDefined();
  });
});

describe('сервер сам будит страницу', () => {
  beforeEach(() => { FakeSocket.last = null; });

  it('подключается к объекту как наблюдатель и не просит снимок', async () => {
    create({ WebSocketCtor: FakeSocket });
    client.connected = false;
    await client.connect();
    expect(FakeSocket.last.url).toContain('/ws?client=v2');
  });

  it('сообщение об изменении забирает обновление, не дожидаясь таймера', async () => {
    create({ WebSocketCtor: FakeSocket });
    client.openSocket();
    FakeSocket.last.open();
    const before = client.fetch.mock.calls.length;
    FakeSocket.last.push({ type: 'changed', cursor: 7, revision: 4 });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(client.fetch.mock.calls.length).toBeGreaterThan(before);
  });

  it('пока сокет открыт, страница не опрашивает объект вовсе', () => {
    create({ WebSocketCtor: FakeSocket });
    client.openSocket();
    client.schedulePoll();
    expect(client.pollTimer).toBeDefined();
    FakeSocket.last.open();
    client.schedulePoll();
    expect(client.pollTimer).toBeFalsy();
  });

  it('сигнал, пришедший во время похода за обновлением, не теряется', async () => {
    create({ WebSocketCtor: FakeSocket });
    client.polling = true;
    await client.resume();
    expect(client.signalWhileBusy).toBe(true);
    client.polling = false;
    const before = client.fetch.mock.calls.length;
    await client.resume();
    expect(client.signalWhileBusy).toBe(false);
    // Поход и его повтор: пропущенный сигнал забран, а не отброшен.
    expect(client.fetch.mock.calls.length).toBeGreaterThan(before + 1);
  });

  it('подключившийся сокет первым делом догоняет пропущенное', async () => {
    create({ WebSocketCtor: FakeSocket });
    client.openSocket();
    const before = client.fetch.mock.calls.length;
    FakeSocket.last.open();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(client.fetch.mock.calls.length).toBeGreaterThan(before);
  });

  it('упавший сокет возвращает обычный опрос и переподключается с отсрочкой', () => {
    create({ WebSocketCtor: FakeSocket });
    client.openSocket();
    FakeSocket.last.open();
    FakeSocket.last.close();
    expect(client.socketOpen()).toBe(false);
    expect(client.pollDelay()).toBe(1200);
    expect(client.socketRetryTimer).toBeTruthy();
  });

  it('скрытая вкладка закрывает сокет, видимая открывает снова', async () => {
    create({ WebSocketCtor: FakeSocket });
    client.mount();
    client.openSocket();
    FakeSocket.last.open();
    Object.defineProperty(document, 'visibilityState', { value: 'hidden', configurable: true });
    document.dispatchEvent(new Event('visibilitychange'));
    expect(client.socket).toBeNull();
    Object.defineProperty(document, 'visibilityState', { value: 'visible', configurable: true });
    document.dispatchEvent(new Event('visibilitychange'));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(client.socket).toBeTruthy();
  });
});
