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

const create = () => {
  client = new Client({ document, storage: sessionStorage, pollMs: 1200, idlePollMs: 15000, fetch: vi.fn(serve) });
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
    expect(client.pollTimer).toBeUndefined();
    Object.defineProperty(document, 'visibilityState', { value: 'visible', configurable: true });
    client.schedulePoll();
    expect(client.pollTimer).toBeDefined();
  });
});
