import { beforeEach, describe, expect, it, vi } from 'vitest';
import { DatabaseSync } from 'node:sqlite';
import { DocumentRuntime } from '../../src/v2/domain/document-runtime.js';
import { GrokPages, PAGE_HEADERS, PAGE_LIMIT_BYTES, isPageId } from '../../worker/v2/grok-pages.js';

// The worker's entry module, with the document object played by the object under test.
let stub;
vi.mock('agents', () => ({ Agent: class {}, getAgentByName: async () => stub }));
const { default: worker } = await import('../../worker/v2/index.js');

function storage() {
  const db = new DatabaseSync(':memory:');
  return { sql: { exec(query, ...params) {
    const prepared = db.prepare(query);
    if (/^\s*select/i.test(query)) { const rows = prepared.all(...params); return { toArray: () => rows }; }
    prepared.run(...params); return { toArray: () => [] };
  } } };
}
/** R2, as far as pages use it. */
function bucket() {
  const objects = new Map();
  return { objects,
    async put(key, bytes) { objects.set(key, new Uint8Array(bytes)); },
    async get(key) { const bytes = objects.get(key); return bytes ? { body: bytes } : null; },
    async delete(key) { objects.delete(key); } };
}

const TOKEN = 'reply-token';
let env, pages, log;
beforeEach(() => {
  env = { GROK_PAGES: bucket(), GROK_PAGES_HOST: 'vlist-pages.smileme.ai', GROK_REPLY_TOKEN: TOKEN };
  const runtime = new DocumentRuntime();
  log = vi.fn();
  pages = new GrokPages({ storage: storage(), env, readItems: async () => runtime.graph.read().items, log });
  stub = {
    publishGrokPage: (page) => pages.publish(page),
    replaceGrokPage: (id, bytes) => pages.replace(id, bytes),
    removeGrokPage: (id) => pages.remove(id)
  };
});

const app = (path, init = {}) => worker.fetch(new Request('https://vlist-v02-prod.smileme.ai' + path, init), env);
const pagesDomain = (path, init = {}) => worker.fetch(new Request('https://vlist-pages.smileme.ai' + path, init), env);
const auth = { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'text/html; charset=utf-8' };
const publish = (html, query = '?taskId=milk1&title=' + encodeURIComponent('Сравнение')) => app('/api/grok/pages' + query, { method: 'POST', headers: auth, body: html });

describe('pages: publishing', () => {
  it('publishes, serves on the pages domain with the sandbox headers, and logs size and sha256', async () => {
    const response = await publish('<h1>Итог</h1>');
    expect(response.status).toBe(201);
    const { id, url } = await response.json();
    expect(isPageId(id)).toBe(true);
    expect(url).toBe(`https://vlist-pages.smileme.ai/p/${id}`);
    expect(pages.forTask('milk1')).toMatchObject([{ id, title: 'Сравнение', size: 17 }]);

    const page = await pagesDomain(`/p/${id}`);
    expect(page.status).toBe(200);
    expect(await page.text()).toBe('<h1>Итог</h1>');
    for (const [name, value] of Object.entries(PAGE_HEADERS)) expect(page.headers.get(name)).toBe(value);
    expect(page.headers.get('Content-Security-Policy')).not.toContain('allow-same-origin');

    expect(log.mock.calls[0][0]).toMatchObject({ type: 'grok.page.publish', page_id: id, task_id: 'milk1', size: 17, sha256: expect.stringMatching(/^[0-9a-f]{64}$/) });
  });

  it('ids are 128 random bits and do not repeat', async () => {
    const ids = new Set();
    for (let index = 0; index < 20; index++) ids.add((await (await publish('x', '')).json()).id);
    expect(ids.size).toBe(20);
    for (const id of ids) expect(id).toMatch(/^[A-Za-z0-9_-]{22}$/);
  });

  it('a page without a task is allowed; an unknown task is 404', async () => {
    expect((await publish('x', '')).status).toBe(201);
    expect((await publish('x', '?taskId=nope')).status).toBe(404);
  });

  it('replacing keeps the link; deleting removes the page', async () => {
    const { id, url } = await (await publish('v1')).json();
    const replaced = await app(`/api/grok/pages/${id}`, { method: 'PUT', headers: auth, body: 'v2' });
    expect(replaced.status).toBe(200);
    expect(await replaced.json()).toEqual({ id, url });
    expect(await (await pagesDomain(`/p/${id}`)).text()).toBe('v2');

    expect((await app(`/api/grok/pages/${id}`, { method: 'DELETE', headers: auth })).status).toBe(204);
    expect((await pagesDomain(`/p/${id}`)).status).toBe(404);
    expect(pages.get(id)).toBeNull();
    expect((await app(`/api/grok/pages/${id}`, { method: 'DELETE', headers: auth })).status).toBe(404);
    expect((await app(`/api/grok/pages/${id}`, { method: 'PUT', headers: auth, body: 'v3' })).status).toBe(404);
  });

  it('more than 10 MB is 413', async () => {
    expect((await publish(new Uint8Array(PAGE_LIMIT_BYTES + 1))).status).toBe(413);
    expect((await publish(new Uint8Array(PAGE_LIMIT_BYTES))).status).toBe(201);
  });

  it('a wrong or missing token is 401 for publish, replace and delete', async () => {
    const { id } = await (await publish('x')).json();
    for (const headers of [{ Authorization: 'Bearer wrong' }, {}]) {
      expect((await app('/api/grok/pages', { method: 'POST', headers, body: 'x' })).status).toBe(401);
      expect((await app(`/api/grok/pages/${id}`, { method: 'PUT', headers, body: 'x' })).status).toBe(401);
      expect((await app(`/api/grok/pages/${id}`, { method: 'DELETE', headers })).status).toBe(401);
    }
    expect(pages.exportAll()).toHaveLength(1);
  });

  it('without a bucket or a pages domain publishing is 503', async () => {
    delete env.GROK_PAGES;
    expect((await publish('x')).status).toBe(503);
  });
});

describe('pages: the two domains', () => {
  it('the pages domain answers only GET /p/<id>', async () => {
    const { id } = await (await publish('x')).json();
    for (const path of ['/', '/health', '/api/tasks/tree.json', '/api/grok/pages', `/p/${id}/x`, '/p/short']) expect((await pagesDomain(path)).status).toBe(404);
    expect((await pagesDomain('/api/grok/pages', { method: 'POST', headers: auth, body: 'x' })).status).toBe(404);
    expect((await pagesDomain(`/p/${id}`, { method: 'DELETE', headers: auth })).status).toBe(404);
  });

  it('the application domain does not serve /p/<id>', async () => {
    const { id } = await (await publish('x')).json();
    expect((await app(`/p/${id}`)).status).toBe(404);
  });
});
