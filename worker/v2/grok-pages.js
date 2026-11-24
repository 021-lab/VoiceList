/** Pages the Grok bot publishes as the result of its work.
 *
 *  The HTML is the bot's, so it is served from a domain of its own: the same worker answers
 *  there, but only `GET /p/<id>`. Served from the application's origin, its script would pass
 *  the same-origin check of `/api/*`. The sandbox CSP (without allow-same-origin) is a second
 *  line. The link is the only access to a page, so the id is 128 random bits.
 *  Contract: docs/Grok-Delegation.md. */

export const PAGE_LIMIT_BYTES = 10 * 1024 * 1024;
export const PAGE_HEADERS = {
  'Content-Type': 'text/html; charset=utf-8',
  'Content-Security-Policy': 'sandbox allow-scripts allow-popups allow-forms allow-modals',
  'X-Robots-Tag': 'noindex',
  'Referrer-Policy': 'no-referrer',
  'Cache-Control': 'no-store'
};

const ID = /^[A-Za-z0-9_-]{22}$/;
export const newPageId = () => {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  let text = ''; for (const byte of bytes) text += String.fromCharCode(byte);
  return btoa(text).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};
export const isPageId = (id) => ID.test(String(id || ''));
const objectKey = (id) => `pages/${id}.html`;
export const pagesHost = (env) => String(env?.GROK_PAGES_HOST || '').toLowerCase();
export const pageUrl = (env, id) => `https://${pagesHost(env)}/p/${id}`;

/** On the pages domain this is the whole worker: one page by id, or nothing. Returns null for
 *  any other host, so the application's routes run there — and `/p/…` is not one of them. */
export async function servePagesHost(request, env) {
  const url = new URL(request.url);
  if (!pagesHost(env) || url.hostname.toLowerCase() !== pagesHost(env)) return null;
  const notFound = () => new Response('Not found', { status: 404, headers: { 'Cache-Control': 'no-store', 'X-Robots-Tag': 'noindex' } });
  const id = /^\/p\/([^/]+)$/.exec(url.pathname)?.[1];
  if (!['GET', 'HEAD'].includes(request.method) || !isPageId(id) || !env.GROK_PAGES) return notFound();
  const object = await env.GROK_PAGES.get(objectKey(id));
  if (!object) return notFound();
  return new Response(request.method === 'HEAD' ? null : object.body, { headers: PAGE_HEADERS });
}

/** The HTML as bytes, refused past the limit without reading the rest. */
export async function readPageBody(request, limit = PAGE_LIMIT_BYTES) {
  if (Number(request.headers.get('Content-Length') || 0) > limit) return { tooLarge: true };
  const reader = request.body?.getReader();
  if (!reader) return { bytes: new Uint8Array(0) };
  const chunks = []; let size = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    size += value.byteLength;
    if (size > limit) { await reader.cancel(); return { tooLarge: true }; }
    chunks.push(value);
  }
  const bytes = new Uint8Array(size); let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return { bytes };
}

const sha256 = async (bytes) => [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(byte => byte.toString(16).padStart(2, '0')).join('');

const row = (value) => ({ id: value.id, taskId: value.task_id ?? null, title: value.title, size: value.size, createdAt: value.created_at, updatedAt: value.updated_at });

export class GrokPages {
  constructor({ storage, env, readItems, log = () => {}, changed = () => {}, now = () => new Date().toISOString() }) {
    Object.assign(this, { sql: storage.sql, env, readItems, log, changed, now });
    this.sql.exec(`CREATE TABLE IF NOT EXISTS grok_pages (
      id TEXT PRIMARY KEY,
      task_id TEXT,
      title TEXT NOT NULL,
      size INTEGER NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL)`);
    this.sql.exec('CREATE INDEX IF NOT EXISTS grok_pages_task ON grok_pages (task_id, created_at)');
  }
  configured() { return Boolean(this.env?.GROK_PAGES && pagesHost(this.env)); }
  get(id) { const found = this.sql.exec('SELECT * FROM grok_pages WHERE id = ?', id).toArray()[0]; return found ? row(found) : null; }
  forTask(taskId) { return this.sql.exec('SELECT * FROM grok_pages WHERE task_id = ? ORDER BY created_at, rowid', taskId).toArray().map(row); }
  exportAll() { return this.sql.exec('SELECT * FROM grok_pages ORDER BY created_at, rowid').toArray().map(row); }
  importAll(pages) {
    for (const page of pages || []) if (!this.get(page.id)) this.sql.exec('INSERT INTO grok_pages (id,task_id,title,size,created_at,updated_at) VALUES (?,?,?,?,?,?)',
      page.id, page.taskId, page.title, page.size, page.createdAt, page.updatedAt);
  }
  record(action, page, bytes, hash) {
    this.log({ type: `grok.page.${action}`, page_id: page.id, task_id: page.taskId, size: bytes?.byteLength ?? page.size, ...(hash ? { sha256: hash } : {}) },
      { sessionId: 'grok:' + (page.taskId || 'pages'), direction: 'in' });
  }

  async publish({ taskId = null, title = '', bytes }) {
    if (!this.configured()) return { status: 503, body: { error: 'Публикация страниц не подключена' } };
    if (taskId != null && taskId !== '' && !(await this.readItems()).some(item => item.id === taskId)) return { status: 404, body: { error: 'Задача не найдена' } };
    const page = { id: newPageId(), taskId: taskId || null, title: String(title || '').trim().slice(0, 200) || 'Страница', size: bytes.byteLength, createdAt: this.now() };
    page.updatedAt = page.createdAt;
    // Listed first, stored second: a failed upload leaves no public page that nothing tracks.
    this.sql.exec('INSERT INTO grok_pages (id,task_id,title,size,created_at,updated_at) VALUES (?,?,?,?,?,?)', page.id, page.taskId, page.title, page.size, page.createdAt, page.updatedAt);
    try { await this.env.GROK_PAGES.put(objectKey(page.id), bytes, { httpMetadata: { contentType: PAGE_HEADERS['Content-Type'] } }); }
    catch (error) { this.sql.exec('DELETE FROM grok_pages WHERE id = ?', page.id); throw error; }
    this.record('publish', page, bytes, await sha256(bytes));
    this.changed();
    return { status: 201, body: { id: page.id, url: pageUrl(this.env, page.id) } };
  }

  async replace(id, bytes) {
    if (!this.configured()) return { status: 503, body: { error: 'Публикация страниц не подключена' } };
    const page = isPageId(id) && this.get(id);
    if (!page) return { status: 404, body: { error: 'Страница не найдена' } };
    await this.env.GROK_PAGES.put(objectKey(id), bytes, { httpMetadata: { contentType: PAGE_HEADERS['Content-Type'] } });
    const updatedAt = this.now();
    this.sql.exec('UPDATE grok_pages SET size = ?, updated_at = ? WHERE id = ?', bytes.byteLength, updatedAt, id);
    this.record('replace', { ...page, size: bytes.byteLength }, bytes, await sha256(bytes));
    this.changed();
    return { status: 200, body: { id, url: pageUrl(this.env, id) } };
  }

  async remove(id) {
    if (!this.configured()) return { status: 503, body: { error: 'Публикация страниц не подключена' } };
    const page = isPageId(id) && this.get(id);
    if (!page) return { status: 404, body: { error: 'Страница не найдена' } };
    await this.env.GROK_PAGES.delete(objectKey(id));
    this.sql.exec('DELETE FROM grok_pages WHERE id = ?', id);
    this.record('delete', page);
    this.changed();
    return { status: 204, body: null };
  }
}
