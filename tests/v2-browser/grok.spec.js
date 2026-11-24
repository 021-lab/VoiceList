import { test, expect } from '@playwright/test';
import { createServer } from 'node:http';

// The Grok bot, played here: the worker under test must be started with
//   --var GROK_WEBHOOK_URL:http://127.0.0.1:4599/hook --var GROK_WEBHOOK_KEY:e2e-hook-key
//   --var GROK_REPLY_TOKEN:e2e-reply-token
// It accepts the call the way the routine's webhook does, then answers on reply.url the way
// docs/Grok-Bot-Routine.md tells the bot to.
const PORT = Number(process.env.GROK_MOCK_PORT || 4599);
const HOOK_KEY = 'e2e-hook-key', REPLY_TOKEN = 'e2e-reply-token';
const BASE_URL = process.env.V02_BASE_URL || 'http://127.0.0.1:4511/';
const calls = [];
let server;

test.beforeAll(async () => {
  server = createServer((request, response) => {
    let raw = '';
    request.on('data', chunk => { raw += chunk; });
    request.on('end', () => {
      if (request.headers.authorization !== `Bearer ${HOOK_KEY}`) { response.writeHead(401).end('{}'); return; }
      const call = JSON.parse(raw); calls.push(call);
      response.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({ success: true, runUuid: 'run-' + calls.length }));
      // wrangler dev rewrites the request host to the route in its config, so the address the
      // worker builds names the deployed host; the bot here answers the local one, same path.
      const replyUrl = new URL(new URL(call.reply.url).pathname, BASE_URL);
      setTimeout(() => fetch(replyUrl, {
        method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${REPLY_TOKEN}` },
        body: JSON.stringify({ taskId: call.taskId, replyTo: call.message.id, replyId: crypto.randomUUID(),
          text: `Нашёл три варианта для «${call.task.title}»\nПодробности: https://example.com/a`, done: false })
      }), 700);
    });
  });
  await new Promise(resolve => server.listen(PORT, '127.0.0.1', resolve));
});
test.afterAll(() => new Promise(resolve => server.close(resolve)));

const created = [];
test.afterEach(async ({ request }) => {
  const clientKey = `grok-cleanup-${Date.now()}`; let seq = 0;
  for (const id of created.splice(0).reverse()) {
    const snapshot = await (await request.get('/api/v2/document')).json();
    const body = { key: { clientKey, seq: ++seq }, context: { elementId: `task:${id}`, view: 'list', revision: snapshot.revision }, command: { command: 'deleteItem', actId: id, actType: 'task', payload: {} } };
    await request.post('/api/v2/input', { data: body });
    await expect.poll(async () => (await (await request.post('/api/v2/input', { data: body })).json()).status, { timeout: 15000 }).toBe('completed');
  }
});

test('delegate from the card: the reply lands in the thread and on the frontier without a reload; a tap there opens the card', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect.poll(() => page.evaluate(() => Boolean(window.__voiceListClient?.document))).toBe(true);
  const stamp = Date.now(), title = `Grok UAT ${stamp}`;
  // A focused parent puts its open child on the frontier.
  const parentId = await page.evaluate(async (t) => (await window.__voiceListClient.command('addItem', null, { title: t }))?.newTarget, `Grok parent ${stamp}`);
  created.push(parentId);
  await page.evaluate((id) => window.__voiceListClient.command('setStatus', id, { status: 'Focus' }), parentId);
  const taskId = await page.evaluate(async ([id, t]) => (await window.__voiceListClient.command('addChild', id, { title: t }))?.newTarget, [parentId, title]);
  created.push(taskId);
  expect(taskId).toBeTruthy();

  await page.evaluate((id) => window.__voiceListClient.navigate('edit', { taskId: id }), taskId);
  await expect(page.locator('#task-page')).toHaveClass(/open/);
  const block = page.locator('#task-page-grok');
  await expect(block).toBeVisible();
  test.skip(await page.locator('#grok-status').textContent() === 'Грок не подключён', 'worker started without GROK_* vars');

  const send = page.getByRole('button', { name: 'Делегировать Гроку' });
  await send.click();
  expect(calls).toHaveLength(0); // an empty field is not sent
  await page.locator('#grok-input').fill('Найди три варианта');
  await send.click();
  await expect(block.locator('[data-role=user]')).toContainText('Найди три варианта');
  await expect(page.locator('#grok-input')).toHaveValue('');
  await expect(block.locator('[data-role=grok]')).toContainText(`Нашёл три варианта для «${title}»`, { timeout: 10000 });
  await expect(block.locator('[data-role=grok] a')).toHaveAttribute('href', 'https://example.com/a');
  await expect(block.locator('.v02-grok-working')).toHaveCount(0);
  expect(new URL(calls.at(-1).reply.url).pathname).toBe('/api/grok/reply');
  expect(calls.at(-1)).toMatchObject({ kind: 'voicelist.delegation', taskId, message: { text: 'Найди три варианта' }, task: { title, path: [`Grok parent ${stamp}`] }, thread: [] });

  // The second message goes into the same thread and carries the first exchange.
  await page.locator('#grok-input').fill('А подешевле?');
  await send.click();
  await expect(block.locator('[data-role=grok]')).toHaveCount(2, { timeout: 10000 });
  expect(calls.at(-1).thread.map(message => message.role)).toEqual(['user', 'grok']);

  // The bot's tools: the thread it reads from its own app, and a page it publishes as a result.
  const auth = { Authorization: `Bearer ${REPLY_TOKEN}` };
  const thread = await (await page.request.get(`/api/grok/thread?taskId=${taskId}`, { headers: auth })).json();
  expect(thread).toMatchObject({ taskId, task: { title }, reply: { url: expect.stringMatching(/\/api\/grok\/reply$/) } });
  expect(thread.thread.map(message => message.role)).toEqual(['user', 'grok', 'user', 'grok']);
  const published = await page.request.post(`/api/grok/pages?taskId=${taskId}&title=${encodeURIComponent('Сравнение вариантов')}`,
    { headers: { ...auth, 'Content-Type': 'text/html; charset=utf-8' }, data: '<h1>Три варианта</h1>' });
  expect(published.status()).toBe(201);
  const { url: pageLink } = await published.json();
  await expect(block.locator('#grok-pages a')).toHaveText('Сравнение вариантов');
  await expect(block.locator('#grok-pages a')).toHaveAttribute('href', pageLink);
  expect(new URL(pageLink).pathname).toMatch(/^\/p\/[A-Za-z0-9_-]{22}$/);
  // On the application's own domain the page is not served.
  expect((await page.request.get(new URL(pageLink).pathname)).status()).toBe(404);
  await expect(page.locator('#grok-open')).toHaveAttribute('href', new RegExp(`^grokbot://app/v1/prompt\\?text=.*${taskId}`));

  await page.locator('#task-page-close').click();
  await page.getByRole('button', { name: 'Фронтир' }).click();
  const row = page.locator(`.list-item-wrapper[data-id="${taskId}"]`);
  const summary = row.locator('.v02-grok-summary');
  await expect(summary).toContainText('Нашёл три варианта');

  // A reply that arrives while the frontier is open shows up there without a reload.
  await page.request.post('/api/grok/reply', { headers: { Authorization: `Bearer ${REPLY_TOKEN}` },
    data: { taskId, replyTo: null, replyId: crypto.randomUUID(), text: 'Свежий ответ без перезагрузки', done: false } });
  await expect(summary).toContainText('Свежий ответ без перезагрузки', { timeout: 10000 });

  await summary.click();
  await expect(page.locator('#task-page')).toHaveClass(/open/);
  await expect(page.locator('#task-page-title')).toHaveValue(title);
  await expect(block.locator('[data-role=grok]')).toHaveCount(3);
  expect(errors).toEqual([]);
});

test('the reply endpoint refuses a wrong token and an unknown task', async ({ request }) => {
  const refused = await request.post('/api/grok/reply', { headers: { Authorization: 'Bearer wrong' }, data: { taskId: 'x', replyId: 'y', text: 'z' } });
  expect(refused.status()).toBe(401);
  const missing = await request.post('/api/grok/reply', { headers: { Authorization: `Bearer ${REPLY_TOKEN}` }, data: { taskId: 'no-such-task', replyId: crypto.randomUUID(), text: 'z' } });
  test.skip(missing.status() === 401, 'worker started without GROK_* vars');
  expect(missing.status()).toBe(404);
});
