import { test, expect } from '@playwright/test';

let createdTaskIds = [];

test.beforeEach(async ({ page, request }) => {
  createdTaskIds = [];
  expect((await request.get('/health')).status()).toBe(200);
  await page.goto('/');
  await expect(page.locator('#list-container')).toBeVisible();
  await expect.poll(() => page.evaluate(() => Boolean(window.__voiceListClient?.document))).toBe(true);
});
test.afterEach(async ({ request }) => {
  const clientKey = `uat-cleanup-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  let seq = 0;
  for (const id of [...new Set(createdTaskIds)].reverse()) {
    const snapshot = await (await request.get('/api/v2/document')).json();
    const body = { key: { clientKey, seq: ++seq }, context: { elementId: `task:${id}`, view: 'list', revision: snapshot.revision }, command: { command: 'deleteItem', actId: id, actType: 'task', payload: {} } };
    const result = await request.post('/api/v2/input', { data: body });
    if (result.status() === 404) continue;
    expect(result.ok()).toBe(true);
    await expect.poll(async () => (await (await request.post('/api/v2/input', { data: body })).json()).status, { timeout: 15000 }).toBe('completed');
  }
});
async function add(page, title) {
  await page.locator('#add-btn').click();
  await page.locator('#input-line1').fill(title);
  await page.locator('#input-line1').press('Enter');
  await expect(page.locator('#modal-overlay')).not.toHaveClass(/open/);
  const row = page.locator('.list-item-wrapper').filter({ has: page.locator('.item-line1', { hasText: title }) });
  await expect(row).toBeVisible(); createdTaskIds.push(await row.getAttribute('data-id')); return row;
}
async function swipe(page, row, label, direction = 1) {
  await row.scrollIntoViewIfNeeded(); const rect = await row.boundingBox();
  const startX = direction > 0 ? rect.x + 24 : rect.x + rect.width - 24;
  await page.mouse.move(startX, rect.y + rect.height / 2); await page.mouse.down();
  await page.mouse.move(startX + direction * 360, rect.y + rect.height / 2, { steps: 8 });
  const item = page.locator(`${direction > 0 ? '#drop-zone-panel' : '#tag-panel'} .panel-item`).filter({ hasText: label });
  await expect(item).toBeVisible(); const target = await item.boundingBox();
  await page.mouse.move(target.x + target.width / 2, target.y + target.height / 2, { steps: 5 }); await page.mouse.up();
}

test('actual server flow: create, edit, status, journal, correction chain rollback and Close', async ({ page }) => {
  const title = `UAT ${Date.now()}`; const renamed = `${title} edited`;
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  const row = await add(page, title); const taskId = await row.getAttribute('data-id');
  await swipe(page, row.locator('.list-item'), 'Edit');
  await expect(page.locator('#task-page')).toHaveClass(/open/);
  await page.locator('#task-page-line1').fill(renamed); await page.locator('#task-page-line2').fill('Details retained');
  await page.locator('#task-page-save').click();
  await expect.poll(() => page.locator('#task-page-line1').inputValue()).toBe(renamed);
  await page.locator('#task-page-close').click();
  const current = page.locator(`.list-item-wrapper[data-id="${taskId}"]`);
  await expect(current).toContainText(renamed); await expect(current).toContainText('Details retained');
  await swipe(page, current.locator('.list-item'), 'Done'); await expect(current).toContainText('Done');
  await page.locator('#view-toggle-btn').click();
  const action = page.locator('.action-log-row').filter({ hasText: 'Статус изменён: Done' }).last();
  await expect(action).toBeVisible(); await action.click();
  await expect(page.locator('#v02-action-page')).toBeVisible();
  await page.locator('#action-correction-input').fill('сделай фокус');
  await page.locator('.v02-correction-form button').click();
  await expect(page.locator('.v02-message').last()).toContainText('Focus');
  await page.locator('#action-rollback').click();
  await expect(page.locator('.v02-message').last()).toContainText(/отмен|откат/i);
  await page.locator('#action-close').click(); await expect(page.locator('#action-log-panel')).toBeVisible();
  await page.locator('#view-toggle-btn').click(); await expect(current).toContainText('Open');
  await page.reload(); await expect(current).toContainText(renamed); await expect(current).toContainText('Open');
  expect(errors).toEqual([]);
});

test('top-edge swipe opens journal; native touch hold accepts voice input', async ({ page, browser }) => {
  await page.mouse.move(300, 10); await page.mouse.down(); await page.mouse.move(300, 120, { steps: 6 }); await page.mouse.up();
  await expect(page.locator('#action-log-panel')).toBeVisible();
  await expect.poll(() => page.locator('#action-log-panel').evaluate(panel => panel.scrollHeight - panel.scrollTop - panel.clientHeight)).toBeLessThan(4);
  await expect(page.locator('.action-log-row').last()).toBeInViewport();
  await page.locator('#action-log-panel').evaluate(panel => { panel.scrollTop = 0; });
  await expect(page.locator('.action-log-row').first()).toBeInViewport();
  await page.mouse.move(300, 570); await page.mouse.down(); await page.mouse.move(300, 460, { steps: 6 }); await page.mouse.up();
  await expect(page.locator('#action-log-panel')).toBeVisible();
  await page.locator('#action-log-panel').evaluate(panel => { panel.scrollTop = panel.scrollHeight; });
  await page.mouse.move(300, 570); await page.mouse.down(); await page.mouse.move(300, 460, { steps: 6 }); await page.mouse.up();
  await expect(page.locator('#action-log-panel')).toBeHidden();
  await expect(page.locator('#list-container')).toBeVisible();
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const mobile = await context.newPage(); await mobile.goto('/');
  await expect.poll(() => mobile.evaluate(() => Boolean(window.__voiceListClient?.document))).toBe(true);
  const row = await add(mobile, `Touch UAT ${Date.now()}`); await row.scrollIntoViewIfNeeded();
  await mobile.evaluate(() => { window.__voiceTest = { phrase: 'сделай фокус' }; });
  const box = await row.boundingBox(); const x = box.x + 80, y = box.y + box.height / 2;
  const cdp = await context.newCDPSession(mobile);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] }); await mobile.waitForTimeout(360);
  await expect(mobile.locator('#v02-transcript')).toContainText('сделай фокус');
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] }); await expect(row).toContainText('Focus');
  await context.close();
});

test('hold voice sends final text only, downward edit and cancellation remain local', async ({ page }) => {
  const title = `Voice UAT ${Date.now()}`; const row = await add(page, title);
  await page.evaluate(() => { window.__voiceTest = { phrase: 'сделай фокус' }; });
  await row.scrollIntoViewIfNeeded();
  const sent = []; page.on('request', request => { if (request.url().endsWith('/api/v2/input')) sent.push(request.postDataJSON()); });
  const rect = await row.boundingBox(); const x = rect.x + 150, y = rect.y + rect.height / 2;
  await page.mouse.move(x, y); await page.mouse.down(); await page.waitForTimeout(360);
  await expect(page.locator('#v02-transcript')).toContainText('сделай фокус'); expect(sent).toHaveLength(0);
  await page.mouse.up(); await expect(row).toContainText('Focus');
  await page.waitForTimeout(1300); const before = sent.length;
  await page.mouse.move(x, y); await page.mouse.down(); await page.waitForTimeout(360); await page.mouse.move(x, y + 55); await page.mouse.up();
  await expect(page.locator('#transcript-edit-input')).toHaveValue('сделай фокус'); expect(sent).toHaveLength(before);
  await page.locator('.v02-edit-transcript .btn-cancel').click();
  await page.mouse.move(x, y); await page.mouse.down(); await page.waitForTimeout(360); await page.mouse.move(x, y + 150); await page.mouse.up();
  await expect(page.locator('.v02-edit-transcript')).toHaveCount(0); expect(sent).toHaveLength(before);
});

test('native touch hold-up selects drag, subsequent up-down moves preserve drag and persist order', async ({ browser }) => {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await context.newPage(); await page.goto('/');
  await expect.poll(() => page.evaluate(() => Boolean(window.__voiceListClient?.document))).toBe(true);
  await page.evaluate(() => { window.__voiceTest = { phrase: '' }; });
  const stamp = Date.now(); const first = await add(page, `Drag first ${stamp}`); const second = await add(page, `Drag second ${stamp}`);
  const firstId = await first.getAttribute('data-id'), secondId = await second.getAttribute('data-id');
  await second.scrollIntoViewIfNeeded();
  const a = await first.boundingBox(), b = await second.boundingBox();
  const x = b.x + 90, y = b.y + b.height / 2;
  const cdp = await context.newCDPSession(page);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] }); await page.waitForTimeout(360);
  for (const nextY of [y - 24, y - 40, y - 10, a.y + a.height / 2]) {
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x, y: nextY }] });
    await page.waitForTimeout(30);
    await expect.poll(() => page.evaluate(() => window.__voiceListClient.gesture.state)).toBe('dragging');
  }
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  const order = () => page.locator('#list-container .list-item-wrapper').evaluateAll(elements => elements.map(el => el.dataset.id));
  await expect.poll(async () => { const ids = await order(); return ids.indexOf(secondId) < ids.indexOf(firstId); }).toBe(true);
  await page.reload();
  await expect.poll(async () => { const ids = await order(); return ids.includes(secondId) && ids.indexOf(secondId) < ids.indexOf(firstId); }).toBe(true);
  await context.close();
});

test('native touch down frames transcript until release and Send; deeper down cancels; pre-hold scroll stays native', async ({ browser }) => {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await context.newPage(); await page.goto('/');
  await expect.poll(() => page.evaluate(() => Boolean(window.__voiceListClient?.document))).toBe(true);
  const row = await add(page, `Touch edit ${Date.now()}`); await row.scrollIntoViewIfNeeded();
  await page.evaluate(() => { window.__voiceTest = { phrase: 'сделай паузу' }; });
  const sent = []; page.on('request', request => { if (request.url().endsWith('/api/v2/input') && request.postDataJSON()?.text) sent.push(request.postDataJSON()); });
  const cdp = await context.newCDPSession(page);
  let box = await row.boundingBox(); let x = box.x + 75, y = box.y + box.height / 2;
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] }); await page.waitForTimeout(360);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x, y: y + 30 }] });
  await expect(page.locator('#v02-transcript')).toHaveAttribute('data-state', 'editing');
  await expect(page.locator('#v02-transcript')).toHaveCSS('border-top-width', '2px'); expect(sent).toHaveLength(0);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x, y: y + 5 }] });
  await expect(page.locator('#v02-transcript')).toHaveAttribute('data-state', 'editing');
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await expect(page.locator('#transcript-edit-input')).toHaveValue('сделай паузу'); expect(sent).toHaveLength(0);
  await page.locator('#transcript-edit-input').fill('сделай фокус'); await page.locator('.v02-edit-transcript .v02-primary').click();
  await expect(row).toContainText('Focus'); await page.waitForTimeout(1200); const before = sent.length;
  box = await row.boundingBox(); x = box.x + 75; y = box.y + box.height / 2;
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] }); await page.waitForTimeout(360);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x, y: y + 140 }] });
  await expect(page.locator('#v02-transcript')).toHaveAttribute('data-state', 'cancelled');
  await expect(row).not.toHaveClass(/v02-voice-target/);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await expect(page.locator('#transcript-edit-input')).toHaveCount(0); await expect(page.locator('#v02-transcript')).toBeHidden(); expect(sent).toHaveLength(before);
  const scrollBefore = await page.evaluate(() => scrollY);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x, y: y + 65 }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await expect.poll(() => page.evaluate(() => scrollY)).toBeLessThan(scrollBefore);
  await page.waitForTimeout(350); await expect(page.locator('#v02-transcript')).toBeHidden(); expect(sent).toHaveLength(before);
  await context.close();
});
