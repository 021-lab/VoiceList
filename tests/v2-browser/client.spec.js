import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page, request }) => {
  expect((await request.get('/health')).status()).toBe(200);
  await page.goto('/');
  await expect(page.locator('#list-container')).toBeVisible();
  await expect.poll(() => page.evaluate(() => Boolean(window.__voiceListClient?.document))).toBe(true);
});
async function add(page, title) {
  await page.locator('#add-btn').click();
  await page.locator('#input-line1').fill(title);
  await page.locator('#input-line1').press('Enter');
  await expect(page.locator('#modal-overlay')).not.toHaveClass(/open/);
  const row = page.locator('.list-item-wrapper').filter({ has: page.locator('.item-line1', { hasText: title }) });
  await expect(row).toBeVisible(); return row;
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
