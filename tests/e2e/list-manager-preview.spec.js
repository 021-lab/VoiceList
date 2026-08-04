import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page, request }) => {
  await page.route('https://firestore.googleapis.com/**', async (route) => {
    await route.fulfill({ status: 404, contentType: 'application/json', body: '{}' });
  });
  await request.post('/reset', {
    headers: {
      'X-VoiceList-Test-Reset': 'local-reset'
    }
  }).catch(() => {});
});

async function confirmModal(page) {
  await expect(page.locator('#modal-overlay')).toHaveClass(/open/);
  await page.locator('#input-line1').press('Enter');
  await expect(page.locator('#modal-overlay')).not.toHaveClass(/open/);
}

async function triggerRightPanelAction(page, rowLocator, actionLabel) {
  await rowLocator.scrollIntoViewIfNeeded();
  await expect(rowLocator).toBeVisible();
  const rowBox = await rowLocator.boundingBox();
  if (!rowBox) throw new Error(`No bounding box for row ${actionLabel}`);
  const viewport = page.viewportSize();
  const swipeDistance = Math.max(320, Math.floor((viewport?.width || 1280) * 0.28));

  await page.mouse.move(rowBox.x + 24, rowBox.y + rowBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(rowBox.x + swipeDistance, rowBox.y + rowBox.height / 2, { steps: 10 });

  const panelItem = page.locator('#drop-zone-panel .panel-item', { hasText: actionLabel });
  await expect(panelItem).toBeVisible();
  const panelBox = await panelItem.boundingBox();
  if (!panelBox) throw new Error(`No panel box for action ${actionLabel}`);

  await page.mouse.move(panelBox.x + panelBox.width / 2, panelBox.y + panelBox.height / 2, { steps: 6 });
  await page.mouse.up();
}

async function triggerRightPanelActionUntil(page, rowLocator, actionLabel, verify) {
  let lastError = null;

  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      await triggerRightPanelAction(page, rowLocator, actionLabel);
      await expect.poll(verify, { timeout: 1000 }).toBe(true);
      return;
    } catch (error) {
      lastError = error;
      await page.mouse.up().catch(() => {});
    }
  }

  throw lastError;
}

test('preview app can create task, create subtask, and change status', async ({ page }) => {
  const taskTitle = `E2E Task ${Date.now()}`;
  const subtaskTitle = `E2E Subtask ${Date.now()}`;

  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();
  await expect(page.locator('#list-container')).toBeVisible();
  await expect(page.locator('#list-container')).toContainText('Молоко 3.2%');

  await page.getByRole('button', { name: 'Добавить задачу' }).click();
  await page.locator('#input-line1').fill(taskTitle);
  await confirmModal(page);

  const taskRow = page.locator('.list-item-wrapper', { hasText: taskTitle });
  await expect(taskRow).toContainText(taskTitle);
  await expect(taskRow).toContainText('Open');

  await page.locator('#view-toggle-btn').click();
  await expect(page.locator('#action-log-list')).toContainText(`Создана задача: ${taskTitle}`);
  await page.locator('#view-toggle-btn').click();

  await triggerRightPanelActionUntil(page, taskRow.locator('.list-item'), 'Edit', async () => (
    (await page.locator('#task-page').getAttribute('aria-hidden')) === 'false'
  ));
  await expect(page.locator('#task-page')).toHaveClass(/open/);
  await page.locator('#task-page-child-input').fill(subtaskTitle);
  await page.locator('#task-page-add-child').click();
  await expect(page.locator('#task-page-subtasks')).toContainText(subtaskTitle);
  await page.locator('#task-page-close').click();

  const subtaskRow = page.locator('.list-item-wrapper', { hasText: subtaskTitle });
  await expect(subtaskRow).toContainText(subtaskTitle);
  expect(Number(await subtaskRow.getAttribute('data-level'))).toBeGreaterThan(Number(await taskRow.getAttribute('data-level')));

  await triggerRightPanelActionUntil(page, taskRow.locator('.list-item'), 'Focus', async () => (
    (await taskRow.textContent()).includes('Focus')
  ));
  await expect(taskRow).toContainText('Focus');

  await page.locator('#frontier-tab-btn').click();
  await expect(page.locator('#list-container')).toContainText(subtaskTitle);
  await expect(page.locator('#list-container .list-item-wrapper', { hasText: taskTitle })).toHaveCount(0);
  await page.locator('#frontier-tab-btn').click();

  await page.locator('#view-toggle-btn').click();
  await expect(page.locator('#action-log-list')).toContainText(`Создана подзадача: ${subtaskTitle}`);
  await expect(page.locator('#action-log-list')).toContainText('Статус изменён: Focus');
  await page.locator('#view-toggle-btn').click();

  await triggerRightPanelActionUntil(page, taskRow.locator('.list-item'), 'Done', async () => (
    (await taskRow.textContent()).includes('Done')
  ));
  await expect(taskRow).toContainText('Done');
  await page.locator('#view-toggle-btn').click();
  await expect(page.locator('#action-log-list')).toContainText('Статус изменён: Done');
});

test('preview app executes voice commands through touch gestures', async ({ page }) => {
  const voiceTitle = `Голосовая задача ${Date.now()}`;
  const normalizedVoiceTitle = voiceTitle.toLowerCase();
  await page.addInitScript((title) => {
    window.__voiceTest = { phrase: `добавь ${title}` };
  }, voiceTitle);

  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();
  await expect(page.locator('#list-container')).toBeVisible();
  await expect(page.locator('.list-item-wrapper', { hasText: 'Входящие' })).toBeVisible();

  await page.mouse.move(220, 620);
  await page.mouse.down();
  await page.waitForTimeout(520);
  await expect(page.locator('#voice-overlay')).toHaveClass(/open/);
  await page.mouse.up();

  const voiceRow = page.locator('.list-item-wrapper', { hasText: normalizedVoiceTitle });
  await expect(voiceRow).toContainText(normalizedVoiceTitle);
  expect(Number(await voiceRow.getAttribute('data-level'))).toBeGreaterThan(0);

  await page.evaluate(() => { window.__voiceTest = { phrase: 'готово' }; });
  const milkRow = page.locator('.list-item-wrapper', { hasText: 'Молоко 3.2%' }).locator('.list-item');
  const box = await milkRow.boundingBox();
  if (!box) throw new Error('No milk row box');
  const viewport = page.viewportSize();
  const leftSwipeDistance = Math.max(320, Math.floor((viewport?.width || 1280) * 0.28));

  await page.mouse.move(box.x + box.width - 24, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width - leftSwipeDistance, box.y + box.height / 2, { steps: 8 });
  await expect(page.locator('#voice-overlay')).toHaveClass(/open/);
  await page.mouse.up();

  await expect(page.locator('.list-item-wrapper', { hasText: 'Молоко 3.2%' })).toContainText('Done');
});

test('preview app positions voice overlay near the pointer and selects a candidate by movement', async ({ page }) => {
  await page.addInitScript(() => {
    window.__voiceTest = { phrase: 'добавь молоко' };
  });

  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();

  const anchor = { x: 220, y: 620 };

  await page.mouse.move(anchor.x, anchor.y);
  await page.mouse.down();
  await page.waitForTimeout(520);
  const overlay = page.locator('#voice-overlay');
  await expect(overlay).toHaveClass(/open/);

  const overlayBox = await overlay.boundingBox();
  if (!overlayBox) throw new Error('No voice overlay box');
  expect(overlayBox.y).toBeLessThan(anchor.y);
  expect(overlayBox.y + overlayBox.height).toBeGreaterThan(anchor.y - 4);

  await page.mouse.move(anchor.x, anchor.y - 28, { steps: 6 });
  await expect(page.locator('#voice-overlay .voice-candidate.selected')).toHaveText('Молоко 3.2%');
  await page.mouse.move(anchor.x, anchor.y - 4, { steps: 4 });
  await expect(page.locator('#voice-overlay .voice-candidate.selected')).toHaveText('Добавить: задачу молоко');
  await page.mouse.move(anchor.x, anchor.y - 28, { steps: 4 });
  await expect(page.locator('#voice-overlay .voice-candidate.selected')).toHaveText('Молоко 3.2%');

  await page.mouse.up();
  await expect(overlay).not.toHaveClass(/open/);
  await expect.poll(() => overlay.evaluate((element) => element.style.top)).toMatch(/px$/);
  await expect(page.locator('#modal-overlay')).toHaveClass(/open/);
  await expect(page.locator('#view-line1')).toHaveText('Молоко 3.2%');
});

test('preview app adds a voice comment to a log entry on press speak release', async ({ page }) => {
  const taskTitle = `Log voice task ${Date.now()}`;
  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();

  await page.getByRole('button', { name: 'Добавить задачу' }).click();
  await page.locator('#input-line1').fill(taskTitle);
  await confirmModal(page);

  await page.evaluate(() => {
    window.__voiceTest = { phrase: 'купить сегодня' };
  });

  await page.locator('#view-toggle-btn').click();
  const logRow = page.locator('.action-log-row', { hasText: taskTitle }).first();
  await expect(logRow).toBeVisible();
  const box = await logRow.boundingBox();
  if (!box) throw new Error('No log row box');

  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.waitForTimeout(520);
  await expect(page.locator('#voice-overlay')).toHaveClass(/open/);
  await page.mouse.up();

  await expect(page.locator('#voice-overlay')).not.toHaveClass(/open/);
  await expect(logRow).toContainText('купить сегодня');
});

test('preview app does not log collapse toggles and undo restores status changes', async ({ page }) => {
  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();

  const breadRow = page.locator('.list-item-wrapper', { hasText: 'Хлеб ржаной' }).locator('.list-item');
  await breadRow.click();
  await expect(page.locator('.list-item-wrapper', { hasText: 'Бородинский' })).toBeHidden();

  await page.locator('#view-toggle-btn').click();
  await expect(page.locator('#action-log-list')).not.toContainText('Переключено сворачивание');
  await page.locator('#view-toggle-btn').click();

  const taskRow = page.locator('.list-item-wrapper', { hasText: 'Молоко 3.2%' });
  await triggerRightPanelActionUntil(page, taskRow.locator('.list-item'), 'Done', async () => (
    (await taskRow.textContent()).includes('Done')
  ));
  await expect(taskRow).toContainText('Done');

  await page.locator('#undo-btn').click();
  await expect(taskRow).toContainText('Open');
});
