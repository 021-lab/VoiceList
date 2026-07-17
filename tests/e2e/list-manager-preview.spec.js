import { test, expect } from '@playwright/test';

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
  await expect(page.locator('.list-item-wrapper').first()).toContainText('Молоко 3.2%');

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
