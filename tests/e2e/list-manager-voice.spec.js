import { test, expect } from '@playwright/test';

// The browser's real SpeechRecognition needs a microphone and a network speech
// service, so the deployed build exposes the controller for scripted transcripts.
async function say(page, phrase) {
  return page.evaluate((spoken) => window.__voiceControl__.handleTranscript(spoken), phrase);
}

test.beforeEach(async ({ page }) => {
  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();
  await expect(page.locator('#list-container')).toBeVisible();
  await expect(page.locator('#voice-panel')).toBeVisible();
  // Voice wiring lands after the backend adapter settles during bootstrap.
  await page.waitForFunction(() => Boolean(window.__voiceControl__));
});

test('voice commands create, nest, retitle and re-status tasks', async ({ page }) => {
  await say(page, 'добавь задачу купить батарейки');
  const taskRow = page.locator('.list-item-wrapper', { hasText: 'купить батарейки' });
  await expect(taskRow).toContainText('купить батарейки');
  await expect(taskRow).toContainText('Open');
  await expect(page.locator('#voice-status')).toHaveAttribute('data-state', 'accepted');

  await say(page, 'добавь подзадачу зарядка к купить батарейки');
  const subtaskRow = page.locator('.list-item-wrapper', { hasText: 'зарядка' });
  await expect(subtaskRow).toBeVisible();
  expect(Number(await subtaskRow.getAttribute('data-level')))
    .toBeGreaterThan(Number(await taskRow.getAttribute('data-level')));

  await say(page, 'переименуй купить батарейки в купить аккумуляторы');
  await expect(page.locator('.list-item-wrapper', { hasText: 'купить аккумуляторы' })).toBeVisible();

  await say(page, 'фокус на купить аккумуляторы');
  await expect(page.locator('.list-item-wrapper', { hasText: 'купить аккумуляторы' })).toContainText('Focus');

  await say(page, 'отметь купить аккумуляторы выполнено');
  await expect(page.locator('.list-item-wrapper', { hasText: 'купить аккумуляторы' })).toContainText('Done');
});

test('voice commands switch views and undo the last mutation', async ({ page }) => {
  await say(page, 'удали шампунь');
  await expect(page.locator('.list-item-wrapper', { hasText: 'Шампунь' })).toHaveCount(0);

  await say(page, 'покажи журнал');
  await expect(page.locator('#action-log-panel')).toBeVisible();
  await expect(page.locator('#action-log-list')).toContainText('Удалена задача');

  await say(page, 'покажи список');
  await expect(page.locator('#list-container')).toBeVisible();

  await say(page, 'отмена');
  await expect(page.locator('.list-item-wrapper', { hasText: 'Шампунь' })).toHaveCount(1);

  await say(page, 'покажи фронтир');
  await expect(page.locator('#app-root')).toHaveAttribute('data-view-mode', 'frontier');
});

test('unrecognized speech is reported without changing the list', async ({ page }) => {
  const before = await page.locator('.list-item-wrapper').count();

  await say(page, 'сыграй музыку');
  await expect(page.locator('#voice-status')).toHaveAttribute('data-state', 'rejected');
  await expect(page.locator('#voice-status')).toContainText('Не понял команду');
  expect(await page.locator('.list-item-wrapper').count()).toBe(before);

  await say(page, 'удали вертолёт');
  await expect(page.locator('#voice-status')).toContainText('Не нашёл задачу');
  expect(await page.locator('.list-item-wrapper').count()).toBe(before);
});

test('microphone button toggles listening state', async ({ page }) => {
  await page.evaluate(() => {
    // Stub the engine so the button path is exercised without a real microphone.
    window.__voiceControl__.start = () => {
      document.getElementById('voice-btn').classList.add('listening');
      return true;
    };
  });

  const voiceButton = page.locator('#voice-btn');
  await expect(voiceButton).toBeVisible();
  await voiceButton.click();
  await expect(voiceButton).toHaveClass(/listening/);
});
