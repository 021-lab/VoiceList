import { test, expect } from '@playwright/test';

// Жесты пользователя проверяются настоящими pointer-событиями в браузере.
// Логика — тот же GestureRecognizer, что в приложении; стенд только
// подаёт в него события и публикует состояние в window.__g.

const HARNESS = '/tests/e2e/gesture-harness.html';
const g = (page) => page.evaluate(() => window.__g);

test.beforeEach(async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(HARNESS);
});

test('удержание взводит оверлей без контекста', async ({ page }) => {
  await page.mouse.move(200, 500);
  await page.mouse.down();
  await page.waitForTimeout(600);
  const s = await g(page);
  expect(s.state).toBe('armed');
  expect(s.mode).toBe('global');
  expect(s.contextId).toBeNull();
  await page.mouse.up();
});

test('удержание на элементе тоже не даёт контекста', async ({ page }) => {
  await page.locator('[agent-action-id="3"]').hover();
  await page.mouse.down();
  await page.waitForTimeout(600);
  expect((await g(page)).contextId).toBeNull();
  await page.mouse.up();
});

test('свайп влево по элементу прицепляет контекст', async ({ page }) => {
  const box = await page.locator('[agent-action-id="2"]').boundingBox();
  const y = box.y + box.height / 2;
  await page.mouse.move(box.x + box.width - 20, y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width - 90, y, { steps: 4 });
  const s = await g(page);
  expect(s.state).toBe('armed');
  expect(s.mode).toBe('context');
  expect(s.contextId).toBe('2');
  await page.mouse.up();
});

test('вертикальная протяжка до таймера отдаётся прокрутке', async ({ page }) => {
  await page.mouse.move(200, 600);
  await page.mouse.down();
  await page.mouse.move(200, 540, { steps: 3 });
  expect((await g(page)).state).toBe('scroll');
  await page.mouse.up();
});

test('короткое касание — тап, оверлея нет', async ({ page }) => {
  await page.mouse.move(200, 500);
  await page.mouse.down();
  await page.waitForTimeout(80);
  await page.mouse.up();
  expect((await g(page)).released.state).toBe('tap');
});

test('зоны выбора: как есть, кандидаты, отмена', async ({ page }) => {
  await page.mouse.move(200, 500);
  await page.mouse.down();
  await page.waitForTimeout(600);

  await page.mouse.move(200, 495);                 // в мёртвой зоне
  expect((await g(page)).zone).toBe('as-is');

  await page.mouse.move(200, 480);                 // первый кандидат
  let s = await g(page);
  expect(s.zone).toBe('candidate');
  expect(s.index).toBe(0);

  await page.mouse.move(200, 420);                 // второй кандидат
  s = await g(page);
  expect(s.index).toBe(1);
  expect(s.frozen).toBe(true);                     // порядок заморожен

  await page.mouse.move(200, 580);                 // ниже — отмена
  expect((await g(page)).zone).toBe('cancel');
  await page.mouse.up();
});

test('клэмп по длине стека', async ({ page }) => {
  await page.evaluate(() => window.__setStack(2));
  await page.mouse.move(200, 600);
  await page.mouse.down();
  await page.waitForTimeout(600);
  await page.mouse.move(200, 200);
  expect((await g(page)).index).toBe(1);
  await page.mouse.up();
});
