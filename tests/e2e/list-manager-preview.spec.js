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

async function triggerLeftPanelAction(page, rowLocator, actionLabel) {
  await rowLocator.scrollIntoViewIfNeeded();
  await expect(rowLocator).toBeVisible();
  const rowBox = await rowLocator.boundingBox();
  if (!rowBox) throw new Error(`No bounding box for row ${actionLabel}`);
  const viewport = page.viewportSize();
  const swipeDistance = Math.max(320, Math.floor((viewport?.width || 1280) * 0.28));

  await page.mouse.move(rowBox.x + rowBox.width - 24, rowBox.y + rowBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(rowBox.x + rowBox.width - swipeDistance, rowBox.y + rowBox.height / 2, { steps: 10 });

  const panelItem = page.locator('#tag-panel .panel-item', { hasText: actionLabel });
  await expect(panelItem).toBeVisible();
  const panelBox = await panelItem.boundingBox();
  if (!panelBox) throw new Error(`No tag panel box for action ${actionLabel}`);

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

async function dragRowVertically(page, rowLocator, deltaY, deltaX = 0) {
  await rowLocator.scrollIntoViewIfNeeded();
  await expect(rowLocator).toBeVisible();
  const rowBox = await rowLocator.boundingBox();
  if (!rowBox) throw new Error('No row box for drag');

  const x = rowBox.x + rowBox.width / 2;
  const y = rowBox.y + rowBox.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.waitForTimeout(520);
  await page.mouse.move(x + deltaX, y + deltaY, { steps: 14 });
  await page.mouse.up();
}

async function visibleListOrder(page) {
  return page.locator('#list-container .list-item-wrapper').evaluateAll((nodes) => (
    nodes
      .filter((node) => node.offsetParent !== null)
      .map((node) => node.dataset.id)
  ));
}

async function mockWorkflowyExport(page) {
  await page.route('https://workflowy.com/**', async (route) => {
    const url = route.request().url();
    const origin = route.request().headers().origin || '*';
    const headers = {
      'Access-Control-Allow-Origin': origin,
      'Access-Control-Allow-Credentials': 'true'
    };
    if (url.includes('/s/task-tree/')) {
      await route.fulfill({
        status: 200,
        headers: {
          ...headers,
          'Content-Type': 'text/html',
          'Set-Cookie': 'sessionid=abc; Path=/; HttpOnly'
        },
        body: '<script>var PROJECT_TREE_DATA_URL_PARAMS = {"share_id":"Share.123"};</script>'
      });
      return;
    }
    if (url.includes('/get_initialization_data')) {
      await route.fulfill({
        status: 200,
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectTreeData: {
            auxiliaryProjectTreeInfos: [{
              rootProject: { id: 'root', nm: 'task tree' }
            }],
            initialMostRecentOperationTransactionId: '42'
          }
        })
      });
      return;
    }
    if (url.includes('/get_tree_data/')) {
      await route.fulfill({
        status: 200,
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          items: [
            { id: 'child', prnt: 'root', pr: 10, nm: 'Child task' },
            { id: 'nested', prnt: 'child', pr: 10, nm: 'Nested task' }
          ]
        })
      });
      return;
    }
    await route.abort();
  });
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

test('cloudflare mode reconnects after an idle WebSocket close and renders status changes without reload', async ({ page }) => {
  await page.addInitScript(() => {
    const NativeWebSocket = window.WebSocket;
    window.__voiceListSockets = [];
    window.WebSocket = class TestWebSocket extends NativeWebSocket {
      constructor(url, protocols) {
        super(url, protocols);
        window.__voiceListSockets.push(this);
      }
    };
    window.WebSocket.CONNECTING = NativeWebSocket.CONNECTING;
    window.WebSocket.OPEN = NativeWebSocket.OPEN;
    window.WebSocket.CLOSING = NativeWebSocket.CLOSING;
    window.WebSocket.CLOSED = NativeWebSocket.CLOSED;
  });

  await page.goto('');
  const cloudflareMode = await page.evaluate(() => !window.location.hash.includes('v=local-dev') && window.location.port !== '4511');
  test.skip(!cloudflareMode, 'Cloudflare document client is disabled in static local-dev mode.');

  const taskRow = page.locator('.list-item-wrapper', { hasText: 'Молоко 3.2%' });
  await expect(taskRow).toContainText('Open');
  await expect.poll(() => page.evaluate(() => window.__voiceListSockets.some((socket) => socket.readyState === WebSocket.OPEN))).toBe(true);

  await page.evaluate(() => {
    for (const socket of window.__voiceListSockets) {
      if (socket.readyState === WebSocket.OPEN) socket.close(1000, 'test idle close');
    }
  });

  await triggerRightPanelActionUntil(page, taskRow.locator('.list-item'), 'Done', async () => (
    (await taskRow.textContent()).includes('Done')
  ));
  await expect(taskRow).toContainText('Done');
});

test('OpenAI Realtime button sends hidden task context, applies a tool call, and persists Dialogues', async ({ page }) => {
  let sessionRequest = null;
  let keySetupRequest = null;
  let keyConfigured = false;
  await page.addInitScript(() => {
    window.__realtimeChannels = [];
    window.__realtimeLifecycle = { channelClosed: 0, peerClosed: 0, tracksStopped: 0 };
    class FakeDataChannel {
      constructor() {
        this.readyState = 'connecting';
        this.listeners = {};
        this.sent = [];
        window.__realtimeChannels.push(this);
      }
      addEventListener(type, handler) {
        this.listeners[type] ||= [];
        this.listeners[type].push(handler);
      }
      emit(type, payload = {}) {
        for (const handler of this.listeners[type] || []) handler(payload);
      }
      send(payload) { this.sent.push(JSON.parse(payload)); }
      close() {
        if (this.readyState === 'closed') return;
        this.readyState = 'closed';
        window.__realtimeLifecycle.channelClosed += 1;
        this.emit('close');
      }
    }
    class FakePeerConnection {
      addTrack() {}
      createDataChannel() {
        this.channel = new FakeDataChannel();
        return this.channel;
      }
      async createOffer() { return { type: 'offer', sdp: 'test-offer-sdp' }; }
      async setLocalDescription() {}
      async setRemoteDescription() {
        this.channel.readyState = 'open';
        this.channel.emit('open');
      }
      close() {
        window.__realtimeLifecycle.peerClosed += 1;
      }
    }
    Object.defineProperty(window, 'RTCPeerConnection', { value: FakePeerConnection, configurable: true });
    Object.defineProperty(navigator, 'mediaDevices', {
      value: {
        async getUserMedia() {
          return { getTracks: () => [{ stop() { window.__realtimeLifecycle.tracksStopped += 1; } }] };
        }
      },
      configurable: true
    });
  });
  await page.route('**/api/realtime/session', async (route) => {
    sessionRequest = route.request().postDataJSON();
    await route.fulfill({ status: 200, contentType: 'application/sdp', body: 'test-answer-sdp' });
  });
  await page.route('**/api/realtime/key/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ configured: keyConfigured, setupAvailable: !keyConfigured })
    });
  });
  await page.route('**/api/realtime/key', async (route) => {
    keySetupRequest = route.request().postDataJSON();
    keyConfigured = true;
    await route.fulfill({ status: 201, contentType: 'application/json', body: JSON.stringify({ configured: true }) });
  });

  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();
  await expect(page.locator('#realtime-voice-btn')).toBeVisible();
  await page.locator('#settings-btn').click();
  await expect(page.locator('#settings-overlay')).toHaveClass(/open/);
  await page.locator('#openai-key-input').fill('sk-example-mobile-key-1234567890');
  await page.locator('#openai-key-save').click();
  await expect(page.locator('#openai-key-status')).toContainText('Ключ сохранён на сервере');
  expect(keySetupRequest).toEqual({
    apiKey: 'sk-example-mobile-key-1234567890'
  });
  expect(await page.evaluate(() => `${JSON.stringify(localStorage)}${JSON.stringify(sessionStorage)}`)).not.toContain('sk-example');
  await page.reload();
  await page.locator('#settings-btn').click();
  await expect(page.locator('#openai-key-field')).toBeHidden();
  await expect(page.locator('#openai-key-status')).toContainText('Ключ сохранён на сервере');
  await page.locator('#settings-close').click();

  const box = await page.locator('#realtime-voice-btn').boundingBox();
  const viewport = page.viewportSize();
  expect(box.x + box.width).toBeGreaterThan(viewport.width - 90);
  expect(Math.abs((box.y + box.height / 2) - viewport.height / 2)).toBeLessThan(8);

  await page.locator('#realtime-voice-btn').click();
  await expect(page.locator('#realtime-voice-btn')).toHaveAttribute('data-state', 'active');
  expect(sessionRequest.sdp).toBe('test-offer-sdp');
  expect(JSON.stringify(sessionRequest.taskTree)).toContain('"id":"milk1"');
  expect(JSON.stringify(sessionRequest.taskTree)).toContain('"title":"Молоко 3.2%"');
  expect(JSON.stringify(sessionRequest)).not.toContain('actionLog');
  expect(JSON.stringify(sessionRequest)).not.toContain('"line2"');

  await page.evaluate(() => {
    const channel = window.__realtimeChannels.at(-1);
    channel.emit('message', {
      data: JSON.stringify({
        type: 'conversation.item.input_audio_transcription.completed',
        event_id: 'user-1',
        transcript: 'Добавь задачу Позвонить маме'
      })
    });
    channel.emit('message', {
      data: JSON.stringify({
        type: 'response.output_item.done',
        event_id: 'tool-1',
        item: {
          type: 'function_call',
          call_id: 'call-1',
          name: 'addItem',
          arguments: JSON.stringify({ line1: 'Позвонить маме' })
        }
      })
    });
  });

  await expect.poll(() => page.evaluate(() => (
    document.body.textContent.includes('Позвонить маме') &&
    window.__realtimeChannels.at(-1).sent.map((event) => event.type).join('|')
  ))).toBe('response.cancel|output_audio_buffer.clear|conversation.item.create|response.create');
  expect(await page.evaluate(() => (
    window.__realtimeChannels.at(-1).sent.at(-1).response.instructions
  ))).toContain('function_call_output');

  await page.evaluate(() => {
    const channel = window.__realtimeChannels.at(-1);
    channel.emit('message', {
      data: JSON.stringify({
        type: 'response.output_audio_transcript.done',
        event_id: 'assistant-1',
        transcript: 'Готово, добавила задачу.'
      })
    });
  });

  await expect(page.locator('#list-container')).toContainText('Позвонить маме');
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { value: 'hidden', configurable: true });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await expect(page.locator('#realtime-voice-btn')).toHaveAttribute('data-state', 'idle');
  await expect.poll(() => page.evaluate(() => window.__realtimeLifecycle)).toEqual({
    channelClosed: 1,
    peerClosed: 1,
    tracksStopped: 1
  });
  await page.locator('#dialogues-tab-btn').click();
  await expect(page.locator('#dialogues-panel')).toBeVisible();
  await expect(page.locator('#dialogues-list')).toContainText('Добавь задачу Позвонить маме');
  await expect(page.locator('#dialogues-list')).toContainText('Готово, добавила задачу.');
  await expect(page.locator('#dialogues-list')).toContainText('Tool addItem');
  await expect(page.locator('#dialogues-list')).toContainText('Добавлена задача: Позвонить маме');

  await page.reload();
  await page.locator('#dialogues-tab-btn').click();
  await expect(page.locator('#dialogues-list')).toContainText('Добавь задачу Позвонить маме');
});

test('OpenAI Realtime session stops on pagehide teardown', async ({ page }) => {
  let keyConfigured = true;
  await page.addInitScript(() => {
    window.__realtimeLifecycle = { channelClosed: 0, peerClosed: 0, tracksStopped: 0 };
    class FakeDataChannel {
      constructor() {
        this.readyState = 'connecting';
        this.listeners = {};
      }
      addEventListener(type, handler) {
        this.listeners[type] ||= [];
        this.listeners[type].push(handler);
      }
      emit(type, payload = {}) {
        for (const handler of this.listeners[type] || []) handler(payload);
      }
      send() {}
      close() {
        if (this.readyState === 'closed') return;
        this.readyState = 'closed';
        window.__realtimeLifecycle.channelClosed += 1;
        this.emit('close');
      }
    }
    class FakePeerConnection {
      addTrack() {}
      createDataChannel() {
        this.channel = new FakeDataChannel();
        return this.channel;
      }
      async createOffer() { return { type: 'offer', sdp: 'pagehide-offer-sdp' }; }
      async setLocalDescription() {}
      async setRemoteDescription() {
        this.channel.readyState = 'open';
        this.channel.emit('open');
      }
      close() {
        window.__realtimeLifecycle.peerClosed += 1;
      }
    }
    Object.defineProperty(window, 'RTCPeerConnection', { value: FakePeerConnection, configurable: true });
    Object.defineProperty(navigator, 'mediaDevices', {
      value: {
        async getUserMedia() {
          return { getTracks: () => [{ stop() { window.__realtimeLifecycle.tracksStopped += 1; } }] };
        }
      },
      configurable: true
    });
  });
  await page.route('**/api/realtime/session', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/sdp', body: 'test-answer-sdp' });
  });
  await page.route('**/api/realtime/key/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ configured: keyConfigured, setupAvailable: false })
    });
  });

  await page.goto('');
  await page.locator('#realtime-voice-btn').click();
  await expect(page.locator('#realtime-voice-btn')).toHaveAttribute('data-state', 'active');
  await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent('pagehide')));
  await expect(page.locator('#realtime-voice-btn')).toHaveAttribute('data-state', 'idle');
  await expect.poll(() => page.evaluate(() => window.__realtimeLifecycle)).toEqual({
    channelClosed: 1,
    peerClosed: 1,
    tracksStopped: 1
  });
});

test('preview app keeps list drag and left tag gestures when task-list voice is disabled', async ({ page }) => {
  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();
  await expect(page.locator('#list-container')).toBeVisible();

  const milkWrapper = page.locator('.list-item-wrapper[data-id="milk1"]');
  const milkRow = milkWrapper.locator('.list-item');
  await expect(milkWrapper).toContainText('Молоко 3.2%');

  await triggerLeftPanelAction(page, milkRow, 'Купить');
  await expect(page.locator('#voice-overlay')).not.toHaveClass(/open/);
  await expect(milkWrapper).toContainText('Купить');

  const initialOrder = await visibleListOrder(page);
  expect(initialOrder.indexOf('milk1')).toBeLessThan(initialOrder.indexOf('bread'));

  await dragRowVertically(page, milkRow, 140);
  await expect.poll(async () => {
    const nextOrder = await visibleListOrder(page);
    return nextOrder.indexOf('milk1') > nextOrder.indexOf('bread');
  }).toBe(true);
  await expect(page.locator('#voice-overlay')).not.toHaveClass(/open/);
});

test('preview app drags through a collapsed parent as one visible row', async ({ page }) => {
  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();
  await expect(page.locator('#list-container')).toBeVisible();

  const milkWrapper = page.locator('.list-item-wrapper[data-id="milk1"]');
  const milkRow = milkWrapper.locator('.list-item');
  const breadRow = page.locator('.list-item-wrapper[data-id="bread"]').locator('.list-item');

  await breadRow.click();
  await expect(page.locator('.list-item-wrapper[data-id="borod"]')).not.toBeVisible();
  await expect(page.locator('.list-item-wrapper[data-id="stoli"]')).not.toBeVisible();

  await dragRowVertically(page, milkRow, 70);

  await expect.poll(async () => {
    const order = await visibleListOrder(page);
    return order.indexOf('bread') < order.indexOf('milk1') &&
      order.indexOf('milk1') < order.indexOf('apple');
  }).toBe(true);
  await expect(milkWrapper).toHaveAttribute('data-level', '0');
});

test('preview app clears an interrupted drag without leaving a blank slot', async ({ page }) => {
  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();
  await expect(page.locator('#list-container')).toBeVisible();

  const appleWrapper = page.locator('.list-item-wrapper[data-id="apple"]');
  const appleRow = appleWrapper.locator('.list-item');
  await appleRow.scrollIntoViewIfNeeded();
  const rowBox = await appleRow.boundingBox();
  if (!rowBox) throw new Error('No apple row box');

  const x = rowBox.x + rowBox.width / 2;
  const y = rowBox.y + rowBox.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.waitForTimeout(520);
  await page.mouse.move(x, y + 260, { steps: 8 });
  await page.evaluate(() => window.dispatchEvent(new Event('blur')));

  await expect(appleWrapper).not.toHaveClass(/is-dragging/);
  await expect.poll(async () => appleWrapper.evaluate((node) => getComputedStyle(node).transform)).toBe('none');
  await page.mouse.up();
});

test('preview app toggles a sublist closed and open from a touch tap', async ({ browser }) => {
  const context = await browser.newContext({
    hasTouch: true,
    isMobile: true,
    viewport: { width: 390, height: 844 }
  });
  const page = await context.newPage();
  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();
  await expect(page.locator('#list-container')).toBeVisible();

  const breadRow = page.locator('.list-item-wrapper[data-id="bread"]').locator('.list-item');
  const breadBox = await breadRow.boundingBox();
  if (!breadBox) throw new Error('No bread row box');

  await page.touchscreen.tap(breadBox.x + 24, breadBox.y + breadBox.height / 2);
  await expect(page.locator('.list-item-wrapper[data-id="borod"]')).toBeHidden();

  const collapsedBreadBox = await breadRow.boundingBox();
  if (!collapsedBreadBox) throw new Error('No collapsed bread row box');
  await page.touchscreen.tap(collapsedBreadBox.x + 24, collapsedBreadBox.y + collapsedBreadBox.height / 2);
  await expect(page.locator('.list-item-wrapper[data-id="borod"]')).toBeVisible();

  await context.close();
});

test('preview app tolerates small finger drift when toggling a sublist', async ({ browser }) => {
  const context = await browser.newContext({
    hasTouch: true,
    isMobile: true,
    viewport: { width: 390, height: 844 }
  });
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();
  await expect(page.locator('#list-container')).toBeVisible();

  const breadRow = page.locator('.list-item-wrapper[data-id="bread"]').locator('.list-item');
  const breadBox = await breadRow.boundingBox();
  if (!breadBox) throw new Error('No bread row box');
  const x = Math.round(breadBox.x + 24);
  const y = Math.round(breadBox.y + breadBox.height / 2);

  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x, y: y + 8 }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });

  await expect(page.locator('.list-item-wrapper[data-id="borod"]')).toBeHidden();
  await context.close();
});

test('preview app keeps a purely vertical upward drag at the original nesting level', async ({ page }) => {
  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();
  await expect(page.locator('#list-container')).toBeVisible();

  const coffeeWrapper = page.locator('.list-item-wrapper[data-id="cofee"]');
  const coffeeRow = coffeeWrapper.locator('.list-item');

  await dragRowVertically(page, coffeeRow, -180);

  await expect(coffeeWrapper).toHaveAttribute('data-level', '0');
});

test('preview app wraps long task titles to no more than two lines', async ({ page }) => {
  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();

  const longTitle = 'Длинная задача должна переноситься ровно на две строки';
  await page.getByRole('button', { name: 'Добавить задачу' }).click();
  await page.locator('#input-line1').fill(longTitle);
  await confirmModal(page);

  const line = page.locator('.list-item-wrapper', { hasText: longTitle }).locator('.item-line1');
  await expect(line).toBeVisible();
  const metrics = await line.evaluate((node) => {
    const style = getComputedStyle(node);
    return {
      clientHeight: node.clientHeight,
      lineHeight: Number.parseFloat(style.lineHeight),
      overflow: style.overflow,
      textOverflow: style.textOverflow,
      whiteSpace: style.whiteSpace
    };
  });

  expect(metrics.whiteSpace).not.toBe('nowrap');
  expect(metrics.textOverflow).not.toBe('ellipsis');
  expect(metrics.overflow).toBe('hidden');
  expect(metrics.clientHeight).toBeLessThanOrEqual(Math.ceil(metrics.lineHeight * 2) + 2);
});

test('preview app allows an upward drag to nest only after a deliberate right shift', async ({ page }) => {
  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();
  await expect(page.locator('#list-container')).toBeVisible();

  const coffeeWrapper = page.locator('.list-item-wrapper[data-id="cofee"]');
  const coffeeRow = coffeeWrapper.locator('.list-item');

  await dragRowVertically(page, coffeeRow, -180, 60);

  await expect(coffeeWrapper).toHaveAttribute('data-level', '2');
});

test('preview app imports a Workflowy shared tree from settings', async ({ page }) => {
  await mockWorkflowyExport(page);
  await page.goto('');
  const cloudflareMode = await page.evaluate(() => !window.location.hash.includes('v=local-dev') && window.location.port !== '4511');
  test.skip(cloudflareMode, 'Workflowy Worker outbound fetch is covered by document-core tests.');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();

  await page.locator('#settings-btn').click();
  await expect(page.locator('#settings-overlay')).toHaveClass(/open/);
  await page.locator('#workflowy-url-input').fill('https://workflowy.com/s/task-tree/iq43ak7FYqEEO1uO');
  await page.locator('#workflowy-import-btn').click();

  await expect(page.locator('#workflowy-import-status')).toContainText('Импорт');
  await expect(page.locator('#list-container')).toContainText('task tree');
  await expect(page.locator('#list-container')).toContainText('Child task');
  await expect(page.locator('#list-container')).toContainText('Nested task');
});

test('preview app keeps long-press voice and uses the shared left menu to postpone focus tasks', async ({ page }) => {
  await page.addInitScript(() => {
    window.__voiceTest = { phrase: 'шум' };
  });

  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();
  await expect(page.locator('#list-container')).toBeVisible();
  await expect(page.locator('.list-item-wrapper', { hasText: 'Входящие' })).toBeVisible();

  const milkRowInList = page.locator('.list-item-wrapper[data-id="milk1"]').locator('.list-item');
  const milkBox = await milkRowInList.boundingBox();
  if (!milkBox) throw new Error('No milk row box');

  await page.mouse.move(milkBox.x + milkBox.width / 2, milkBox.y + milkBox.height / 2);
  await page.mouse.down();
  await page.waitForTimeout(520);
  await expect(page.locator('#voice-overlay')).not.toHaveClass(/open/);
  await page.mouse.up();
  await page.waitForTimeout(300);

  const viewport = page.viewportSize();
  const leftSwipeDistance = Math.max(320, Math.floor((viewport?.width || 1280) * 0.28));

  await page.mouse.move(milkBox.x + milkBox.width - 24, milkBox.y + milkBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(milkBox.x + milkBox.width - leftSwipeDistance, milkBox.y + milkBox.height / 2, { steps: 8 });
  await expect(page.locator('#voice-overlay')).not.toHaveClass(/open/);
  await page.mouse.up();

  await page.locator('#frontier-tab-btn').click();
  await expect(page.locator('#frontier-tab-btn')).toHaveClass(/active/);
  const goldenRow = page.locator('.list-item-wrapper[data-id="goldn"]').locator('.list-item');
  const goldenBox = await goldenRow.boundingBox();
  if (!goldenBox) throw new Error('No Golden frontier row box');

  await page.mouse.move(goldenBox.x + goldenBox.width / 2, goldenBox.y + goldenBox.height / 2);
  await page.mouse.down();
  await page.waitForTimeout(520);
  await expect(page.locator('#voice-overlay')).toHaveClass(/open/);
  await page.mouse.up();
  await expect(page.locator('#voice-overlay')).not.toHaveClass(/open/);

  await triggerLeftPanelAction(page, goldenRow, 'Неделя');
  await expect(page.locator('#voice-overlay')).not.toHaveClass(/open/);
  await expect(page.locator('.list-item-wrapper[data-id="goldn"] .item-index')).toHaveText('7');

  const grannyRow = page.locator('.list-item-wrapper[data-id="grnsm"]').locator('.list-item');
  await triggerLeftPanelAction(page, grannyRow, 'Завтра');
  await expect(page.locator('.list-item-wrapper[data-id="grnsm"] .item-index')).toHaveText('1');

  const focusedOrder = await visibleListOrder(page);
  expect(focusedOrder.indexOf('grnsm')).toBeLessThan(focusedOrder.indexOf('goldn'));
});

test('preview app can mark a task as Info from the right menu and hides it from frontier', async ({ page }) => {
  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();
  await expect(page.locator('#list-container')).toBeVisible();

  const coffeeWrapper = page.locator('.list-item-wrapper[data-id="cofee"]');
  await expect(coffeeWrapper).toContainText('Кофе');

  await triggerRightPanelActionUntil(page, coffeeWrapper.locator('.list-item'), 'Info', async () => (
    (await coffeeWrapper.textContent()).includes('Info')
  ));
  await expect(coffeeWrapper).toContainText('Info');

  await page.locator('#frontier-tab-btn').click();
  await expect(page.locator('#frontier-tab-btn')).toHaveClass(/active/);
  await expect(page.locator('.list-item-wrapper[data-id="cofee"]')).toHaveCount(0);

  await page.locator('#frontier-tab-btn').click();
  await expect(coffeeWrapper).toContainText('Info');
});

test('preview app logs unrecognized voice fallback utterances', async ({ page }) => {
  await page.addInitScript(() => {
    window.__voiceTest = { phrase: 'позвонить Ване' };
  });

  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();
  await page.locator('#frontier-tab-btn').click();
  await expect(page.locator('#frontier-tab-btn')).toHaveClass(/active/);

  const frontierRow = page.locator('.list-item-wrapper[data-id="goldn"]').locator('.list-item');
  const box = await frontierRow.boundingBox();
  if (!box) throw new Error('No frontier row box');

  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.waitForTimeout(520);
  await expect(page.locator('#voice-overlay')).toHaveClass(/open/);
  await page.mouse.up();
  await expect(page.locator('#voice-overlay')).not.toHaveClass(/open/);

  await page.locator('#view-toggle-btn').click();
  await expect(page.locator('#action-log-list')).toContainText('Нераспознано: позвонить Ване');
  await expect(page.locator('#action-log-list')).toContainText('позвонить Ване');
});

test('preview app positions voice overlay near the pointer and selects a candidate by movement', async ({ page }) => {
  await page.addInitScript(() => {
    window.__voiceTest = { phrase: 'добавь молоко' };
  });

  await page.goto('');
  await page.evaluate(() => window.localStorage.clear());
  await page.reload();
  await page.locator('#frontier-tab-btn').click();
  await expect(page.locator('#frontier-tab-btn')).toHaveClass(/active/);

  const frontierRow = page.locator('.list-item-wrapper', { hasText: 'Голден' }).locator('.list-item').first();
  const frontierBox = await frontierRow.boundingBox();
  if (!frontierBox) throw new Error('No frontier row box');
  const anchor = { x: frontierBox.x + frontierBox.width / 2, y: frontierBox.y + frontierBox.height / 2 };

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
