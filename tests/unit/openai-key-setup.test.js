import { describe, expect, test, vi } from 'vitest';

import { handleOpenAIKeySetup, handleOpenAIKeyStatus } from '../../worker/openai-key-setup.js';

async function sha256(value) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

describe('OpenAI key server setup', () => {
  test('reports only setup state and never returns the saved key', async () => {
    const response = await handleOpenAIKeyStatus({ configured: true, setupAvailable: false });
    await expect(response.json()).resolves.toEqual({ configured: true, setupAvailable: false });
    expect(response.headers.get('Cache-Control')).toBe('no-store');
  });

  test('accepts a same-origin key once with the hashed setup token', async () => {
    const setupToken = 'one-time-mobile-setup-token-1234567890';
    const configureKey = vi.fn(async () => true);
    const request = new Request('https://vlist-dev.smileme.ai/api/realtime/key', {
      method: 'POST',
      headers: { Origin: 'https://vlist-dev.smileme.ai', 'Content-Type': 'application/json' },
      body: JSON.stringify({
        apiKey: 'sk-example-server-key-1234567890',
        setupToken
      })
    });

    const response = await handleOpenAIKeySetup(request, {
      setupTokenHash: await sha256(setupToken),
      configureKey
    });

    expect(response.status).toBe(201);
    expect(configureKey).toHaveBeenCalledWith('sk-example-server-key-1234567890');
    expect(await response.text()).not.toContain('sk-example');
  });

  test('rejects cross-origin, invalid-token, and already-used setup attempts', async () => {
    const setupToken = 'one-time-mobile-setup-token-1234567890';
    const body = JSON.stringify({ apiKey: 'sk-example-server-key-1234567890', setupToken });
    const crossOrigin = new Request('https://vlist-dev.smileme.ai/api/realtime/key', {
      method: 'POST',
      headers: { Origin: 'https://example.com', 'Content-Type': 'application/json' },
      body
    });
    expect((await handleOpenAIKeySetup(crossOrigin, {
      setupTokenHash: await sha256(setupToken),
      configureKey: vi.fn()
    })).status).toBe(403);

    const invalidToken = new Request('https://vlist-dev.smileme.ai/api/realtime/key', {
      method: 'POST',
      headers: { Origin: 'https://vlist-dev.smileme.ai', 'Content-Type': 'application/json' },
      body: JSON.stringify({ apiKey: 'sk-example-server-key-1234567890', setupToken: 'wrong-token-that-is-long-enough-123456' })
    });
    expect((await handleOpenAIKeySetup(invalidToken, {
      setupTokenHash: await sha256(setupToken),
      configureKey: vi.fn()
    })).status).toBe(403);

    const alreadyUsed = new Request('https://vlist-dev.smileme.ai/api/realtime/key', {
      method: 'POST',
      headers: { Origin: 'https://vlist-dev.smileme.ai', 'Content-Type': 'application/json' },
      body
    });
    expect((await handleOpenAIKeySetup(alreadyUsed, {
      setupTokenHash: await sha256(setupToken),
      configureKey: vi.fn(async () => false)
    })).status).toBe(409);
  });
});
