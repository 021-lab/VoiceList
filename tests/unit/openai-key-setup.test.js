import { describe, expect, test, vi } from 'vitest';

import { handleOpenAIKeySetup, handleOpenAIKeyStatus } from '../../worker/openai-key-setup.js';

describe('OpenAI key server setup', () => {
  test('reports only setup state and never returns the saved key', async () => {
    const response = await handleOpenAIKeyStatus({ configured: true, setupAvailable: false });
    await expect(response.json()).resolves.toEqual({ configured: true, setupAvailable: false });
    expect(response.headers.get('Cache-Control')).toBe('no-store');
  });

  test('accepts a same-origin key from Settings without a setup link', async () => {
    const configureKey = vi.fn(async () => true);
    const request = new Request('https://vlist-dev.smileme.ai/api/realtime/key', {
      method: 'POST',
      headers: { Origin: 'https://vlist-dev.smileme.ai', 'Content-Type': 'application/json' },
      body: JSON.stringify({
        apiKey: 'sk-example-server-key-1234567890'
      })
    });

    const response = await handleOpenAIKeySetup(request, {
      configureKey
    });

    expect(response.status).toBe(201);
    expect(configureKey).toHaveBeenCalledWith('sk-example-server-key-1234567890');
    expect(await response.text()).not.toContain('sk-example');
  });

  test('rejects cross-origin, invalid-key, and already-used setup attempts', async () => {
    const body = JSON.stringify({ apiKey: 'sk-example-server-key-1234567890' });
    const crossOrigin = new Request('https://vlist-dev.smileme.ai/api/realtime/key', {
      method: 'POST',
      headers: { Origin: 'https://example.com', 'Content-Type': 'application/json' },
      body
    });
    expect((await handleOpenAIKeySetup(crossOrigin, {
      configureKey: vi.fn()
    })).status).toBe(403);

    const invalidKey = new Request('https://vlist-dev.smileme.ai/api/realtime/key', {
      method: 'POST',
      headers: { Origin: 'https://vlist-dev.smileme.ai', 'Content-Type': 'application/json' },
      body: JSON.stringify({ apiKey: 'not-an-openai-key' })
    });
    expect((await handleOpenAIKeySetup(invalidKey, {
      configureKey: vi.fn()
    })).status).toBe(400);

    const alreadyUsed = new Request('https://vlist-dev.smileme.ai/api/realtime/key', {
      method: 'POST',
      headers: { Origin: 'https://vlist-dev.smileme.ai', 'Content-Type': 'application/json' },
      body
    });
    expect((await handleOpenAIKeySetup(alreadyUsed, {
      configureKey: vi.fn(async () => false)
    })).status).toBe(409);
  });
});
