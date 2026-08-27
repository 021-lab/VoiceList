import { DurableObject } from 'cloudflare:workers';

import { seedState } from '../list-data.js';
import { LIST_MANAGER_HTML } from './generated-html.js';
import { createDocumentCore } from './list-document-core.js';
import { handleOpenAIKeySetup, handleOpenAIKeyStatus } from './openai-key-setup.js';
import { getDefaultRealtimeSystemPrompt, handleOpenAIRealtimeSession } from './openai-realtime.js';
import { taskTreeFromItems } from './task-tree.js';

const STORAGE_KEY = 'voicelist.document.v1';
const OPENAI_API_KEY_STORAGE_KEY = 'voicelist.openai-api-key.v1';
const OPENAI_SETUP_USED_STORAGE_KEY = 'voicelist.openai-setup-used.v1';
const OPENAI_SYSTEM_PROMPT_STORAGE_KEY = 'voicelist.openai-system-prompt.v1';

function json(data, init = {}) {
  return Response.json(data, {
    ...init,
    headers: {
      'Cache-Control': 'no-store',
      ...(init.headers || {})
    }
  });
}

async function readRequestJson(request) {
  try {
    return await request.json();
  } catch {
    return null;
  }
}

export class ListDocumentDO extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    this.core = null;
    this.queue = Promise.resolve();
    this.openAIApiKeyPromise = null;
    this.openAISystemPromptPromise = null;
  }

  async ensureCore() {
    if (this.core) return this.core;
    const initialState = await this.ctx.storage.get(STORAGE_KEY);
    this.core = createDocumentCore({
      seedState,
      initialState,
      openRouterApiKey: this.env.OPENROUTER_API_KEY || '',
      openRouterModel: this.env.OPENROUTER_MODEL || 'openai/gpt-4.1-mini'
    });
    await this.core.init();
    return this.core;
  }

  async persist() {
    await this.ctx.storage.put(STORAGE_KEY, this.core.exportState());
  }

  async reset() {
    this.core = createDocumentCore({
      seedState,
      openRouterApiKey: this.env.OPENROUTER_API_KEY || '',
      openRouterModel: this.env.OPENROUTER_MODEL || 'openai/gpt-4.1-mini'
    });
    await this.core.init();
    await this.persist();
    const state = this.core.getSnapshot();
    this.broadcastState(state);
    return state;
  }

  async getOpenAIApiKey() {
    if (!this.openAIApiKeyPromise) {
      this.openAIApiKeyPromise = this.ctx.storage
        .get(OPENAI_API_KEY_STORAGE_KEY)
        .then((value) => value || '');
    }
    return await this.openAIApiKeyPromise;
  }

  async getOpenAISystemPrompt() {
    if (!this.openAISystemPromptPromise) {
      this.openAISystemPromptPromise = this.ctx.storage
        .get(OPENAI_SYSTEM_PROMPT_STORAGE_KEY)
        .then((value) => value || '');
    }
    return await this.openAISystemPromptPromise;
  }

  async isOpenAIKeyConfigured() {
    return Boolean(await this.getOpenAIApiKey());
  }

  async getTaskTree() {
    const core = await this.ensureCore();
    return taskTreeFromItems(core.getSnapshot().content.snapshot.items);
  }

  async configureOpenAIApiKey(apiKey) {
    if (await this.ctx.storage.get(OPENAI_SETUP_USED_STORAGE_KEY)) return false;
    await this.ctx.storage.put({
      [OPENAI_API_KEY_STORAGE_KEY]: apiKey,
      [OPENAI_SETUP_USED_STORAGE_KEY]: true
    });
    this.openAIApiKeyPromise = Promise.resolve(apiKey);
    return true;
  }

  async configureOpenAISystemPrompt(prompt) {
    const value = String(prompt || '').trim();
    await this.ctx.storage.put(OPENAI_SYSTEM_PROMPT_STORAGE_KEY, value);
    this.openAISystemPromptPromise = Promise.resolve(value);
    return true;
  }

  broadcastState(state) {
    const payload = JSON.stringify({ type: 'state', state });
    for (const ws of this.ctx.getWebSockets()) {
      try {
        ws.send(payload);
      } catch {
        // Stale sockets disappear from getWebSockets after close completion.
      }
    }
  }

  async fetch(request) {
    const upgrade = request.headers.get('Upgrade') || '';
    if (upgrade.toLowerCase() !== 'websocket') return json({ error: 'Expected WebSocket upgrade' }, { status: 426 });

    const pair = new WebSocketPair();
    const [client, server] = Object.values(pair);
    this.ctx.acceptWebSocket(server);
    const core = await this.ensureCore();
    server.send(JSON.stringify({ type: 'state', state: core.getSnapshot() }));
    return new Response(null, { status: 101, webSocket: client });
  }

  async processWebSocketMessage(ws, rawMessage) {
    const core = await this.ensureCore();
    let message;
    try {
      message = JSON.parse(String(rawMessage));
    } catch {
      ws.send(JSON.stringify({
        type: 'ack',
        ack: { seq: null, id: null, status: 'rejected', reason: 'Invalid JSON message', newTarget: null }
      }));
      return;
    }

    if (message.type === 'hello') {
      ws.send(JSON.stringify({ type: 'state', state: core.getSnapshot() }));
      return;
    }

    const result = await core.handleClientMessage(message);
    try {
      await this.persist();
    } catch (error) {
      ws.send(JSON.stringify({
        type: 'ack',
        ack: {
          seq: result.ack?.seq ?? message.seq ?? null,
          id: null,
          status: 'rejected',
          reason: `Persist failed: ${error.message}`,
          newTarget: null
        }
      }));
      return;
    }
    ws.send(JSON.stringify({ type: 'ack', ack: result.ack }));
    this.broadcastState(result.state);
  }

  webSocketMessage(ws, message) {
    this.queue = this.queue.then(() => this.processWebSocketMessage(ws, message));
    this.ctx.waitUntil(this.queue);
  }

  webSocketClose() {}

  webSocketError() {}
}

function documentStub(env) {
  return env.LIST_DOCUMENT.getByName('main');
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === '/health') {
      return new Response('ok\n', {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'Cache-Control': 'no-store'
        }
      });
    }

    if (url.pathname === '/ws') {
      return documentStub(env).fetch(request);
    }

    if (url.pathname === '/api/realtime/key/status') {
      const configured = Boolean(env.OPENAI_API_KEY) || await documentStub(env).isOpenAIKeyConfigured();
      return handleOpenAIKeyStatus({
        configured,
        setupAvailable: !configured
      });
    }

    if (url.pathname === '/api/realtime/key') {
      return handleOpenAIKeySetup(request, {
        configureKey: (apiKey) => documentStub(env).configureOpenAIApiKey(apiKey)
      });
    }

    if (url.pathname === '/api/realtime/prompt') {
      if (request.method === 'GET') {
        const prompt = await documentStub(env).getOpenAISystemPrompt();
        return json({ prompt: prompt || getDefaultRealtimeSystemPrompt() });
      }
      if (request.method === 'POST') {
        const body = await readRequestJson(request);
        if (!body || typeof body.prompt !== 'string') return json({ error: 'Invalid prompt' }, { status: 400 });
        await documentStub(env).configureOpenAISystemPrompt(body.prompt);
        return json({ configured: true });
      }
      return json({ error: 'Method not allowed' }, { status: 405 });
    }

    if (url.pathname === '/api/realtime/session') {
      const [storedApiKey, systemPrompt] = await Promise.all([
        env.OPENAI_API_KEY ? Promise.resolve('') : documentStub(env).getOpenAIApiKey(),
        documentStub(env).getOpenAISystemPrompt()
      ]);
      const apiKey = env.OPENAI_API_KEY || storedApiKey;
      return handleOpenAIRealtimeSession(request, env, { apiKey, systemPrompt });
    }

    if (url.pathname === '/api/tasks/tree.json') {
      return json({ tasks: await documentStub(env).getTaskTree() });
    }

    if (url.pathname === '/reset' && request.method === 'POST') {
      const token = request.headers.get('X-VoiceList-Test-Reset') || '';
      if (!env.TEST_RESET_TOKEN || token !== env.TEST_RESET_TOKEN) return new Response('Not found', { status: 404 });
      const state = await documentStub(env).reset();
      return json({ state });
    }

    if (url.pathname === '/' || url.pathname === '/index.html') {
      return new Response(LIST_MANAGER_HTML, {
        headers: {
          'Content-Type': 'text/html; charset=utf-8',
          'Cache-Control': 'no-store'
        }
      });
    }

    return new Response('Not found', { status: 404 });
  }
};
