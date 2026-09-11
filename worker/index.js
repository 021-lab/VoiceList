import { DurableObject } from 'cloudflare:workers';
import { z } from 'zod';

import { seedState } from '../list-data.js';
import { LIST_MANAGER_HTML } from './generated-html.js';
import { createDocumentCore } from './list-document-core.js';
import { handleOpenAIKeySetup, handleOpenAIKeyStatus } from './openai-key-setup.js';
import { getDefaultRealtimeSystemPrompt, handleOpenAIRealtimeSession } from './openai-realtime.js';
import { taskFrontierFromItems } from './task-frontier.js';
import { taskTreeFromItems } from './task-tree.js';

const STORAGE_KEY = 'voicelist.document.v1';
const OPENAI_API_KEY_STORAGE_KEY = 'voicelist.openai-api-key.v1';
const OPENAI_SETUP_USED_STORAGE_KEY = 'voicelist.openai-setup-used.v1';
const OPENAI_SYSTEM_PROMPT_STORAGE_KEY = 'voicelist.openai-system-prompt.v1';
const REALTIME_DIAGNOSTICS_STORAGE_KEY = 'voicelist.realtime-diagnostics.v1';
const MAX_REALTIME_DIAGNOSTICS = 50;
const MCP_PROTOCOL_VERSION = '2025-11-25';
const MCP_SUPPORTED_PROTOCOLS = new Set(['2026-07-28', '2025-11-25', '2025-06-18', '2025-03-26']);
const TASK_STATUSES = ['Open', 'Focus', 'Pause', 'Done', 'Archive', 'Info'];
const MCP_CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'content-type, accept, mcp-protocol-version',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Max-Age': '86400'
};
const OAUTH_DISCOVERY_PATHS = new Set([
  '/.well-known/oauth-authorization-server',
  '/.well-known/oauth-protected-resource'
]);

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

function notFound() {
  return new Response('Not found', {
    status: 404,
    headers: {
      'Cache-Control': 'no-store',
      'Content-Type': 'text/plain; charset=utf-8'
    }
  });
}

function mcpJson(data, init = {}) {
  return json(data, {
    ...init,
    headers: {
      ...MCP_CORS_HEADERS,
      ...(init.headers || {})
    }
  });
}

function mcpEmpty(status = 202) {
  return new Response(null, {
    status,
    headers: MCP_CORS_HEADERS
  });
}

class ToolCallError extends Error {}

const EmptyInputSchema = z.object({}).strict().describe('No input.');
const IdInputSchema = z.object({
  id: z.string().trim().min(1).describe('Exact VoiceList task id.')
}).strict();
const LineInputSchema = z.object({
  line1: z.string().trim().min(1).describe('Task title.')
}).strict();
const ChildInputSchema = z.object({
  parentId: z.string().trim().min(1).describe('Exact parent task id.'),
  line1: z.string().trim().min(1).describe('Task title.')
}).strict();
const StatusInputSchema = z.object({
  id: z.string().trim().min(1).describe('Exact VoiceList task id.'),
  status: z.enum(TASK_STATUSES).describe('Task status.')
}).strict();
const MoveInputSchema = z.object({
  id: z.string().trim().min(1).describe('Exact moved task id.'),
  parentId: z.string().trim().min(1).nullable().describe('Exact new parent task id, or null to move to root.')
}).strict();
const RenameInputSchema = z.object({
  id: z.string().trim().min(1).describe('Exact VoiceList task id.'),
  line1: z.string().trim().min(1).describe('New task title.')
}).strict();

function jsonSchema(schema) {
  return z.toJSONSchema(schema);
}

const MCP_TOOLS = [
  {
    name: 'voicelist_get_task_tree',
    description: 'Return the VoiceList task tree without archived tasks.',
    schema: EmptyInputSchema,
    inputSchema: jsonSchema(EmptyInputSchema),
    annotations: { readOnlyHint: true },
    mutates: false
  },
  {
    name: 'voicelist_get_task_subgraph',
    description: 'Return path, task attributes, and direct non-archived children for one exact task id.',
    schema: IdInputSchema,
    inputSchema: jsonSchema(IdInputSchema),
    annotations: { readOnlyHint: true },
    mutates: false
  },
  {
    name: 'voicelist_add_root_task',
    description: 'Add a new root task.',
    schema: LineInputSchema,
    inputSchema: jsonSchema(LineInputSchema),
    annotations: { destructiveHint: false, openWorldHint: true },
    mutates: true
  },
  {
    name: 'voicelist_add_child_task',
    description: 'Add a new child task under an exact parent id.',
    schema: ChildInputSchema,
    inputSchema: jsonSchema(ChildInputSchema),
    annotations: { destructiveHint: false, openWorldHint: true },
    mutates: true
  },
  {
    name: 'voicelist_add_task_note',
    description: 'Add an informational child note under an exact parent id.',
    schema: ChildInputSchema,
    inputSchema: jsonSchema(ChildInputSchema),
    annotations: { destructiveHint: false, openWorldHint: true },
    mutates: true
  },
  {
    name: 'voicelist_set_status',
    description: 'Set a task status by exact id.',
    schema: StatusInputSchema,
    inputSchema: jsonSchema(StatusInputSchema),
    annotations: { idempotentHint: true, openWorldHint: true },
    mutates: true
  },
  {
    name: 'voicelist_rename_task',
    description: 'Rename a task by exact id.',
    schema: RenameInputSchema,
    inputSchema: jsonSchema(RenameInputSchema),
    annotations: { destructiveHint: false, openWorldHint: true },
    mutates: true
  },
  {
    name: 'voicelist_move_task',
    description: 'Move a task under another exact parent id, or to root with parentId null.',
    schema: MoveInputSchema,
    inputSchema: jsonSchema(MoveInputSchema),
    annotations: { destructiveHint: false, openWorldHint: true },
    mutates: true
  },
  {
    name: 'voicelist_delete_task',
    description: 'Archive a task by exact id. The node is not physically deleted.',
    schema: IdInputSchema,
    inputSchema: jsonSchema(IdInputSchema),
    annotations: { idempotentHint: true, destructiveHint: true, openWorldHint: true },
    mutates: true
  },
  {
    name: 'voicelist_undo',
    description: 'Undo the last server-applied mutating action.',
    schema: EmptyInputSchema,
    inputSchema: jsonSchema(EmptyInputSchema),
    annotations: { destructiveHint: true, openWorldHint: true },
    mutates: true
  }
];
const MCP_TOOL_BY_NAME = new Map(MCP_TOOLS.map((tool) => [tool.name, tool]));

function formatZodError(error) {
  return error.issues.map((issue) => {
    const path = issue.path.length ? issue.path.join('.') : 'input';
    return `${path}: ${issue.message}`;
  }).join('; ');
}

function parseToolArguments(tool, args) {
  const result = tool.schema.safeParse(args || {});
  if (!result.success) throw new ToolCallError(`некорректные аргументы: ${formatZodError(result.error)}`);
  return result.data;
}

function requireTask(core, id) {
  const task = core.getTaskById(id);
  if (!task) throw new ToolCallError(`узел с id "${id}" не найден`);
  return task;
}

async function applyMcpCommand(core, input) {
  const ack = await core.applyCommand(input, {
    message: {
      clientKey: 'mcp',
      seq: Date.now()
    }
  });
  if (ack.status !== 'applied') throw new ToolCallError(ack.reason || 'команда не применена');
  const task = ack.newTarget ? core.getTaskById(ack.newTarget) : null;
  return { ack, task };
}

async function executeMcpTool(core, name, args) {
  const tool = MCP_TOOL_BY_NAME.get(name);
  if (!tool) throw new ToolCallError(`неизвестный тул: ${name}`);
  const input = parseToolArguments(tool, args);

  if (name === 'voicelist_get_task_tree') {
    return { tasks: core.getActiveTaskTree() };
  }

  if (name === 'voicelist_get_task_subgraph') {
    const result = core.getTaskSubgraph(input.id);
    if (result.status !== 'found') throw new ToolCallError(`узел с id "${input.id}" не найден`);
    return result.subgraph;
  }

  if (name === 'voicelist_add_root_task') {
    const result = await applyMcpCommand(core, {
      actId: 'list',
      actType: 'list',
      command: 'addItem',
      payload: { line1: input.line1, line2: '' },
      source: 'mcp'
    });
    return result.task;
  }

  if (name === 'voicelist_add_child_task' || name === 'voicelist_add_task_note') {
    requireTask(core, input.parentId);
    const result = await applyMcpCommand(core, {
      actId: input.parentId,
      actType: 'task',
      command: 'addChild',
      payload: {
        line1: input.line1,
        line2: '',
        ...(name === 'voicelist_add_task_note' ? { status: 'Info' } : {})
      },
      source: 'mcp'
    });
    return result.task;
  }

  if (name === 'voicelist_set_status') {
    requireTask(core, input.id);
    const result = await applyMcpCommand(core, {
      actId: input.id,
      actType: 'task',
      command: 'setStatus',
      payload: { status: input.status },
      source: 'mcp'
    });
    return result.task;
  }

  if (name === 'voicelist_rename_task') {
    const task = requireTask(core, input.id);
    const result = await applyMcpCommand(core, {
      actId: input.id,
      actType: 'task',
      command: 'editItem',
      payload: { line1: input.line1, line2: task.line2 || '' },
      source: 'mcp'
    });
    return result.task;
  }

  if (name === 'voicelist_move_task') {
    requireTask(core, input.id);
    if (input.parentId !== null) requireTask(core, input.parentId);
    const result = await applyMcpCommand(core, {
      actId: input.id,
      actType: 'task',
      command: 'setParent',
      payload: { parentId: input.parentId },
      source: 'mcp'
    });
    return result.task;
  }

  if (name === 'voicelist_delete_task') {
    requireTask(core, input.id);
    const result = await applyMcpCommand(core, {
      actId: input.id,
      actType: 'task',
      command: 'setStatus',
      payload: { status: 'Archive' },
      source: 'mcp'
    });
    return result.task;
  }

  if (name === 'voicelist_undo') {
    const result = await core.undoLastAction({
      clientKey: 'mcp',
      seq: Date.now(),
      source: 'mcp'
    });
    if (result.status === 'error') throw new ToolCallError(result.error);
    if (result.status !== 'applied') throw new ToolCallError(result.ack?.reason || 'откат не применен');
    return {
      undone: result.undone,
      task: result.node
    };
  }

  throw new ToolCallError(`неизвестный тул: ${name}`);
}

function mcpToolResult(value) {
  return {
    content: [{ type: 'text', text: JSON.stringify(value, null, 2) }],
    structuredContent: value
  };
}

function mcpToolError(message) {
  return {
    isError: true,
    content: [{ type: 'text', text: message }]
  };
}

function jsonRpcResult(id, result) {
  return mcpJson({ jsonrpc: '2.0', id, result });
}

function jsonRpcError(id, code, message) {
  return mcpJson({
    jsonrpc: '2.0',
    id,
    error: { code, message }
  });
}

function protocolVersion(params = {}) {
  const requested = params?.protocolVersion;
  return MCP_SUPPORTED_PROTOCOLS.has(requested) ? requested : MCP_PROTOCOL_VERSION;
}

function listMcpTools() {
  return {
    tools: MCP_TOOLS.map(({ name, description, inputSchema, annotations }) => ({
      name,
      description,
      inputSchema,
      annotations
    }))
  };
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

  async getTaskFrontier() {
    const core = await this.ensureCore();
    return taskFrontierFromItems(core.getSnapshot().content.snapshot.items);
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

  async recordRealtimeDiagnostics(rawEntry) {
    const number = (value) => Number.isFinite(value) && value >= 0 && value <= 120_000 ? Math.round(value) : null;
    const entry = {
      at: new Date().toISOString(),
      outcome: ['active', 'cancelled', 'error'].includes(rawEntry?.outcome) ? rawEntry.outcome : 'error',
      prewarmedTransport: Boolean(rawEntry?.prewarmedTransport),
      totalMs: number(rawEntry?.totalMs),
      microphoneMs: number(rawEntry?.microphoneMs),
      offerMs: number(rawEntry?.offerMs),
      localDescriptionMs: number(rawEntry?.localDescriptionMs),
      sessionRequestMs: number(rawEntry?.sessionRequestMs),
      remoteDescriptionMs: number(rawEntry?.remoteDescriptionMs),
      dataChannelMs: number(rawEntry?.dataChannelMs),
      failedStage: ['microphone', 'session', 'transport'].includes(rawEntry?.failedStage) ? rawEntry.failedStage : null
    };
    const existing = await this.ctx.storage.get(REALTIME_DIAGNOSTICS_STORAGE_KEY);
    const entries = Array.isArray(existing) ? existing : [];
    entries.unshift(entry);
    await this.ctx.storage.put(REALTIME_DIAGNOSTICS_STORAGE_KEY, entries.slice(0, MAX_REALTIME_DIAGNOSTICS));
    return entry;
  }

  async getRealtimeDiagnostics() {
    const entries = await this.ctx.storage.get(REALTIME_DIAGNOSTICS_STORAGE_KEY);
    return Array.isArray(entries) ? entries : [];
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
    const url = new URL(request.url);
    if (url.pathname === '/mcp') {
      return this.handleMcpRequest(request);
    }

    const upgrade = request.headers.get('Upgrade') || '';
    if (upgrade.toLowerCase() !== 'websocket') return json({ error: 'Expected WebSocket upgrade' }, { status: 426 });

    const pair = new WebSocketPair();
    const [client, server] = Object.values(pair);
    this.ctx.acceptWebSocket(server);
    const core = await this.ensureCore();
    server.send(JSON.stringify({ type: 'state', state: core.getSnapshot() }));
    return new Response(null, { status: 101, webSocket: client });
  }

  async handleMcpRequest(request) {
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 204,
        headers: MCP_CORS_HEADERS
      });
    }
    if (request.method !== 'POST') {
      return mcpJson({ error: 'Method not allowed' }, {
        status: 405,
        headers: { Allow: 'POST, OPTIONS' }
      });
    }

    let rpc;
    try {
      rpc = await request.json();
    } catch {
      return jsonRpcError(null, -32700, 'Parse error');
    }

    if (Array.isArray(rpc)) return jsonRpcError(null, -32600, 'Batch requests are not supported');
    if (!rpc || rpc.jsonrpc !== '2.0' || typeof rpc.method !== 'string') {
      return jsonRpcError(rpc?.id ?? null, -32600, 'Invalid Request');
    }

    const hasId = Object.prototype.hasOwnProperty.call(rpc, 'id');
    if (!hasId) {
      if (rpc.method === 'notifications/initialized') return mcpEmpty(202);
      return mcpEmpty(202);
    }

    if (rpc.method === 'initialize') {
      return jsonRpcResult(rpc.id, {
        protocolVersion: protocolVersion(rpc.params),
        capabilities: { tools: {} },
        serverInfo: {
          name: 'voicelist-worker-mcp',
          version: '1.0.0'
        }
      });
    }

    if (rpc.method === 'tools/list') {
      return jsonRpcResult(rpc.id, listMcpTools());
    }

    if (rpc.method === 'tools/call') {
      const core = await this.ensureCore();
      const name = String(rpc.params?.name || '');
      const tool = MCP_TOOL_BY_NAME.get(name);
      try {
        const result = await executeMcpTool(core, name, rpc.params?.arguments || {});
        if (tool?.mutates) {
          await this.persist();
          this.broadcastState(core.getSnapshot());
        }
        return jsonRpcResult(rpc.id, mcpToolResult(result));
      } catch (error) {
        if (error instanceof ToolCallError) return jsonRpcResult(rpc.id, mcpToolError(error.message));
        return jsonRpcResult(rpc.id, mcpToolError('внутренняя ошибка выполнения тула'));
      }
    }

    return jsonRpcError(rpc.id, -32601, 'Method not found');
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

    if (OAUTH_DISCOVERY_PATHS.has(url.pathname)) {
      return notFound();
    }

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

    if (url.pathname === '/mcp') {
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

    if (url.pathname === '/api/realtime/diagnostics') {
      if (request.method === 'POST') {
        const origin = request.headers.get('Origin');
        if (origin && origin !== url.origin) return json({ error: 'Origin not allowed' }, { status: 403 });
        const body = await readRequestJson(request);
        if (!body || typeof body !== 'object') return json({ error: 'Invalid diagnostics' }, { status: 400 });
        await documentStub(env).recordRealtimeDiagnostics(body);
        return json({ recorded: true }, { status: 202 });
      }
      if (request.method === 'GET') {
        if (!env.REALTIME_DIAGNOSTICS_TOKEN || request.headers.get('X-VoiceList-Diagnostics-Token') !== env.REALTIME_DIAGNOSTICS_TOKEN) {
          return new Response('Not found', { status: 404 });
        }
        return json({ entries: await documentStub(env).getRealtimeDiagnostics() });
      }
      return json({ error: 'Method not allowed' }, { status: 405 });
    }

    if (url.pathname === '/api/tasks/tree.json') {
      return json({ tasks: await documentStub(env).getTaskTree() });
    }

    if (url.pathname === '/api/tasks/frontier.json') {
      if (request.method !== 'GET') return json({ error: 'Method not allowed' }, { status: 405 });
      return json({ frontier: await documentStub(env).getTaskFrontier() });
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

    return notFound();
  }
};
