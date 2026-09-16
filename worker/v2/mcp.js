import { z } from 'zod';
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
const LOCAL_MCP_HOSTS = new Set(['localhost', '127.0.0.1', '::1']);

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

function parseAllowedMcpHosts(env) {
  const value = typeof env.MCP_ALLOWED_HOSTS === 'string' ? env.MCP_ALLOWED_HOSTS : '';
  return new Set(value
    .split(',')
    .map((host) => host.trim().toLowerCase())
    .filter(Boolean));
}

function isMcpHostAllowed(request, env) {
  const hostname = new URL(request.url).hostname.toLowerCase();
  const allowedHosts = parseAllowedMcpHosts(env);
  if (allowedHosts.size === 0) return LOCAL_MCP_HOSTS.has(hostname);
  return allowedHosts.has(hostname);
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

export async function handleMcpRequest(request, core) {
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
      const name = String(rpc.params?.name || '');
      const tool = MCP_TOOL_BY_NAME.get(name);
      try {
        const result = await executeMcpTool(core, name, rpc.params?.arguments || {});
        if (tool?.mutates) {
        }
        return jsonRpcResult(rpc.id, mcpToolResult(result));
      } catch (error) {
        if (error instanceof ToolCallError) return jsonRpcResult(rpc.id, mcpToolError(error.message));
        return jsonRpcResult(rpc.id, mcpToolError('внутренняя ошибка выполнения тула'));
      }
    }

    return jsonRpcError(rpc.id, -32601, 'Method not found');
  }


export { isMcpHostAllowed };
