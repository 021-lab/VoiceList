import { parseCommand, toCommand } from '../src/command-resolver.js';
import { createInterpreter } from '../src/list-interpreter.js';
import { importWorkflowyTreeFromUrl } from '../src/workflowy-import.js';

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function encodeId(value) {
  return Number(value).toString(36);
}

const INBOX_ITEM = {
  id: 'inbox',
  parentId: null,
  order: 0,
  status: 'Open',
  line1: 'Входящие',
  line2: '',
  collapsed: false,
  tags: []
};

function commandName(entry) {
  return entry?.op || entry?.command?.command || '';
}

function compactLogCommand(command) {
  const nextCommand = clone(command || {});
  if (nextCommand.command === 'undo' && nextCommand.payload?.snapshot) {
    nextCommand.payload = {
      ...nextCommand.payload,
      snapshot: '[omitted]'
    };
  }
  return nextCommand;
}

function compactLogEntry(entry) {
  return {
    ...clone(entry),
    command: compactLogCommand(entry.command),
    patch: []
  };
}

function compactStateLog(state) {
  state.log = (state.log || [])
    .filter((entry) => commandName(entry) !== 'toggleCollapse')
    .map(compactLogEntry);
}

function normalizeSeed(seedState) {
  const items = clone(seedState?.snapshot?.items || []);
  const normalizedItems = items.some((item) => item.id === INBOX_ITEM.id) ? items : [clone(INBOX_ITEM), ...items];
  return {
    snapshot: {
      items: normalizedItems
    },
    actionLog: clone(seedState?.actionLog || [])
  };
}

function ensureInboxContent(content) {
  const nextContent = {
    snapshot: {
      items: clone(content?.snapshot?.items || [])
    },
    actionLog: clone(content?.actionLog || [])
  };
  if (!nextContent.snapshot.items.some((item) => item.id === INBOX_ITEM.id)) {
    nextContent.snapshot.items = [clone(INBOX_ITEM), ...nextContent.snapshot.items];
  }
  return nextContent;
}

function decodePathSegment(segment) {
  return segment.replace(/~1/g, '/').replace(/~0/g, '~');
}

function applyJsonPatch(document, patch) {
  const nextDocument = clone(document);

  for (const operation of patch) {
    const segments = operation.path.split('/').slice(1).map(decodePathSegment);
    const lastSegment = segments.pop();
    let target = nextDocument;

    for (const segment of segments) {
      target = target[Array.isArray(target) ? Number(segment) : segment];
    }

    if (operation.op === 'replace' || operation.op === 'add') {
      target[Array.isArray(target) ? Number(lastSegment) : lastSegment] = clone(operation.value);
    } else if (operation.op === 'remove') {
      if (Array.isArray(target)) target.splice(Number(lastSegment), 1);
      else delete target[lastSegment];
    } else {
      throw new Error(`Unsupported patch op: ${operation.op}`);
    }
  }

  return nextDocument;
}

function findNewTarget(beforeItems, afterItems) {
  const beforeIds = new Set(beforeItems.map((item) => item.id));
  return afterItems.find((item) => !beforeIds.has(item.id))?.id || null;
}

function itemSort(left, right) {
  return (left.order || 0) - (right.order || 0) ||
    String(left.line1 || '').localeCompare(String(right.line1 || '')) ||
    String(left.id || '').localeCompare(String(right.id || ''));
}

function createItemIndexes(items) {
  const byId = new Map();
  const childrenByParent = new Map();

  for (const item of items) {
    byId.set(item.id, item);
    const parentKey = item.parentId ?? null;
    const children = childrenByParent.get(parentKey) || [];
    children.push(item);
    childrenByParent.set(parentKey, children);
  }

  for (const children of childrenByParent.values()) children.sort(itemSort);
  return { byId, childrenByParent };
}

function oneLine(value) {
  return String(value ?? '').replace(/\s+/g, ' ').trim();
}

function formatTaskTitleTree(items) {
  const { byId, childrenByParent } = createItemIndexes(items);
  const lines = [];
  const visited = new Set();

  function appendChildren(parentId, depth) {
    for (const item of childrenByParent.get(parentId) || []) {
      if (visited.has(item.id)) continue;
      visited.add(item.id);
      const parent = item.parentId ? byId.get(item.parentId) : null;
      const parentLabel = parent ? oneLine(parent.line1) : 'root';
      lines.push(`${item.id} >> ${'  '.repeat(depth)}${oneLine(item.line1)} >> ${parentLabel}`);
      appendChildren(item.id, depth + 1);
    }
  }

  appendChildren(null, 0);

  for (const item of [...items].sort(itemSort)) {
    if (!visited.has(item.id)) {
      visited.add(item.id);
      const parentLabel = item.parentId ? 'missing parent' : 'root';
      lines.push(`${item.id} >> ${oneLine(item.line1)} >> ${parentLabel}`);
      appendChildren(item.id, 1);
    }
  }

  return `${lines.join('\n')}\n`;
}

function createTaskNode(item, childrenByParent, visited = new Set()) {
  if (visited.has(item.id)) {
    return {
      ...clone(item),
      children: [],
      cycleDetected: true
    };
  }

  const nextVisited = new Set(visited);
  nextVisited.add(item.id);
  return {
    ...clone(item),
    children: (childrenByParent.get(item.id) || []).map((child) => createTaskNode(child, childrenByParent, nextVisited))
  };
}

function findTaskTree(items, query = {}) {
  const { byId, childrenByParent } = createItemIndexes(items);
  const id = String(query.id || '').trim();

  if (!id) return { status: 'missing-query' };
  const item = byId.get(id);
  return item ? { status: 'found', task: createTaskNode(item, childrenByParent) } : { status: 'not-found' };
}

function toTaskSummary(item) {
  return {
    id: item.id,
    title: item.line1,
    status: item.status
  };
}

function createActiveTaskTree(items) {
  const { byId, childrenByParent } = createItemIndexes(items);
  const visited = new Set();

  function isPrunedByArchivedAncestor(item) {
    let parentId = item.parentId ?? null;
    const seen = new Set([item.id]);
    while (parentId) {
      if (seen.has(parentId)) return false;
      seen.add(parentId);
      const parent = byId.get(parentId);
      if (!parent) return false;
      if (parent.status === 'Archive') return true;
      parentId = parent.parentId ?? null;
    }
    return false;
  }

  function visit(item) {
    if (visited.has(item.id) || item.status === 'Archive') return null;
    visited.add(item.id);
    return {
      ...toTaskSummary(item),
      children: (childrenByParent.get(item.id) || [])
        .map(visit)
        .filter(Boolean)
    };
  }

  const roots = [];
  for (const item of childrenByParent.get(null) || []) {
    const node = visit(item);
    if (node) roots.push(node);
  }

  for (const item of [...items].sort(itemSort)) {
    if (!visited.has(item.id) && item.status !== 'Archive' && !isPrunedByArchivedAncestor(item)) {
      const node = visit(item);
      if (node) roots.push(node);
    }
  }

  return roots;
}

function createTaskSubgraph(items, id) {
  const { byId, childrenByParent } = createItemIndexes(items);
  const task = byId.get(String(id || '').trim());
  if (!task) return { status: 'not-found' };

  const path = [];
  const seen = new Set([task.id]);
  let parentId = task.parentId ?? null;
  while (parentId) {
    if (seen.has(parentId)) break;
    seen.add(parentId);
    const parent = byId.get(parentId);
    if (!parent) break;
    path.unshift(toTaskSummary(parent));
    parentId = parent.parentId ?? null;
  }

  return {
    status: 'found',
    subgraph: {
      path,
      task: clone(task),
      children: (childrenByParent.get(task.id) || [])
        .filter((child) => child.status !== 'Archive')
        .map(toTaskSummary)
    }
  };
}

function createActionLogView(log) {
  return log.map((entry) => ({
    id: entry.id,
    createdAt: entry.at,
    transcript: entry.transcript ?? null,
    command: entry.command,
    patch: entry.patch,
    label: entry.label,
    syncStatus: 'synced',
    comments: entry.comments || []
  }));
}

function createStateEnvelope(state) {
  return {
    rev: state.rev,
    content: {
      snapshot: clone(state.content.snapshot),
      actionLog: createActionLogView(state.log)
    }
  };
}

function getClientResults(state, clientKey) {
  if (!state.clients[clientKey]) state.clients[clientKey] = {};
  return state.clients[clientKey];
}

function buildPrompt({ content, target, transcript }) {
  return [
    'You convert a Russian voice command for a nested list into JSON.',
    'Return only one JSON object with command, actId, actType, payload.',
    'Supported commands: addChild, editItem, setParent, setStatus, deleteItem, setTags, undo.',
    `Target id: ${target || 'list'}`,
    `Transcript: ${transcript}`,
    `Document items: ${JSON.stringify(content.snapshot.items.slice(0, 80))}`
  ].join('\n');
}

async function resolveWithOpenRouter({ content, target, transcript, openRouterApiKey, openRouterModel, fetchImpl }) {
  if (!openRouterApiKey) throw new Error('OPENROUTER_API_KEY is not configured');

  const response = await fetchImpl('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${openRouterApiKey}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'https://vlist-cloudflare-backend.smileme.ai',
      'X-Title': 'VoiceList Cloudflare Backend'
    },
    body: JSON.stringify({
      model: openRouterModel,
      messages: [
        {
          role: 'system',
          content: 'Return strict JSON only. No markdown.'
        },
        {
          role: 'user',
          content: buildPrompt({ content, target, transcript })
        }
      ],
      response_format: { type: 'json_object' }
    })
  });

  const raw = await response.text();
  if (!response.ok) throw new Error(`OpenRouter request failed: ${response.status} ${raw.slice(0, 240)}`);

  const data = JSON.parse(raw);
  const contentText = data?.choices?.[0]?.message?.content;
  if (!contentText) throw new Error('OpenRouter returned empty content');
  return {
    command: JSON.parse(contentText),
    llmRaw: raw
  };
}

export function createDocumentCore({
  seedState,
  initialState = null,
  openRouterApiKey = '',
  openRouterModel = 'openai/gpt-4.1-mini',
  fetchImpl = fetch,
  now = () => new Date()
} = {}) {
  let state = null;

  async function init() {
    state = initialState ? clone(initialState) : {
      content: normalizeSeed(seedState),
      log: [],
      rev: 0,
      nextId: 1000,
      clients: {}
    };
    state.content = ensureInboxContent(state.content);
    state.log ||= [];
    state.rev ||= 0;
    state.nextId ||= 1000;
    state.clients ||= {};
    compactStateLog(state);
    return createStateEnvelope(state);
  }

  function ensureReady() {
    if (!state) throw new Error('Document core is not initialized');
  }

  function getSnapshot() {
    ensureReady();
    return createStateEnvelope(state);
  }

  function getTaskTitleTreeText() {
    ensureReady();
    return formatTaskTitleTree(state.content.snapshot.items);
  }

  function getTaskTree(query) {
    ensureReady();
    return findTaskTree(state.content.snapshot.items, query);
  }

  function getTaskById(id) {
    ensureReady();
    const taskId = String(id || '').trim();
    return clone(state.content.snapshot.items.find((item) => item.id === taskId) || null);
  }

  function getActiveTaskTree() {
    ensureReady();
    return createActiveTaskTree(state.content.snapshot.items);
  }

  function getTaskSubgraph(id) {
    ensureReady();
    return createTaskSubgraph(state.content.snapshot.items, id);
  }

  function listLog() {
    ensureReady();
    return clone(state.log);
  }

  function rememberAck(clientKey, seq, ack) {
    getClientResults(state, clientKey)[seq] = clone(ack);
    return ack;
  }

  function createRejectedAck(message, reason) {
    return {
      seq: message.seq,
      id: null,
      status: 'rejected',
      reason,
      newTarget: null
    };
  }

  function findLastUndoableLogEntry() {
    const undone = new Set(state.log
      .filter((entry) => entry.op === 'undo' && entry.undoes)
      .map((entry) => entry.undoes));
    for (let index = state.log.length - 1; index >= 0; index -= 1) {
      const entry = state.log[index];
      if (entry.op !== 'undo' && entry.undo?.snapshot && !undone.has(entry.id)) return entry;
    }
    return null;
  }

  async function applyCommand(input, { message = {}, metadata = {} } = {}) {
    ensureReady();
    const clientKey = String(message.clientKey || 'server');
    const seq = Number.isFinite(Number(message.seq)) ? Number(message.seq) : state.rev + 1;
    const commandMessage = { ...message, clientKey, seq };
    const beforeItems = clone(state.content.snapshot.items);
    const beforeSnapshot = clone(state.content.snapshot);
    let allocatedId = null;
    if (input.command === 'undo' && !input.payload?.snapshot) {
      const undoEntry = findLastUndoableLogEntry();
      if (!undoEntry) throw new Error('нечего откатывать');
      input = {
        ...input,
        actId: undoEntry.target || input.actId || 'list',
        actType: undoEntry.target ? 'task' : 'list',
        payload: {
          ...(input.payload || {}),
          id: undoEntry.id,
          snapshot: undoEntry.undo.snapshot
        }
      };
    }
    if (input.command === 'importWorkflowy') {
      const url = String(input.payload?.url || '').trim();
      const tree = await importWorkflowyTreeFromUrl(url, { fetchImpl });
      input = {
        ...input,
        command: 'importWorkflowyTree',
        payload: {
          sourceUrl: url,
          tree
        }
      };
    }
    const interpreter = createInterpreter({
      createItemId(existingIds) {
        let nextId;
        do {
          nextId = encodeId(state.nextId);
          state.nextId += 1;
        } while (existingIds.has(nextId));
        if (!allocatedId) allocatedId = nextId;
        return nextId;
      },
      createLogId() {
        return encodeId(state.rev + 1);
      },
      now
    });
    const result = interpreter.execute(state.content, input);

    if (result.viewMode || result.effect) {
      return createRejectedAck(commandMessage, 'UI-only command is handled by the HTML frontend');
    }
    if (!result.patch?.length && !result.logEntryDraft) {
      return createRejectedAck(commandMessage, 'Command produced no document change');
    }

    const nextContent = result.patch?.length ? applyJsonPatch(state.content, result.patch) : clone(state.content);
    const newTarget = allocatedId || findNewTarget(beforeItems, nextContent.snapshot.items);
    const rev = state.rev + 1;
    let logEntry = null;

    state.rev = rev;
    state.content = nextContent;
    if (result.logEntryDraft) {
      logEntry = {
        id: encodeId(rev),
        rev,
        clientKey: commandMessage.clientKey,
        seq: commandMessage.seq,
        op: input.command,
        target: newTarget || input.actId || null,
        value: clone(input.payload || null),
        undo: input.command === 'undo' ? null : { snapshot: beforeSnapshot },
        undoes: input.command === 'undo' ? input.payload?.id || null : null,
        transcript: metadata.transcript ?? input.transcript ?? null,
        llm_raw: metadata.llmRaw ?? null,
        command: compactLogCommand(input),
        patch: [],
        label: result.logEntryDraft.label,
        comments: [],
        at: now().toISOString()
      };
      state.log.push(logEntry);
    }

    return {
      seq: commandMessage.seq,
      id: logEntry?.id || null,
      status: 'applied',
      reason: null,
      newTarget: logEntry?.target || newTarget || input.actId || null
    };
  }

  async function undoLastAction({ clientKey = 'server', seq = null, source = 'server' } = {}) {
    ensureReady();
    const undoEntry = findLastUndoableLogEntry();
    if (!undoEntry) {
      return {
        status: 'error',
        error: 'нечего откатывать'
      };
    }
    const ack = await applyCommand({
      actId: undoEntry.target || 'list',
      actType: undoEntry.target ? 'task' : 'list',
      command: 'undo',
      payload: { id: undoEntry.id, snapshot: undoEntry.undo.snapshot },
      source
    }, {
      message: {
        clientKey,
        seq: seq ?? state.rev + 1
      }
    });
    return {
      status: ack.status,
      ack,
      undone: {
        logId: undoEntry.id,
        command: undoEntry.op,
        id: undoEntry.target || null
      },
      node: undoEntry.target ? getTaskById(undoEntry.target) : null
    };
  }

  function applyLogComment(message, input) {
    const text = String(input.payload?.text || '').trim();
    if (!text) return createRejectedAck(message, 'Empty log comment');
    const target = state.log.find((entry) => entry.id === input.actId);
    if (!target) return createRejectedAck(message, 'Log entry was not found');

    const rev = state.rev + 1;
    const comment = {
      id: `c-${encodeId(rev)}`,
      createdAt: now().toISOString(),
      text
    };
    target.comments = [...(target.comments || []), comment];
    const logEntry = {
      id: encodeId(rev),
      rev,
      clientKey: message.clientKey,
      seq: message.seq,
      op: 'commentLogEntry',
      target: target.id,
      value: { text },
      undo: null,
      undoes: null,
      transcript: input.transcript ?? null,
      llm_raw: null,
      command: compactLogCommand(input),
      patch: [],
      label: 'Добавлен комментарий к записи журнала',
      comments: [],
      at: now().toISOString()
    };
    state.rev = rev;
    state.log.push(logEntry);
    return {
      seq: message.seq,
      id: logEntry.id,
      status: 'applied',
      reason: null,
      newTarget: target.id
    };
  }

  async function resolveUtterance(message) {
    const transcript = String(message.transcript || '').trim();
    const context = message.target || null;
    const parsed = parseCommand(transcript, context);
    if (parsed.kind === 'one') {
      return {
        input: {
          ...toCommand(parsed.hypothesis, context),
          transcript,
          source: 'voice'
        },
        metadata: { transcript }
      };
    }

    const resolved = await resolveWithOpenRouter({
      content: state.content,
      target: context,
      transcript,
      openRouterApiKey,
      openRouterModel,
      fetchImpl
    });
    return {
      input: {
        ...resolved.command,
        transcript,
        source: 'voice-llm'
      },
      metadata: {
        transcript,
        llmRaw: resolved.llmRaw
      }
    };
  }

  async function logFallbackUtterance(message, reason = null) {
    const transcript = String(message.transcript || '').trim();
    if (!transcript) return createRejectedAck(message, reason || 'Empty fallback utterance');
    return applyCommand({
      actId: message.target || 'list',
      actType: message.target ? 'task' : 'list',
      command: 'logFallbackUtterance',
      payload: {
        text: transcript,
        ...(reason ? { reason } : {})
      },
      source: 'voice-fallback',
      transcript
    }, {
      message
    });
  }

  async function handleClientMessage(message) {
    ensureReady();
    const clientKey = String(message.clientKey || 'anonymous');
    const seq = Number(message.seq);
    const clientResults = getClientResults(state, clientKey);
    if (clientResults[seq]) return { ack: clone(clientResults[seq]), state: getSnapshot() };

    let ack;
    try {
      if (message.type === 'command') {
        const input = message.input || {};
        if (input.command === 'commentLogEntry') ack = applyLogComment({ ...message, clientKey, seq }, input);
        else ack = await applyCommand(input, { message: { ...message, clientKey, seq } });
      } else if (message.type === 'utterance') {
        const resolved = await resolveUtterance(message);
        ack = await applyCommand(resolved.input, {
          message: { ...message, clientKey, seq },
          metadata: resolved.metadata
        });
      } else {
        ack = createRejectedAck({ ...message, seq }, `Unsupported message type: ${message.type}`);
      }
    } catch (error) {
      if (message.type === 'utterance') {
        try {
          ack = await logFallbackUtterance({ ...message, clientKey, seq }, error.message);
        } catch (fallbackError) {
          ack = createRejectedAck({ ...message, seq }, fallbackError.message);
        }
      } else {
        ack = createRejectedAck({ ...message, seq }, error.message);
      }
    }

    rememberAck(clientKey, seq, ack);
    return { ack: clone(ack), state: getSnapshot() };
  }

  function exportState() {
    ensureReady();
    const exported = clone(state);
    compactStateLog(exported);
    return exported;
  }

  return {
    exportState,
    applyCommand,
    getActiveTaskTree,
    getTaskTitleTreeText,
    getTaskById,
    getTaskSubgraph,
    getTaskTree,
    getSnapshot,
    handleClientMessage,
    init,
    listLog,
    undoLastAction
  };
}
