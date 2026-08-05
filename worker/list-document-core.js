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
    return createStateEnvelope(state);
  }

  function ensureReady() {
    if (!state) throw new Error('Document core is not initialized');
  }

  function getSnapshot() {
    ensureReady();
    return createStateEnvelope(state);
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

  async function applyCommand(message, input, metadata = {}) {
    const beforeItems = clone(state.content.snapshot.items);
    let allocatedId = null;
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
      return createRejectedAck(message, 'UI-only command is handled by the HTML frontend');
    }
    if (!result.patch?.length && !result.logEntryDraft) {
      return createRejectedAck(message, 'Command produced no document change');
    }

    const nextContent = result.patch?.length ? applyJsonPatch(state.content, result.patch) : clone(state.content);
    const newTarget = allocatedId || findNewTarget(beforeItems, nextContent.snapshot.items);
    const rev = state.rev + 1;
    const logEntry = {
      id: encodeId(rev),
      rev,
      clientKey: message.clientKey,
      seq: message.seq,
      op: input.command,
      target: newTarget || input.actId || null,
      value: clone(input.payload || null),
      undo: null,
      undoes: input.command === 'undo' ? input.payload?.id || null : null,
      transcript: metadata.transcript ?? input.transcript ?? null,
      llm_raw: metadata.llmRaw ?? null,
      command: clone(input),
      patch: clone(result.patch || []),
      label: result.logEntryDraft?.label || input.command,
      comments: [],
      at: now().toISOString()
    };

    state.rev = rev;
    state.content = nextContent;
    state.log.push(logEntry);

    return {
      seq: message.seq,
      id: logEntry.id,
      status: 'applied',
      reason: null,
      newTarget: logEntry.target
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
      command: clone(input),
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
        else ack = await applyCommand({ ...message, clientKey, seq }, input);
      } else if (message.type === 'utterance') {
        const resolved = await resolveUtterance(message);
        ack = await applyCommand({ ...message, clientKey, seq }, resolved.input, resolved.metadata);
      } else {
        ack = createRejectedAck({ ...message, seq }, `Unsupported message type: ${message.type}`);
      }
    } catch (error) {
      ack = createRejectedAck({ ...message, seq }, error.message);
    }

    rememberAck(clientKey, seq, ack);
    return { ack: clone(ack), state: getSnapshot() };
  }

  function exportState() {
    ensureReady();
    return clone(state);
  }

  return {
    exportState,
    getSnapshot,
    handleClientMessage,
    init,
    listLog
  };
}
