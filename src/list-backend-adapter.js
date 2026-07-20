const LOCAL_TO_BACKEND_STATUS = Object.freeze({
  Open: 'open',
  Done: 'closed',
  Focus: 'focus',
  Pause: 'pause',
  Archive: 'superseded'
});

const BACKEND_TO_LOCAL_STATUS = Object.freeze({
  open: 'Open',
  closed: 'Done',
  focus: 'Focus',
  pause: 'Pause',
  superseded: 'Archive',
  in_progress: 'Open',
  blocked: 'Open'
});

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function withQuery(path, params) {
  const query = new URLSearchParams(params);
  return `${path}?${query.toString()}`;
}

function jsonHeaders() {
  return { 'Content-Type': 'application/json' };
}

function sortByOrder(items) {
  return [...items].sort((left, right) => {
    const leftParent = left.parentId || '';
    const rightParent = right.parentId || '';
    if (leftParent !== rightParent) return leftParent.localeCompare(rightParent);
    return (left.order || 0) - (right.order || 0);
  });
}

function buildActionLogBody(actionLogEntry) {
  const command = actionLogEntry?.command || {};
  return {
    schema: 'voicelist.action.v1',
    id: actionLogEntry?.id,
    ts: actionLogEntry?.createdAt,
    op: command.command,
    payload: {
      taskId: command.actId,
      source: command.source,
      ...(command.payload || {})
    }
  };
}

export function localStatusToBackend(status) {
  return LOCAL_TO_BACKEND_STATUS[status] || 'open';
}

export function backendStatusToLocal(status) {
  return BACKEND_TO_LOCAL_STATUS[status] || 'Open';
}

export function createBackendAdapter({
  apiBase = '/api',
  fetchImpl = fetch,
  createdBy = 'user',
  actionThread = 'voicelist-log',
  project = 'voicelist'
} = {}) {
  const base = apiBase.replace(/\/$/, '');

  async function request(path, options = {}) {
    const response = await fetchImpl(`${base}${path}`, options);
    if (!response.ok) throw new Error(`taosmd request failed: ${response.status} ${path}`);
    return response.json();
  }

  async function post(path, body) {
    return request(path, {
      method: 'POST',
      headers: jsonHeaders(),
      body: JSON.stringify(body)
    });
  }

  async function loadBackend() {
    const [tasksBody, edgesBody] = await Promise.all([
      request(withQuery('/tasks', { limit: 500, project })),
      request(withQuery('/tasks/edges', { limit: 2000 }))
    ]);
    return {
      tasks: tasksBody.tasks || [],
      edges: edgesBody.edges || []
    };
  }

  function stateFromBackend(tasks, edges) {
    const parentByChild = new Map();
    for (const edge of edges) {
      if (edge.type === 'parent' && !edge.removed_ts) parentByChild.set(edge.from_id, edge.to_id);
    }

    const items = tasks.map((task, index) => ({
      id: task.id,
      parentId: parentByChild.get(task.id) || null,
      order: Number.isFinite(task.priority) ? task.priority : (index + 1) * 10,
      status: backendStatusToLocal(task.status),
      line1: task.title || '',
      collapsed: false,
      tags: []
    }));

    return {
      snapshot: { items: sortByOrder(items) },
      actionLog: []
    };
  }

  async function ensureTask(item, backendById) {
    if (backendById.has(item.id)) {
      const backend = backendById.get(item.id);
      const payload = {
        title: item.line1,
        status: localStatusToBackend(item.status),
        priority: item.order || 0,
        created_by: createdBy
      };
      if ((backend.title || '') !== item.line1 || backend.status !== payload.status || backend.priority !== payload.priority) {
        await post(`/tasks/${encodeURIComponent(item.id)}`, payload);
      }
      return item.id;
    }

    const created = await post('/tasks', {
      title: item.line1,
      status: localStatusToBackend(item.status),
      priority: item.order || 0,
      project,
      created_by: createdBy
    });
    backendById.set(created.id, created);
    return created.id;
  }

  async function ensureChildEdges(childId, parentId, edgeKeys) {
    for (const type of ['parent', 'blocks']) {
      const key = `${childId}->${parentId}:${type}`;
      if (edgeKeys.has(key)) continue;
      await post(`/tasks/${encodeURIComponent(childId)}/edges`, {
        to_id: parentId,
        type,
        created_by: createdBy
      });
      edgeKeys.add(key);
    }
  }

  async function sendActionLog(actionLogEntry) {
    if (!actionLogEntry) return;
    await post('/a2a/send', {
      from: createdBy,
      thread: actionThread,
      body: JSON.stringify(buildActionLogBody(actionLogEntry))
    });
  }

  return {
    async load() {
      try {
        const { tasks, edges } = await loadBackend();
        return stateFromBackend(tasks, edges);
      } catch (error) {
        console.warn('taosmd bootstrap skipped', error);
        return null;
      }
    },

    async save(state, { actionLogEntry = null } = {}) {
      const nextState = clone(state);
      const { tasks, edges } = await loadBackend();
      const backendById = new Map(tasks.map((task) => [task.id, task]));
      const edgeKeys = new Set(
        edges
          .filter((edge) => !edge.removed_ts)
          .map((edge) => `${edge.from_id}->${edge.to_id}:${edge.type}`)
      );
      const idMap = new Map();

      for (const item of sortByOrder(nextState.snapshot.items)) {
        const backendId = await ensureTask(item, backendById);
        idMap.set(item.id, backendId);
        item.id = backendId;
      }

      for (const item of nextState.snapshot.items) {
        if (!item.parentId) continue;
        item.parentId = idMap.get(item.parentId) || item.parentId;
        await ensureChildEdges(item.id, item.parentId, edgeKeys);
      }

      const fullLogEntry = actionLogEntry?.command
        ? actionLogEntry
        : nextState.actionLog.find((entry) => entry.id === actionLogEntry?.id);
      await sendActionLog(fullLogEntry);
      return nextState;
    }
  };
}
