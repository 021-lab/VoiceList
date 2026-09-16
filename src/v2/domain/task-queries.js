import { clone } from './contracts.js';
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


export { formatTaskTitleTree, findTaskTree, createActiveTaskTree, createTaskSubgraph };
