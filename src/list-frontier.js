const CLOSED_STATUSES = new Set(['done', 'archive']);
const BLOCKED_BRANCH_STATUSES = new Set(['archive']);
const FRONTIER_STATUSES = new Set(['open', 'focus']);

function normalizedStatus(task) {
  return String(task.status || '').toLowerCase();
}

function sortByOrder(left, right) {
  return (left.task.order || 0) - (right.task.order || 0);
}

function createNode(task) {
  return {
    id: task.id,
    parentId: task.parentId || null,
    task,
    children: [],
    hasFocusInSubtree: false,
    synthetic: false
  };
}

function buildTree(tasks) {
  const nodes = new Map();

  for (const task of tasks || []) {
    if (nodes.has(task.id)) throw new Error(`Duplicate task id: ${task.id}`);
    nodes.set(task.id, createNode(task));
  }

  const root = {
    id: '__root__',
    parentId: null,
    task: null,
    children: [],
    hasFocusInSubtree: false,
    synthetic: true
  };

  for (const node of nodes.values()) {
    if (node.parentId === null) {
      root.children.push(node);
      continue;
    }

    const parent = nodes.get(node.parentId);
    if (!parent) throw new Error(`Missing parent ${node.parentId} for task ${node.id}`);
    parent.children.push(node);
  }

  for (const node of nodes.values()) node.children.sort(sortByOrder);
  root.children.sort(sortByOrder);

  if (tasks.length > 0 && root.children.length === 0) {
    throw new Error('No reachable root tasks found');
  }

  return root;
}

function validateReachable(root, expectedTaskCount) {
  const visiting = new Set();
  const visited = new Set();

  function walk(node) {
    if (visiting.has(node.id)) throw new Error(`Cycle detected at task ${node.id}`);
    if (visited.has(node.id)) return;

    visiting.add(node.id);
    for (const child of node.children) walk(child);
    visiting.delete(node.id);
    visited.add(node.id);
  }

  walk(root);
  if (visited.size - 1 !== expectedTaskCount) {
    throw new Error('Some tasks are not reachable from a root');
  }
}

function markFocus(node) {
  let childHasFocus = false;

  for (const child of node.children) {
    if (markFocus(child)) childHasFocus = true;
  }

  node.hasFocusInSubtree = normalizedStatus(node.task || {}) === 'focus' || childHasFocus;
  return node.hasFocusInSubtree;
}

export function calculateFrontier(tasks = []) {
  const root = buildTree(tasks);
  validateReachable(root, tasks.length);
  markFocus(root);

  const frontier = [];
  const focusHighlights = [];

  function visit(node, paused) {
    const status = normalizedStatus(node.task || {});

    if (BLOCKED_BRANCH_STATUSES.has(status)) return false;
    if (CLOSED_STATUSES.has(status) && !node.hasFocusInSubtree) return false;
    if (status === 'focus') focusHighlights.push(node.task);

    const focusInsidePause = paused && status === 'focus';
    const insidePause = (paused && status !== 'focus') || status === 'pause';
    if (insidePause && !node.hasFocusInSubtree) return false;

    if (!node.synthetic && focusInsidePause) {
      frontier.push(node.task);
    }

    const activeChildren = node.children.filter((child) => (
      !CLOSED_STATUSES.has(normalizedStatus(child.task)) || child.hasFocusInSubtree
    ));
    const focusedChildren = activeChildren.filter((child) => normalizedStatus(child.task) === 'focus');
    const childrenToVisit = focusedChildren.length ? focusedChildren : activeChildren;

    let descendantInFrontier = false;
    for (const child of childrenToVisit) {
      if (visit(child, insidePause)) descendantInFrontier = true;
    }

    if (!node.synthetic && FRONTIER_STATUSES.has(status) && !descendantInFrontier && !focusInsidePause) {
      frontier.push(node.task);
      return true;
    }

    return focusInsidePause || descendantInFrontier;
  }

  visit(root, false);

  return { frontier, focusHighlights };
}
