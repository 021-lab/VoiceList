export function taskTreeFromItems(items = []) {
  const nodes = new Map();
  for (const item of items) {
    const id = String(item?.id || '');
    if (!id) continue;
    nodes.set(id, {
      id,
      title: String(item?.line1 || ''),
      status: String(item?.status || 'Open'),
      children: []
    });
  }

  const roots = [];
  for (const item of items) {
    const id = String(item?.id || '');
    const node = nodes.get(id);
    if (!node) continue;
    const parent = item?.parentId == null ? null : nodes.get(String(item.parentId));
    if (parent) parent.children.push(node);
    else roots.push(node);
  }
  return roots;
}
