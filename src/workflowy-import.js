const WORKFLOWY_HOSTS = new Set(['workflowy.com', 'www.workflowy.com', 'beta.workflowy.com']);
const IGNORED_COOKIE_FIELDS = new Set(['path', 'domain', 'expires', 'max-age', 'samesite', 'secure', 'httponly']);

function stripHtml(value) {
  return String(value || '')
    .replace(/<[^>]*>/g, '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

function parseWorkflowyUrl(rawUrl) {
  const url = new URL(String(rawUrl || '').trim());
  if (!WORKFLOWY_HOSTS.has(url.hostname)) throw new Error('Введите публичную ссылку Workflowy');
  if (!url.pathname.startsWith('/s/')) throw new Error('Ссылка Workflowy должна вести на shared tree');
  return url;
}

function extractShareId(html) {
  const match = String(html || '').match(/PROJECT_TREE_DATA_URL_PARAMS\s*=\s*({[^;]+})/);
  if (!match) throw new Error('Не удалось найти share_id на странице Workflowy');
  const data = JSON.parse(match[1]);
  if (!data.share_id) throw new Error('Workflowy не вернул share_id');
  return data.share_id;
}

function mergeCookies(currentCookieHeader, setCookieHeader) {
  if (!setCookieHeader) return currentCookieHeader;
  const cookies = new Map();
  for (const part of String(currentCookieHeader || '').split(';')) {
    const trimmed = part.trim();
    const index = trimmed.indexOf('=');
    if (index > 0) cookies.set(trimmed.slice(0, index), trimmed.slice(index + 1));
  }
  for (const match of String(setCookieHeader).matchAll(/(?:^|,\s*|;\s*)([^=;,\s]+)=([^;,]+)/g)) {
    const name = match[1].trim();
    const lower = name.toLowerCase();
    if (IGNORED_COOKIE_FIELDS.has(lower)) continue;
    cookies.set(name, match[2].trim());
  }
  return [...cookies.entries()].map(([name, value]) => `${name}=${value}`).join('; ');
}

async function readJsonResponse(response, label) {
  if (!response.ok) throw new Error(`${label}: Workflowy returned ${response.status}`);
  return response.json();
}

async function readTextResponse(response, label) {
  if (!response.ok) throw new Error(`${label}: Workflowy returned ${response.status}`);
  return response.text();
}

function workflowyJsonHeaders(cookieHeader, fetchImpl) {
  const canSetCookie = typeof window === 'undefined' || fetchImpl !== globalThis.fetch;
  return {
    Accept: 'application/json',
    'X-Requested-With': 'XMLHttpRequest',
    ...(cookieHeader && canSetCookie ? { Cookie: cookieHeader } : {})
  };
}

function renderPlainTextExport(rootProject, treeData) {
  const rootId = rootProject?.id;
  const title = stripHtml(rootProject?.nm);
  if (!rootId || !title) throw new Error('Workflowy tree root is empty');

  const childrenByParent = new Map();
  for (const item of treeData?.items || []) {
    if (!item?.id || !item.prnt) continue;
    if (!childrenByParent.has(item.prnt)) childrenByParent.set(item.prnt, []);
    childrenByParent.get(item.prnt).push(item);
  }
  for (const children of childrenByParent.values()) {
    children.sort((left, right) => (left.pr || 0) - (right.pr || 0));
  }

  const lines = [title, ''];
  function appendChildren(parentId, depth) {
    for (const item of childrenByParent.get(parentId) || []) {
      const itemTitle = stripHtml(item.nm);
      if (!itemTitle) continue;
      lines.push(`${'  '.repeat(depth)}- ${itemTitle}`);
      appendChildren(item.id, depth + 1);
    }
  }
  appendChildren(rootId, 0);
  return lines.join('\n');
}

export function parseWorkflowyPlainTextExport(text) {
  const lines = String(text || '').replace(/\r\n?/g, '\n').split('\n');
  const firstLineIndex = lines.findIndex((line) => line.trim());
  if (firstLineIndex < 0) throw new Error('Workflowy export is empty');
  const root = {
    title: lines[firstLineIndex].trim(),
    children: []
  };
  const stack = [root];
  let lastNode = root;

  for (const line of lines.slice(firstLineIndex + 1)) {
    if (!line.trim()) continue;
    const bullet = line.match(/^(\s*)-\s?(.*)$/);
    if (!bullet) {
      if (lastNode) lastNode.title = `${lastNode.title} ${line.trim()}`.trim();
      continue;
    }
    const title = bullet[2].trim();
    if (!title) continue;
    const level = Math.floor(bullet[1].length / 2) + 1;
    const parent = stack[level - 1] || root;
    const node = { title, children: [] };
    parent.children.push(node);
    stack[level] = node;
    stack.length = level + 1;
    lastNode = node;
  }

  return root;
}

export async function loadWorkflowyPlainTextExportFromUrl(rawUrl, { fetchImpl = fetch } = {}) {
  const shareUrl = parseWorkflowyUrl(rawUrl);
  let cookieHeader = '';

  const shareResponse = await fetchImpl(shareUrl.href, { redirect: 'follow' });
  cookieHeader = mergeCookies(cookieHeader, shareResponse.headers?.get?.('set-cookie'));
  const shareHtml = await readTextResponse(shareResponse, 'Workflowy share page');
  const shareId = extractShareId(shareHtml);

  const initUrl = new URL('/get_initialization_data', shareUrl.origin);
  initUrl.searchParams.set('share_id', shareId);
  initUrl.searchParams.set('client_version', '21');
  initUrl.searchParams.set('client_version_v2', '28');
  initUrl.searchParams.set('no_root_children', '1');
  initUrl.searchParams.set('include_main_tree', '1');
  const initData = await readJsonResponse(await fetchImpl(initUrl.href, {
    headers: workflowyJsonHeaders(cookieHeader, fetchImpl),
    credentials: 'include'
  }), 'Workflowy initialization');
  const rootProject = initData?.projectTreeData?.auxiliaryProjectTreeInfos?.[0]?.rootProject ||
    initData?.projectTreeData?.mainProjectTreeInfo?.rootProject;
  const operationId = initData?.projectTreeData?.initialMostRecentOperationTransactionId || '0';

  const treeUrl = new URL('/get_tree_data/', shareUrl.origin);
  treeUrl.searchParams.set('share_id', shareId);
  treeUrl.searchParams.set('ot', operationId);
  const treeData = await readJsonResponse(await fetchImpl(treeUrl.href, {
    headers: workflowyJsonHeaders(cookieHeader, fetchImpl),
    credentials: 'include'
  }), 'Workflowy tree data');

  return renderPlainTextExport(rootProject, treeData);
}

export async function importWorkflowyTreeFromUrl(rawUrl, { fetchImpl = fetch } = {}) {
  const text = await loadWorkflowyPlainTextExportFromUrl(rawUrl, { fetchImpl });
  return parseWorkflowyPlainTextExport(text);
}

export { extractShareId, mergeCookies, renderPlainTextExport };
