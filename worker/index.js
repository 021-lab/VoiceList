import { LIST_MANAGER_HTML } from './generated-list-manager.js';

const DEFAULT_TAOSMD_ORIGIN = 'https://recorder.smileme.ai/taosmd';

function htmlResponse() {
  return new Response(LIST_MANAGER_HTML, {
    headers: {
      'Content-Type': 'text/html; charset=utf-8',
      'Cache-Control': 'no-store'
    }
  });
}

function joinPath(left, right) {
  return `${left.replace(/\/$/, '')}/${right.replace(/^\//, '')}`;
}

async function proxyApi(request, env) {
  const sourceUrl = new URL(request.url);
  const upstreamBase = new URL(env.TAOSMD_ORIGIN || DEFAULT_TAOSMD_ORIGIN);
  const upstreamPath = sourceUrl.pathname.replace(/^\/api\/?/, '/');
  const upstreamUrl = new URL(upstreamBase);
  upstreamUrl.pathname = joinPath(upstreamBase.pathname, upstreamPath);
  upstreamUrl.search = sourceUrl.search;

  const headers = new Headers(request.headers);
  headers.delete('Host');
  if (env.TAOSMD_TOKEN) headers.set('Authorization', `Bearer ${env.TAOSMD_TOKEN}`);

  return fetch(upstreamUrl, {
    method: request.method,
    headers,
    body: request.method === 'GET' || request.method === 'HEAD' ? undefined : request.body,
    redirect: 'manual'
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === '/' || url.pathname === '/list-manager.html') return htmlResponse();
    if (url.pathname === '/health') return Response.json({ status: 'ok' });
    if (url.pathname.startsWith('/api/')) return proxyApi(request, env);

    return new Response('Not found', { status: 404 });
  }
};
