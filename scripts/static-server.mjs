import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import url from 'node:url';

const rootDir = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), '..');
const port = Number(process.env.PORT || 4511);
const host = process.env.HOST || (process.env.PORT ? '0.0.0.0' : '127.0.0.1');

const contentTypes = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8'
};

const server = http.createServer((req, res) => {
  const requestUrl = new URL(req.url, `http://${req.headers.host}`);

  if (requestUrl.pathname === '/health') {
    res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('ok');
    return;
  }

  const pathname = requestUrl.pathname === '/' ? '/list-manager.html' : requestUrl.pathname;
  const filePath = path.join(rootDir, pathname);

  if (!filePath.startsWith(rootDir) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Not found');
    return;
  }

  const ext = path.extname(filePath);
  const contentType = contentTypes[ext] || 'application/octet-stream';
  // The bundle is inlined into the HTML, so revalidating avoids serving a stale build.
  res.writeHead(200, {
    'Content-Type': contentType,
    'Cache-Control': ext === '.html' ? 'no-cache' : 'public, max-age=300'
  });
  fs.createReadStream(filePath).pipe(res);
});

server.listen(port, host, () => {
  process.stdout.write(`VoiceList static server listening on http://${host}:${port}\n`);
});
