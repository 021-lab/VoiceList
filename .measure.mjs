import { chromium } from '@playwright/test';
const base = process.argv[2];
const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
const p = await b.newPage();
await p.goto(base, { waitUntil: 'networkidle' });
await p.waitForTimeout(1500);
for (const title of ['Первая проба', 'Вторая проба']) {
  const log = []; const t0 = Date.now();
  const onReq = r => { if (r.url().includes('/api/')) log.push(`${Date.now() - t0}ms → ${r.method()} ${new URL(r.url()).pathname}`); };
  const onRes = r => { if (r.url().includes('/api/')) log.push(`${Date.now() - t0}ms ← ${r.status()} ${new URL(r.url()).pathname}`); };
  p.on('request', onReq); p.on('response', onRes);
  await p.click('#add-btn'); await p.fill('#input-title', title);
  const start = Date.now(); log.push('0ms click confirm'); 
  await p.click('#btn-confirm');
  await p.waitForFunction(t => [...document.querySelectorAll('.item-title')].some(e => e.textContent.includes(t)), title, { timeout: 15000 });
  log.push(`${Date.now() - t0}ms VISIBLE (${Date.now() - start}ms after confirm)`);
  await p.waitForTimeout(2500);
  p.off('request', onReq); p.off('response', onRes);
  console.log(`== ${title}\n` + log.join('\n'));
}
await b.close();
