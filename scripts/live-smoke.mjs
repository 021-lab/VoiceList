// End-to-end check of a deployed GPT-Live path: a real WebRTC offer from Chromium, a real
// session against the worker, and the events the sideband recorded for it.
// Run: LIVE_LOG_TOKEN=... node scripts/live-smoke.mjs
import { createServer } from 'node:http';
import { chromium } from '@playwright/test';

const BASE = process.env.LIVE_SMOKE_BASE || 'https://vlist-v02-dev.smileme.ai';
const TOKEN = process.env.LIVE_LOG_TOKEN || '';

// A local page, because getUserMedia needs a trustworthy origin. The browser only produces
// the offer and consumes the answer; every call to the deployed worker goes through Node,
// which already trusts the session's proxy CA.
const page_html = `<!doctype html><meta charset="utf-8"><title>live smoke</title>`;
const server = createServer((_request, response) => {
  response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
  response.end(page_html);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium',
  args: ['--no-proxy-server', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--autoplay-policy=no-user-gesture-required']
});
const context = await browser.newContext({ permissions: ['microphone'] });
const page = await context.newPage();
page.on('pageerror', error => console.log('pageerror:', error.message));
await page.goto(origin);

const offer = await page.evaluate(async () => {
  const peer = new RTCPeerConnection({});
  window.__peer = peer;
  window.__events = [];
  const media = await navigator.mediaDevices.getUserMedia({ audio: true });
  for (const track of media.getTracks()) peer.addTrack(track, media);
  const channel = peer.createDataChannel('oai-events');
  channel.addEventListener('message', event => window.__events.push(String(event.data).slice(0, 200)));
  channel.addEventListener('open', () => window.__events.push('__channel_open__'));
  const description = await peer.createOffer();
  await peer.setLocalDescription(description);
  await new Promise(resolve => {
    if (peer.iceGatheringState === 'complete') return resolve();
    peer.addEventListener('icegatheringstatechange', () => { if (peer.iceGatheringState === 'complete') resolve(); });
    setTimeout(resolve, 4000);
  });
  return peer.localDescription.sdp;
});
console.log('offer получен, длина:', offer.length, '| аудио-строк m=:', (offer.match(/^m=audio/gm) || []).length);

const created = await fetch(`${BASE}/api/live/session`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sdp: offer })
});
const payload = await created.json();
console.log('создание сессии:', created.status, JSON.stringify(payload).slice(0, 400));

if (created.ok) {
  const state = await page.evaluate(async (sdp) => {
    await window.__peer.setRemoteDescription({ type: 'answer', sdp });
    await new Promise(resolve => setTimeout(resolve, 9000));
    return {
      connection: window.__peer.connectionState,
      ice: window.__peer.iceConnectionState,
      events: window.__events.length,
      sample: window.__events.slice(0, 4)
    };
  }, payload.sdp);
  console.log('состояние WebRTC:', JSON.stringify(state, null, 2));

  const status = await (await fetch(`${BASE}/api/live/session`)).json();
  console.log('сессия на сервере:', JSON.stringify(status));

  if (TOKEN) {
    const log = await (await fetch(`${BASE}/api/live/log?limit=60`, { headers: { 'X-VoiceList-Log-Token': TOKEN } })).json();
    const mine = log.entries.filter(entry => entry.liveSessionId === payload.sessionId);
    console.log(`лог этой сессии: ${mine.length} записей из ${log.stats.events}`);
    for (const entry of mine.slice(0, 30)) {
      const text = entry.payload?.delta || entry.payload?.reason || entry.payload?.error?.message || '';
      console.log(`  seq=${entry.seq} ${entry.direction} ${entry.type} ${String(text).slice(0, 80)}`);
    }
  }
  const stopped = await (await fetch(`${BASE}/api/live/session/stop`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })).json();
  console.log('остановка:', JSON.stringify(stopped));
}

await browser.close();
server.close();
