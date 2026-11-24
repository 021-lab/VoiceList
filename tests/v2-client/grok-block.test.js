import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { Client } from '../../src/v2/client/client.js';

const node = (type, id, props = {}, children = []) => ({ type, id, props, children });
const card = (grok) => ({ schemaVersion: 1, revision: 1, cursor: 0, view: 'edit', grok: 'g1',
  root: node('application', 'app', {}, [node('toolbar', 'toolbar'), node('task-editor', 'task:t1', { taskId: 't1', title: 'Купить', status: 'Open', mode: 'edit', subtasks: [], grok })]) });
const frontier = (summary) => ({ schemaVersion: 1, revision: 1, cursor: 0, view: 'frontier', grok: 'g1',
  root: node('application', 'app', {}, [node('toolbar', 'toolbar'), node('task-list', 'screen:frontier', {}, [node('task', 'task:t1', { taskId: 't1', title: 'Купить', status: 'Open', grok: summary })])]) });

let client;
beforeEach(() => {
  document.body.innerHTML = readFileSync('list-manager.template.html', 'utf8').match(/<body>([\s\S]*)<script>/)[1];
  sessionStorage.clear();
});
afterEach(() => { client?.disconnect(); vi.restoreAllMocks(); });
const create = (fetch = vi.fn()) => { client = new Client({ document, storage: sessionStorage, fetch }); client.mount(); return client; };

describe('the Grok block on the task card', () => {
  it('without the secrets says so and does not send', () => {
    create().render(card({ configured: false, messages: [] }));
    expect(document.getElementById('grok-status').textContent).toBe('Грок не подключён');
    expect(document.getElementById('grok-send').disabled).toBe(true);
  });

  it('shows Grok text as text, keeps line breaks, makes links clickable', () => {
    create().render(card({ configured: true, messages: [
      { id: 'm1', role: 'user', text: 'Найди', at: '2026-11-24T10:00:00Z', state: 'accepted' },
      { id: 'r1', role: 'grok', text: '<b>жирно</b>\nсм. https://example.com/x.', at: '2026-11-24T10:01:00Z', replyTo: 'm1' }
    ] }));
    const reply = document.querySelector('[data-role=grok] .v02-grok-text');
    expect(reply.querySelector('b')).toBeNull();
    expect(reply.textContent).toBe('<b>жирно</b>\nсм. https://example.com/x.');
    expect(reply.querySelector('a').getAttribute('href')).toBe('https://example.com/x');
    expect(document.querySelector('.v02-grok-working')).toBeNull();
  });

  it('«Грок работает…» under an accepted message with no answer; «Повторить» on a failed one resends the same id', async () => {
    const fetch = vi.fn(async (path) => ({ ok: true, status: 200, json: async () => (path.startsWith('/api/grok') ? { message: {} } : card({ configured: true, messages: [] })) }));
    create(fetch).render(card({ configured: true, messages: [
      { id: 'm1', role: 'user', text: 'Первое', at: '1', state: 'failed' },
      { id: 's1', role: 'system', text: 'Вебхук Грока больше не существует', at: '2', replyTo: 'm1' },
      { id: 'm2', role: 'user', text: 'Второе', at: '3', state: 'accepted' }
    ] }));
    expect(document.querySelectorAll('.v02-grok-working')).toHaveLength(1);
    expect(document.querySelector('[data-role=system]').textContent).toBe('Вебхук Грока больше не существует');
    document.querySelector('.v02-grok-retry').click();
    await vi.waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/grok/delegate', expect.anything()));
    const sent = JSON.parse(fetch.mock.calls.find(([path]) => path === '/api/grok/delegate')[1].body);
    expect(sent).toEqual({ taskId: 't1', messageId: 'm1', text: 'Первое' });
  });

  it('a refused send leaves the button usable', async () => {
    const fetch = vi.fn(async (path) => (path.startsWith('/api/grok') ? { ok: false, status: 400, json: async () => ({ error: 'Сообщение длиннее 8000 символов' }) } : { ok: true, status: 200, json: async () => card({ configured: true, messages: [] }) }));
    create(fetch).render(card({ configured: true, messages: [] }));
    document.getElementById('grok-input').value = 'x';
    await client.delegateToGrok();
    expect(document.getElementById('grok-send').disabled).toBe(false);
    expect(document.getElementById('grok-input').value).toBe('x');
  });

  it('«Открыть в Гроке» is the deep link, and inert without Grok', () => {
    create().render(card({ configured: true, messages: [], link: 'grokbot://app/v1/prompt?text=x' }));
    const open = document.getElementById('grok-open');
    expect(open.textContent).toBe('Открыть в Гроке');
    expect(open.getAttribute('href')).toBe('grokbot://app/v1/prompt?text=x');
    client.render(card({ configured: false, messages: [], link: null }));
    expect(open.hasAttribute('href')).toBe(false);
    expect(open.getAttribute('aria-disabled')).toBe('true');
  });

  it('lists the task\'s pages under the thread as links that open in a new tab', () => {
    create().render(card({ configured: true, messages: [], pages: [{ id: 'p1', title: 'Сравнение', url: 'https://vlist-pages.smileme.ai/p/p1', createdAt: '2026-11-24T10:00:00Z', updatedAt: '2026-11-24T10:05:00Z' }] }));
    const link = document.querySelector('#grok-pages a');
    expect(link.textContent).toBe('Сравнение');
    expect(link.getAttribute('href')).toBe('https://vlist-pages.smileme.ai/p/p1');
    expect(link.getAttribute('target')).toBe('_blank');
  });

  it('a link keeps a bracket it opened and drops one that closes the sentence', () => {
    create().render(card({ configured: true, messages: [
      { id: 'r1', role: 'grok', text: 'см. https://en.wikipedia.org/wiki/Mercury_(planet) и (https://example.com/a).', at: '1' }
    ] }));
    expect([...document.querySelectorAll('[data-role=grok] a')].map(link => link.getAttribute('href'))).toEqual(['https://en.wikipedia.org/wiki/Mercury_(planet)', 'https://example.com/a']);
  });

  it('a stale call offers «Повторить» instead of «Грок работает…»', () => {
    create().render(card({ configured: true, messages: [{ id: 'm1', role: 'user', text: 'x', at: '1', state: 'sending', stale: true }] }));
    expect(document.querySelector('.v02-grok-retry')).not.toBeNull();
    expect(document.querySelector('.v02-grok-working')).toBeNull();
  });

  it('an empty field is not sent', async () => {
    const fetch = vi.fn();
    create(fetch).render(card({ configured: true, messages: [] }));
    document.getElementById('grok-input').value = '   ';
    await client.delegateToGrok();
    expect(fetch).not.toHaveBeenCalled();
  });
});

describe('the Grok line on the frontier', () => {
  it('shows the last word and opens the card on the Grok block', () => {
    create().render(frontier({ kind: 'grok', text: 'Нашёл три варианта', at: '2026-11-24T10:01:00Z' }));
    const summary = document.querySelector('.v02-grok-summary');
    expect(summary.dataset.componentId).toBe('task:t1');
    expect(summary.textContent).toContain('Нашёл три варианта');
    const navigate = vi.spyOn(client, 'navigate').mockResolvedValue();
    client.tap({ id: 'task:t1', taskId: 't1', element: summary });
    expect(navigate).toHaveBeenCalledWith('edit', { taskId: 't1', focus: 'grok' });
  });

  it('a moved Grok mark makes the page fetch the document again', async () => {
    const fetch = vi.fn(async (path) => ({ ok: true, status: 200, json: async () => (path.startsWith('/api/v2/updates') ? { events: [], actions: [], uiEffects: [], nextCursor: 0, grok: 'g2' } : frontier(null)) }));
    create(fetch).render(frontier(null));
    client.cursor = 0;
    await client.resume();
    expect(fetch.mock.calls.some(([path]) => path.startsWith('/api/v2/document'))).toBe(true);
  });
});
