import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { Client } from '../../src/v2/client/client.js';
import { GestureController } from '../../src/v2/client/gesture-controller.js';
import { MockASR } from '../../src/asr-mock.js';

const node = (type, id, props = {}, children = []) => ({ type, id, props, children });
const task = (id, props = {}) => node('task', `task:${id}`, { taskId: id, line1: id, status: 'Open', parentId: null, level: 0, ...props });
const documentTree = (children = [task('one')], view = 'list', cursor = 0) => ({ schemaVersion: 1, revision: 3, cursor, view, root: node('application', 'app', {}, [node('toolbar', 'toolbar'), node('task-list', 'screen:list', {}, children)]) });
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
let client;
beforeEach(() => {
  const template = readFileSync('list-manager.template.html', 'utf8');
  document.body.innerHTML = template.match(/<body>([\s\S]*)<script>/)[1];
  sessionStorage.clear();
  vi.stubGlobal('scrollTo', vi.fn());
});
afterEach(() => { client?.disconnect(); vi.useRealTimers(); vi.restoreAllMocks(); });
function create(fetcher) {
  client = new Client({ document, storage: sessionStorage, pollMs: 0, fetch: fetcher || vi.fn(async () => response(documentTree())) });
  client.mount(); return client;
}

describe('server component Client', () => {
  it('renders escaped task data using stable component and task IDs', () => {
    create().render(documentTree([task('one', { line1: '<img src=x onerror=alert(1)>' })]));
    expect(document.querySelector('[data-component-id="task:one"]')).toBeTruthy();
    expect(document.querySelector('.item-line1').textContent).toContain('<img');
    expect(document.querySelector('.item-line1 img')).toBeNull();
  });
  it('collapses and expands server-collapsed branches locally without sending a mutation', () => {
    const fetcher = vi.fn(); create(fetcher);
    const parent = task('one', { hasChildren: true, collapsed: true });
    client.render(documentTree([parent, task('child', { parentId: 'one', level: 1 })]));
    expect(document.querySelector('[data-id="child"]').hidden).toBe(true);
    client.tap({ taskId: 'one', node: parent, element: document.querySelector('[data-id="one"]') });
    expect(document.querySelector('[data-id="child"]').hidden).toBe(false);
    expect(fetcher).not.toHaveBeenCalled();
  });
  it('does not interpret or stream text, sends only a final transcript with context', async () => {
    const posts = [];
    create(async (url, options) => { if (options?.method === 'POST') { posts.push(JSON.parse(options.body)); return response({ status: 'completed', requestId: 'e1' }); } return response(documentTree()); });
    client.render(documentTree()); client.asrFactory = () => new MockASR({ phrase: 'назови это новая задача' });
    const target = { id: 'task:one', element: document.querySelector('[data-id="one"]') };
    client.startVoice(target, { x: 100, y: 200 });
    expect(posts).toHaveLength(0);
    expect(document.getElementById('v02-transcript').textContent).toBe('назови это новая задача');
    await client.finishVoice(target);
    expect(posts).toHaveLength(1);
    expect(posts[0]).toMatchObject({ text: 'назови это новая задача', context: { elementId: 'task:one', revision: 3 } });
    expect(posts[0].command).toBeUndefined();
  });
  it('downward edit keeps transcript local until Send, Cancel sends nothing', async () => {
    const fetcher = vi.fn(async () => response(documentTree())); create(fetcher).render(documentTree());
    client.asrFactory = () => new MockASR({ phrase: 'исправь задачу' });
    const target = { id: 'task:one', element: document.querySelector('[data-id="one"]') };
    client.startVoice(target, { x: 100, y: 200 }); await client.finishVoice(target, true);
    expect(document.getElementById('transcript-edit-input').value).toBe('исправь задачу');
    expect(fetcher).not.toHaveBeenCalled();
    document.querySelector('.v02-edit-transcript .btn-cancel').click();
    expect(document.querySelector('.v02-edit-transcript')).toBeNull(); expect(fetcher).not.toHaveBeenCalled();
  });
  it('persists network retry with exactly the same request key and drops terminal rejection', async () => {
    let fail = true; const bodies = [];
    create(async (url, options) => {
      if (options?.method !== 'POST') return response(documentTree());
      bodies.push(JSON.parse(options.body)); if (fail) throw new Error('offline');
      return response({ error: { message: 'No such task' } }, 404);
    }).render(documentTree());
    await expect(client.submit({ text: 'команда' })).rejects.toThrow('offline');
    expect(client.pending.size).toBe(1); fail = false;
    await expect(client.flush()).rejects.toThrow('No such task');
    expect(bodies[0]).toEqual(bodies[1]); expect(client.pending.size).toBe(0);
  });
  it('document reload does not consume unseen action events and stacks toasts', async () => {
    create(async () => response({ events: [], actions: [{ actionId: 'a1', label: 'Первое' }, { actionId: 'a2', label: 'Второе' }], nextCursor: 5 })).render(documentTree([], 'list', 2));
    client.render(documentTree([], 'list', 5));
    expect(client.cursor).toBe(2);
    client.loadDocument = vi.fn(async () => {});
    await client.resume();
    expect(document.querySelectorAll('.v02-toast')).toHaveLength(2);
    expect(client.cursor).toBe(5);
  });
  it('action page exposes correction dialog, rollback and close with same action identity', () => {
    create(); client.viewContext = { view: 'action', actionId: 'a1' };
    const tree = documentTree([], 'action'); tree.root.children[1] = node('action-page', 'action:a1', { actionId: 'a1', title: 'Изменено', sourceText: 'назови иначе', result: 'Готово', canRollback: true, messages: [{ role: 'assistant', text: 'Что исправить?' }] });
    client.render(tree);
    expect(document.getElementById('action-correction-input')).toBeTruthy();
    expect(document.getElementById('action-rollback').disabled).toBe(false);
    expect(document.getElementById('action-close').textContent).toBe('Закрыть');
    expect(document.querySelector('[data-role="assistant"]').textContent).toBe('Что исправить?');
  });
  it('shows unknown component safely instead of executing arbitrary markup', () => {
    create().render(documentTree([node('script', 'unsafe', { html: '<script>bad()</script>' })]));
    expect(document.querySelector('.v02-error').textContent).toContain('script');
  });
  it('uses idempotent receipt polling to wait for actual execution', async () => {
    const body = { key: { clientKey: 'test', seq: 1 }, text: 'test' };
    create(async (url, options) => options?.method === 'POST' ? response({ status: 'completed', actions: [{ target: 'one' }] }) : response(documentTree())).render(documentTree());
    client.requests.set('request1', body);
    expect(await client.waitForCompletion('request1')).toEqual({ status: 'applied', newTarget: 'one' });
  });
});

describe('exclusive gesture state machine', () => {
  function make() {
    vi.useFakeTimers();
    const callbacks = Object.fromEntries(['startVoice', 'finishVoice', 'cancelVoice', 'editVoice', 'drag', 'swipe', 'tap'].map(k => [k, vi.fn()]));
    return { callbacks, gesture: new GestureController(callbacks) };
  }
  it('hold then release sends one final voice input', async () => {
    const { gesture, callbacks } = make(); gesture.begin({ x: 0, y: 100 }, {}); vi.advanceTimersByTime(301); await gesture.end();
    expect(callbacks.startVoice).toHaveBeenCalledOnce(); expect(callbacks.finishVoice).toHaveBeenCalledOnce(); expect(callbacks.tap).not.toHaveBeenCalled();
  });
  it('up selects drag and moving back down never enters transcript editor', async () => {
    const { gesture, callbacks } = make(); gesture.begin({ x: 0, y: 100 }, { draggable: true }); vi.advanceTimersByTime(301);
    gesture.move({ x: 0, y: 70 }); gesture.move({ x: 0, y: 300 }); await gesture.end();
    expect(callbacks.cancelVoice).toHaveBeenCalledOnce(); expect(callbacks.drag).toHaveBeenLastCalledWith('end', expect.anything(), expect.anything(), expect.anything());
    expect(callbacks.finishVoice).not.toHaveBeenCalled(); expect(callbacks.editVoice).not.toHaveBeenCalled();
  });
  it('down edits, further down cancels without sending', async () => {
    const { gesture, callbacks } = make(); gesture.begin({ x: 0, y: 100 }, {}); vi.advanceTimersByTime(301); gesture.move({ x: 0, y: 150 }); await gesture.end();
    expect(callbacks.editVoice).toHaveBeenCalledOnce();
    callbacks.editVoice.mockClear(); gesture.begin({ x: 0, y: 100 }, {}); vi.advanceTimersByTime(301); gesture.move({ x: 0, y: 250 }); await gesture.end();
    expect(callbacks.editVoice).not.toHaveBeenCalled(); expect(callbacks.finishVoice).not.toHaveBeenCalled(); expect(callbacks.cancelVoice).toHaveBeenCalledOnce();
  });
  it('ordinary scrolling cancels pending hold without opening microphone', () => {
    const { gesture, callbacks } = make(); gesture.begin({ x: 0, y: 100 }, {}); gesture.move({ x: 0, y: 120 }); vi.advanceTimersByTime(400);
    expect(gesture.state).toBe('idle'); expect(callbacks.startVoice).not.toHaveBeenCalled();
  });
  it('small downward selection remains editing even when finger returns upward', async () => {
    const { gesture, callbacks } = make(); gesture.begin({ x: 0, y: 100 }, { draggable: true }); vi.advanceTimersByTime(301);
    gesture.move({ x: 0, y: 130 }); expect(gesture.state).toBe('editing');
    gesture.move({ x: 0, y: 80 }); expect(gesture.state).toBe('editing');
    await gesture.end(); expect(callbacks.editVoice).toHaveBeenCalledOnce(); expect(callbacks.drag).not.toHaveBeenCalled();
  });
  it('deep cancellation is terminal until release even after moving back', async () => {
    const { gesture, callbacks } = make(); gesture.begin({ x: 0, y: 100 }, { draggable: true }); vi.advanceTimersByTime(301);
    gesture.move({ x: 0, y: 245 }); gesture.move({ x: 0, y: 60 });
    expect(gesture.state).toBe('cancelled'); await gesture.end();
    expect(callbacks.finishVoice).not.toHaveBeenCalled(); expect(callbacks.editVoice).not.toHaveBeenCalled(); expect(callbacks.drag).not.toHaveBeenCalled();
  });
});
