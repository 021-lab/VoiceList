import { describe, expect, test, vi } from 'vitest';

import {
  backendStatusToLocal,
  createBackendAdapter,
  localStatusToBackend
} from '../../src/list-backend-adapter.js';

function response(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    async json() {
      return body;
    }
  };
}

const emptyState = {
  snapshot: { items: [] },
  actionLog: []
};

describe('taosmd backend adapter', () => {
  test('maps local and backend statuses', () => {
    expect(localStatusToBackend('Open')).toBe('open');
    expect(localStatusToBackend('Done')).toBe('closed');
    expect(localStatusToBackend('Focus')).toBe('focus');
    expect(localStatusToBackend('Pause')).toBe('pause');
    expect(localStatusToBackend('Archive')).toBe('superseded');

    expect(backendStatusToLocal('open')).toBe('Open');
    expect(backendStatusToLocal('closed')).toBe('Done');
    expect(backendStatusToLocal('focus')).toBe('Focus');
    expect(backendStatusToLocal('pause')).toBe('Pause');
    expect(backendStatusToLocal('superseded')).toBe('Archive');
    expect(backendStatusToLocal('in_progress')).toBe('Open');
  });

  test('bootstraps state from tasks and parent edges', async () => {
    const fetchImpl = vi.fn(async (url) => {
      if (url === '/api/tasks?limit=500&project=voicelist') {
        return response({
          tasks: [
            { id: 't-parent', title: 'Parent', status: 'open', priority: 20, created_ts: 2 },
            { id: 't-child', title: 'Child', status: 'focus', priority: 10, created_ts: 1 }
          ]
        });
      }
      if (url === '/api/tasks/edges?limit=2000') {
        return response({ edges: [{ from_id: 't-child', to_id: 't-parent', type: 'parent' }] });
      }
      throw new Error(`unexpected ${url}`);
    });

    const adapter = createBackendAdapter({ fetchImpl });
    const state = await adapter.load();

    expect(state.snapshot.items).toEqual([
      {
        id: 't-parent',
        parentId: null,
        order: 20,
        status: 'Open',
        line1: 'Parent',
        collapsed: false,
        tags: []
      },
      {
        id: 't-child',
        parentId: 't-parent',
        order: 10,
        status: 'Focus',
        line1: 'Child',
        collapsed: false,
        tags: []
      }
    ]);
  });

  test('creates tasks, paired child edges, status update, and A2A log', async () => {
    const calls = [];
    const fetchImpl = vi.fn(async (url, options = {}) => {
      calls.push({ url, options });
      if (url.startsWith('/api/tasks?')) {
        return response({
          tasks: [{ id: 't-parent', title: 'Parent', status: 'open', priority: 10, created_ts: 1 }]
        });
      }
      if (url.startsWith('/api/tasks/edges?')) return response({ edges: [] });
      if (url === '/api/tasks' && options.method === 'POST') {
        const body = JSON.parse(options.body);
        return response({
          id: body.title === 'Parent' ? 't-parent' : 't-child',
          title: body.title,
          status: body.status || 'open',
          priority: body.priority || 0,
          created_ts: body.title === 'Parent' ? 1 : 2
        });
      }
      if (url === '/api/tasks/t-child/edges' && options.method === 'POST') return response({});
      if (url === '/api/tasks/t-parent' && options.method === 'POST') return response({});
      if (url === '/api/a2a/send' && options.method === 'POST') return response({ id: 1 });
      throw new Error(`unexpected ${url}`);
    });

    const adapter = createBackendAdapter({ fetchImpl });
    const saved = await adapter.save({
      snapshot: {
        items: [
          { id: 't-parent', parentId: null, order: 10, status: 'Focus', line1: 'Parent', collapsed: false, tags: [] },
          { id: 'local-child', parentId: 't-parent', order: 10, status: 'Open', line1: 'Child', collapsed: false, tags: [] }
        ]
      },
      actionLog: [
        {
          id: 'log1',
          createdAt: '2026-07-20T12:00:00.000Z',
          command: { command: 'setStatus', actId: 'local-parent', payload: { status: 'Focus' } },
          label: 'Статус изменён: Focus',
          syncStatus: 'pending'
        }
      ]
    }, { actionLogEntry: { id: 'log1' } });

    expect(saved.snapshot.items.map((item) => item.id)).toEqual(['t-parent', 't-child']);
    expect(saved.snapshot.items[1].parentId).toBe('t-parent');
    expect(calls.some((call) => call.url === '/api/tasks/t-child/edges' && call.options.body.includes('"type":"parent"'))).toBe(true);
    expect(calls.some((call) => call.url === '/api/tasks/t-child/edges' && call.options.body.includes('"type":"blocks"'))).toBe(true);
    expect(calls.some((call) => call.url === '/api/tasks/t-parent' && call.options.body.includes('"status":"focus"'))).toBe(true);
    expect(calls.some((call) => call.url === '/api/tasks' && call.options.body.includes('"project":"voicelist"'))).toBe(true);

    const logCall = calls.find((call) => call.url === '/api/a2a/send');
    expect(logCall).toBeTruthy();
    const logPayload = JSON.parse(JSON.parse(logCall.options.body).body);
    expect(logPayload).toMatchObject({
      schema: 'voicelist.action.v1',
      id: 'log1',
      op: 'setStatus'
    });
  });
});
