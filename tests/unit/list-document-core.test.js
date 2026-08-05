import { describe, expect, test } from 'vitest';

import { seedState } from '../../list-data.js';
import { createDocumentCore } from '../../worker/list-document-core.js';

function workflowyResponse(body, headers = {}) {
  return {
    ok: true,
    status: 200,
    headers: {
      get(name) {
        return headers[name.toLowerCase()] || null;
      }
    },
    async text() {
      return typeof body === 'string' ? body : JSON.stringify(body);
    },
    async json() {
      return typeof body === 'string' ? JSON.parse(body) : body;
    }
  };
}

describe('Cloudflare list document core', () => {
  test('starts from seed and applies commands through the server interpreter log', async () => {
    const core = createDocumentCore({ seedState, openRouterApiKey: '' });
    await core.init();

    const initial = core.getSnapshot();
    expect(initial.rev).toBe(0);
    expect(initial.content.snapshot.items[0]).toMatchObject({
      id: 'inbox',
      line1: 'Входящие'
    });
    expect(initial.content.snapshot.items.some((item) => item.id === 'milk1')).toBe(true);

    const applied = await core.handleClientMessage({
      type: 'command',
      clientKey: 'tab-a',
      seq: 1,
      input: {
        actId: 'list',
        actType: 'list',
        command: 'addItem',
        payload: { line1: 'Server task', line2: '' },
        source: 'unit-test'
      }
    });

    expect(applied.ack).toMatchObject({
      seq: 1,
      status: 'applied',
      newTarget: 'rs'
    });
    expect(applied.state.rev).toBe(1);
    expect(applied.state.content.snapshot.items.find((item) => item.id === 'rs')).toMatchObject({
      id: 'rs',
      parentId: null,
      status: 'Open',
      line1: 'Server task'
    });
    expect(core.listLog()).toHaveLength(1);
    expect(core.listLog()[0]).toMatchObject({
      id: '1',
      rev: 1,
      clientKey: 'tab-a',
      seq: 1,
      op: 'addItem',
      target: 'rs'
    });

    const duplicate = await core.handleClientMessage({
      type: 'command',
      clientKey: 'tab-a',
      seq: 1,
      input: {
        actId: 'list',
        actType: 'list',
        command: 'addItem',
        payload: { line1: 'Server task again', line2: '' },
        source: 'unit-test'
      }
    });

    expect(duplicate.ack).toEqual(applied.ack);
    expect(duplicate.state.rev).toBe(1);
    expect(core.listLog()).toHaveLength(1);
  });

  test('stores voice comments on the addressed log entry', async () => {
    const core = createDocumentCore({ seedState, openRouterApiKey: '' });
    await core.init();

    await core.handleClientMessage({
      type: 'command',
      clientKey: 'tab-a',
      seq: 1,
      input: {
        actId: 'list',
        actType: 'list',
        command: 'addItem',
        payload: { line1: 'Commented task', line2: '' },
        source: 'unit-test'
      }
    });

    const commented = await core.handleClientMessage({
      type: 'command',
      clientKey: 'tab-a',
      seq: 2,
      input: {
        actId: '1',
        actType: 'log-entry',
        command: 'commentLogEntry',
        payload: { text: 'купить сегодня' },
        source: 'unit-test'
      }
    });

    expect(commented.ack).toMatchObject({
      seq: 2,
      status: 'applied',
      newTarget: '1'
    });
    expect(commented.state.content.actionLog.find((entry) => entry.id === '1').comments).toEqual([
      expect.objectContaining({ text: 'купить сегодня' })
    ]);
  });

  test('returns an explicit OpenRouter key error when LLM fallback is required', async () => {
    const core = createDocumentCore({ seedState, openRouterApiKey: '' });
    await core.init();

    const result = await core.handleClientMessage({
      type: 'utterance',
      clientKey: 'tab-a',
      seq: 2,
      target: 'milk1',
      transcript: 'сделай что-нибудь очень странное'
    });

    expect(result.ack).toMatchObject({
      seq: 2,
      status: 'rejected',
      reason: 'OPENROUTER_API_KEY is not configured'
    });
    expect(result.state.rev).toBe(0);
    expect(core.listLog()).toHaveLength(0);
  });

  test('imports a Workflowy shared tree through a server command', async () => {
    const fetchImpl = async (url, init = {}) => {
      const textUrl = String(url);
      if (textUrl.includes('/s/task-tree/')) {
        return workflowyResponse(
          '<script>var PROJECT_TREE_DATA_URL_PARAMS = {"share_id":"Share.123"};</script>',
          { 'set-cookie': 'sessionid=abc; Path=/; HttpOnly' }
        );
      }
      if (textUrl.includes('/get_initialization_data')) {
        expect(init.headers.Cookie).toContain('sessionid=abc');
        return workflowyResponse({
          projectTreeData: {
            auxiliaryProjectTreeInfos: [{
              rootProject: { id: 'root', nm: 'task tree' }
            }],
            initialMostRecentOperationTransactionId: '42'
          }
        });
      }
      if (textUrl.includes('/get_tree_data/')) {
        expect(textUrl).toContain('ot=42');
        return workflowyResponse({
          items: [
            { id: 'child', prnt: 'root', pr: 10, nm: 'Child task' },
            { id: 'nested', prnt: 'child', pr: 10, nm: 'Nested task' }
          ]
        });
      }
      throw new Error(`Unexpected fetch ${textUrl}`);
    };
    const core = createDocumentCore({ seedState, openRouterApiKey: '', fetchImpl });
    await core.init();

    const result = await core.handleClientMessage({
      type: 'command',
      clientKey: 'tab-a',
      seq: 1,
      input: {
        actId: 'list',
        actType: 'list',
        command: 'importWorkflowy',
        payload: { url: 'https://workflowy.com/s/task-tree/iq43ak7FYqEEO1uO' },
        source: 'settings-import'
      }
    });

    expect(result.ack).toMatchObject({
      seq: 1,
      status: 'applied',
      newTarget: 'rs'
    });
    const importedRoot = result.state.content.snapshot.items.find((item) => item.id === 'rs');
    const importedChild = result.state.content.snapshot.items.find((item) => item.parentId === 'rs');
    const importedNested = result.state.content.snapshot.items.find((item) => item.parentId === importedChild.id);
    expect(importedRoot).toMatchObject({ line1: 'task tree', status: 'Open', parentId: null });
    expect(importedChild).toMatchObject({ line1: 'Child task', status: 'Open' });
    expect(importedNested).toMatchObject({ line1: 'Nested task', status: 'Open' });
    expect(core.listLog()[0]).toMatchObject({
      op: 'importWorkflowyTree',
      label: 'Импортировано дерево Workflowy: task tree'
    });
  });
});
