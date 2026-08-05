import { describe, expect, test } from 'vitest';

import { importWorkflowyTreeFromUrl, parseWorkflowyPlainTextExport } from '../../src/workflowy-import.js';

const SHARE_HTML = `
<script>
var PROJECT_TREE_DATA_URL_PARAMS = {"share_id": "Share.123"};
</script>
`;

const INIT_DATA = {
  projectTreeData: {
    auxiliaryProjectTreeInfos: [{
      rootProject: {
        id: 'root',
        nm: 'task tree'
      }
    }],
    initialMostRecentOperationTransactionId: '42'
  }
};

const TREE_DATA = {
  items: [
    { id: 'a', prnt: 'root', pr: 20, nm: 'Second' },
    { id: 'b', prnt: 'root', pr: 10, nm: 'First' },
    { id: 'b1', prnt: 'b', pr: 10, nm: '<b>Nested</b>' }
  ]
};

function response(body, headers = {}) {
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

describe('Workflowy import', () => {
  test('parses Workflowy Plain Text export indentation', () => {
    expect(parseWorkflowyPlainTextExport(`task tree

- First
  - Nested
- multiline
  continuation
- Second
- `)).toEqual({
      title: 'task tree',
      children: [
        { title: 'First', children: [{ title: 'Nested', children: [] }] },
        { title: 'multiline continuation', children: [] },
        { title: 'Second', children: [] }
      ]
    });
  });

  test('loads a shared Workflowy tree through the page Plain Text export flow', async () => {
    const calls = [];
    const fetchImpl = async (url, init = {}) => {
      calls.push({
        url: String(url),
        cookie: init.headers?.Cookie || init.headers?.cookie || '',
        accept: init.headers?.Accept || init.headers?.accept || '',
        requestedWith: init.headers?.['X-Requested-With'] || init.headers?.['x-requested-with'] || ''
      });
      if (String(url).includes('/s/task-tree/')) {
        return response(SHARE_HTML, { 'set-cookie': 'sessionid=abc; Path=/; HttpOnly' });
      }
      if (String(url).includes('/get_initialization_data')) return response(INIT_DATA);
      if (String(url).includes('/get_tree_data/')) return response(TREE_DATA);
      throw new Error(`Unexpected URL ${url}`);
    };

    const tree = await importWorkflowyTreeFromUrl('https://workflowy.com/s/task-tree/iq43ak7FYqEEO1uO', { fetchImpl });

    expect(tree).toEqual({
      title: 'task tree',
      children: [
        {
          title: 'First',
          children: [{ title: 'Nested', children: [] }]
        },
        {
          title: 'Second',
          children: []
        }
      ]
    });
    expect(calls[1].cookie).toContain('sessionid=abc');
    expect(calls[1].accept).toBe('application/json');
    expect(calls[1].requestedWith).toBe('XMLHttpRequest');
    expect(calls[2].url).toContain('ot=42');
    expect(calls[2].accept).toBe('application/json');
    expect(calls[2].requestedWith).toBe('XMLHttpRequest');
  });
});
