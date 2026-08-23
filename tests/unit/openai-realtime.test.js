import { describe, expect, test, vi } from 'vitest';

import {
  buildRealtimeSessionConfig,
  handleOpenAIRealtimeSession,
  OPENAI_REALTIME_MODEL,
  TASK_OPERATION_INSTRUCTIONS
} from '../../worker/openai-realtime.js';

describe('OpenAI Realtime Worker session', () => {
  test('builds hidden task context with only the published task operations', () => {
    const config = buildRealtimeSessionConfig([{
      id: 'milk1',
      title: 'Молоко 3.2%',
      status: 'Open',
      children: []
    }]);

    expect(config.model).toBe(OPENAI_REALTIME_MODEL);
    expect(config.instructions).toContain('Молоко 3.2%');
    expect(config.instructions).toContain('current_task_tree_json');
    expect(config.instructions).toContain('Never read, enumerate, or summarize it automatically');
    expect(config.instructions).toContain(TASK_OPERATION_INSTRUCTIONS);
    expect(config.instructions).toContain('A question about a status is not a request to set that status.');
    expect(config.instructions).toContain('"Какой статус у задачи Первый поход?" -> No tool.');
    expect(config.instructions).toContain('"Поставь задачу Первый поход в фокус" -> Call setStatus once');
    expect(config.instructions).toContain('addInfo(parentId, line1)');
    expect(config.instructions).toContain('"Добавь информацию к Яблокам');
    expect(config.instructions).toContain('silently estimate your confidence');
    expect(config.instructions).toContain('The user\'s current spoken title overrides prior conversational focus');
    expect(config.instructions).toContain('Deletion is unavailable');
    expect(config.instructions).toContain('call the matching tool immediately without a spoken preamble');
    expect(config.instructions).not.toContain('Купить вечером');
    expect(config.instructions).not.toContain('line2');
    expect(config.instructions).not.toContain('details');
    expect(config.tools.map((tool) => tool.name)).toEqual([
      'addItem',
      'addChild',
      'addInfo',
      'setStatus',
      'editItem',
      'setParent'
    ]);
    expect(config.tools.flatMap((tool) => Object.keys(tool.parameters.properties))).not.toContain('line2');
    expect(config.tool_choice).toBe('auto');
    expect(config.instructions).not.toContain('deleteItem');
  });

  test('keeps mandatory VoiceList rules after a saved custom prompt', () => {
    const config = buildRealtimeSessionConfig([], {
      systemPrompt: 'Всегда называй себя Помощником.'
    });

    expect(config.instructions.startsWith('Всегда называй себя Помощником.')).toBe(true);
    expect(config.instructions).toContain(TASK_OPERATION_INSTRUCTIONS);
    expect(config.instructions).toContain('current_task_tree_json');
  });

  test('proxies SDP and server-owned session configuration without exposing the API key', async () => {
    const fetchImpl = vi.fn(async (_url, options) => {
      expect(options.headers.Authorization).toBe('Bearer test-openai-key');
      expect(options.headers['OpenAI-Safety-Identifier']).toMatch(/^voicelist-[a-f0-9]{32}$/);
      expect(options.body).toBeInstanceOf(FormData);
      expect(options.body.get('sdp')).toBe('offer-sdp');
      const session = JSON.parse(options.body.get('session'));
      expect(session.instructions).toContain('Task from browser');
      return new Response('answer-sdp', { status: 200 });
    });
    const request = new Request('https://vlist-dev.smileme.ai/api/realtime/session', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Origin: 'https://vlist-dev.smileme.ai',
        'CF-Connecting-IP': '203.0.113.10'
      },
      body: JSON.stringify({
        sdp: 'offer-sdp',
        taskTree: [{ id: 'task1', title: 'Task from browser', status: 'Open', children: [] }]
      })
    });

    const response = await handleOpenAIRealtimeSession(
      request,
      { OPENAI_API_KEY: 'test-openai-key' },
      { fetchImpl }
    );

    expect(fetchImpl).toHaveBeenCalledWith('https://api.openai.com/v1/realtime/calls', expect.any(Object));
    expect(response.status).toBe(200);
    expect(response.headers.get('Content-Type')).toBe('application/sdp');
    await expect(response.text()).resolves.toBe('answer-sdp');
  });

  test('rejects cross-origin and unconfigured session requests', async () => {
    const crossOrigin = new Request('https://vlist-dev.smileme.ai/api/realtime/session', {
      method: 'POST',
      headers: { Origin: 'https://example.com', 'Content-Type': 'application/json' },
      body: JSON.stringify({ sdp: 'offer', tasks: [] })
    });
    expect((await handleOpenAIRealtimeSession(crossOrigin, { OPENAI_API_KEY: 'key' })).status).toBe(403);

    const noKey = new Request('https://vlist-dev.smileme.ai/api/realtime/session', {
      method: 'POST',
      headers: { Origin: 'https://vlist-dev.smileme.ai', 'Content-Type': 'application/json' },
      body: JSON.stringify({ sdp: 'offer', tasks: [] })
    });
    expect((await handleOpenAIRealtimeSession(noKey, {})).status).toBe(503);
  });
});
