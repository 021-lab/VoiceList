import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { describe, expect, test } from 'vitest';

import { renderRealtimeAgentInstructionsMarkdown } from '../../scripts/realtime-agent-instructions-doc.mjs';

describe('Realtime agent instructions documentation', () => {
  test('matches the current Realtime configuration source', async () => {
    const actual = await readFile(
      resolve(process.cwd(), 'docs/realtime-agent-instructions.md'),
      'utf8'
    );

    expect(actual).toBe(renderRealtimeAgentInstructionsMarkdown());
  });
});
