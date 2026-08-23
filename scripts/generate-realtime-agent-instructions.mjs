import { writeFile } from 'node:fs/promises';

import { renderRealtimeAgentInstructionsMarkdown } from './realtime-agent-instructions-doc.mjs';

await writeFile(
  new URL('../docs/realtime-agent-instructions.md', import.meta.url),
  renderRealtimeAgentInstructionsMarkdown(),
  'utf8'
);
