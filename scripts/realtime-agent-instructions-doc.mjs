import {
  OPENAI_REALTIME_MODEL,
  TASK_OPERATION_TOOLS,
  getDefaultRealtimeSystemPrompt
} from '../worker/openai-realtime.js';

export function renderRealtimeAgentInstructionsMarkdown() {
  const inlineCode = String.fromCharCode(96);
  const codeFence = inlineCode.repeat(3);
  const tools = JSON.stringify(TASK_OPERATION_TOOLS, null, 2);
  return [
    '# Realtime Agent Instructions',
    '',
    '> Generated reference, not a runtime input file. The runtime source is',
    '> worker/openai-realtime.js. Refresh with npm run docs:realtime whenever',
    '> the model, instructions, or tools change.',
    '',
    'Model: ' + inlineCode + OPENAI_REALTIME_MODEL + inlineCode,
    '',
    '## Instructions',
    '',
    'The placeholder below is intentional: the browser supplies the current task',
    'tree when it creates a session. No live development tasks are committed here.',
    '',
    codeFence + 'text',
    getDefaultRealtimeSystemPrompt(),
    codeFence,
    '',
    '## Tools',
    '',
    'Generated from TASK_OPERATION_TOOLS.',
    '',
    codeFence + 'json',
    tools,
    codeFence,
    '',
    '## Tool Choice',
    '',
    'Generated from buildRealtimeSessionConfig().tool_choice.',
    '',
    codeFence + 'json',
    '"auto"',
    codeFence,
    ''
  ].join('\n');
}
