import { Client } from './client.js';
import { createRealtimeVoiceAgent } from '../../realtime-voice-agent.js';

async function bootstrap() {
  const client = new Client();
  window.__voiceListClient = client;
  document.documentElement.dataset.previewBuildHash = typeof __PREVIEW_BUILD_HASH__ === 'undefined' ? 'v02-dev' : __PREVIEW_BUILD_HASH__;
  await client.connect();
  const $ = (id) => document.getElementById(id);
  const realtime = createRealtimeVoiceAgent({
    voiceButton: $('realtime-voice-btn'), voiceStatus: $('realtime-voice-status'),
    dialoguesButton: $('dialogues-tab-btn'), dialoguesPanel: $('dialogues-panel'), dialoguesList: $('dialogues-list'), dialoguesClose: $('dialogues-close'),
    openAIKeyInput: $('openai-key-input'), openAIKeyField: $('openai-key-field'), openAIKeySaveButton: $('openai-key-save'), openAIKeyStatus: $('openai-key-status'),
    openAIPromptInput: $('openai-prompt-input'), openAIPromptSaveButton: $('openai-prompt-save'), openAIPromptStatus: $('openai-prompt-status'),
    settingsOverlay: $('settings-overlay'), rootPanel: $('app-root'), navigationButtons: [$('frontier-tab-btn'), $('view-toggle-btn')],
    getTaskState: () => client.getTaskState(),
    fetchImpl: async (url, options) => {
      if (url === '/api/realtime/session' && options?.method === 'POST') {
        const { tasks } = await client.request('/api/tasks/tree.json');
        options = { ...options, body: JSON.stringify({ ...JSON.parse(options.body), taskTree: tasks }) };
      }
      return fetch(url, options);
    },
    executeTaskCommand: async (command) => {
      const receipt = await client.submit({ command, context: client.context(command.actId ? { id: `task:${command.actId}` } : null) });
      if (receipt.status === 'accepted') return client.waitForCompletion(receipt.requestId);
      return { status: 'applied', newTarget: receipt.newTarget || command.actId };
    }
  });
  client.realtime = realtime;
  window.__realtimeVoiceAgent = realtime;
}

bootstrap().catch((error) => {
  console.error('VoiceList v0.2 failed to start', error);
  const message = document.createElement('p'); message.className = 'v02-error'; message.textContent = `Не удалось открыть приложение: ${error.message}`;
  document.getElementById('app-root')?.prepend(message);
});
