import { Client } from './client.js';
import { createLiveVoice } from './live-voice.js';

async function bootstrap() {
  const client = new Client();
  window.__voiceListClient = client;
  document.documentElement.dataset.previewBuildHash = typeof __PREVIEW_BUILD_HASH__ === 'undefined' ? 'v02-dev' : __PREVIEW_BUILD_HASH__;
  await client.connect();
  const $ = (id) => document.getElementById(id);

  // Voice is thin here on purpose: the browser only carries audio, while the session's
  // events, its log and every tool call live on the server behind the sideband connection.
  const live = createLiveVoice({
    voiceButton: $('realtime-voice-btn'), voiceStatus: $('realtime-voice-status'),
    dialoguesButton: $('dialogues-tab-btn'), dialoguesPanel: $('dialogues-panel'),
    dialoguesList: $('dialogues-list'), dialoguesClose: $('dialogues-close'),
    rootPanel: $('app-root'), onCloseDialogues: () => client.navigate('list'),
    settingsElements: {
      keyInput: $('openai-key-input'), keyField: $('openai-key-field'), keySave: $('openai-key-save'), keyStatus: $('openai-key-status'),
      voicePromptInput: $('live-voice-prompt'), voicePromptSave: $('live-voice-prompt-save'), voicePromptReset: $('live-voice-prompt-reset'), voicePromptStatus: $('live-voice-prompt-status'),
      backendPromptInput: $('live-backend-prompt'), backendPromptSave: $('live-backend-prompt-save'), backendPromptReset: $('live-backend-prompt-reset'), backendPromptStatus: $('live-backend-prompt-status'),
      backendModelInput: $('live-backend-model'), backendModelSave: $('live-backend-model-save'), backendModelStatus: $('live-backend-model-status')
    }
  });
  client.realtime = live;
  window.__liveVoice = live;
  $('settings-btn')?.addEventListener('click', () => { live.settings.load(); });
  await live.settings.load();
}

bootstrap().catch((error) => {
  console.error('VoiceList v0.2 failed to start', error);
  const message = document.createElement('p'); message.className = 'v02-error'; message.textContent = `Не удалось открыть приложение: ${error.message}`;
  document.getElementById('app-root')?.prepend(message);
});
