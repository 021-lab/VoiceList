import { fail } from '../../src/v2/domain/contracts.js';

/** Speech from a held task is transcribed here rather than in the browser. Browser recognition
 *  knows nothing of the list and turned «рисёрч по сауне» into «решишь по саванне»; this one is
 *  told the titles around the task in hand, so names already in the list come back spelled. */
const MODEL = 'gpt-4o-transcribe';
const PROMPT_LIMIT = 800;
const ACTIVE = new Set(['Open', 'Focus', 'Pause', 'Info']);
const EXTENSIONS = [['audio/webm', 'webm'], ['audio/mp4', 'mp4'], ['audio/m4a', 'm4a'], ['audio/x-m4a', 'm4a'], ['audio/aac', 'mp4'], ['audio/ogg', 'ogg'], ['audio/wav', 'wav'], ['audio/mpeg', 'mp3']];

/** Titles nearest to the task in hand first: the task, its parent, its neighbours and children,
 *  then the rest of the active list until the prompt is full. */
export function transcriptionPrompt(items = [], targetId = null) {
  const active = items.filter(item => item.id !== 'inbox' && ACTIVE.has(item.status));
  const target = active.find(item => item.id === targetId);
  const near = target ? [
    target,
    ...active.filter(item => item.id === target.parentId),
    ...active.filter(item => item.id !== target.id && (item.parentId === target.parentId || item.parentId === target.id))
  ] : [];
  const titles = [...new Set([...near, ...active].map(item => item.title.trim()).filter(Boolean))];
  let prompt = 'Задачи из списка:';
  for (const title of titles) {
    const next = `${prompt} ${title}.`;
    if (next.length > PROMPT_LIMIT) break;
    prompt = next;
  }
  return prompt;
}

export async function transcribeOpenAI({ apiKey, audio, contentType = '', prompt = '', fetchImpl = (...args) => fetch(...args) }) {
  const type = String(contentType).split(';')[0].trim().toLowerCase();
  const extension = EXTENSIONS.find(([known]) => known === type)?.[1];
  if (!extension) fail('INVALID_INPUT', 'Неподдерживаемый формат звука');
  if (!apiKey) fail('MODEL_UNAVAILABLE', 'Для распознавания речи настройте ключ OpenAI в Настройках');
  const form = new FormData();
  form.append('file', new File([audio], `speech.${extension}`, { type }));
  form.append('model', MODEL);
  form.append('language', 'ru');
  if (prompt) form.append('prompt', prompt);
  const response = await fetchImpl('https://api.openai.com/v1/audio/transcriptions', {
    method: 'POST', headers: { Authorization: 'Bearer ' + apiKey }, body: form, signal: AbortSignal.timeout(30000)
  });
  if (!response.ok) { await response.body?.cancel(); fail('MODEL_UNAVAILABLE', `Распознавание недоступно (${response.status})`); }
  const data = await response.json();
  return String(data.text || '').trim();
}
