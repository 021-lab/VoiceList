import { fail } from '../../src/v2/domain/contracts.js';

/** Speech from a held task is transcribed here rather than in the browser. Browser recognition
 *  knows nothing of the list and turned «рисёрч по сауне» into «решишь по саванне»; this one is
 *  told the titles around the task in hand, so names already in the list come back spelled. */
const MODEL = 'gpt-4o-transcribe';
const PROMPT_LIMIT = 800;
const HEADER = 'Задачи из списка:';
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
  let prompt = HEADER;
  for (const title of titles) {
    const next = `${prompt}\n${title}`;
    if (next.length > PROMPT_LIMIT) break;
    prompt = next;
  }
  return prompt;
}

const normalize = (text) => String(text).toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();

/** On silence the model does not return nothing: it reads the prompt back. Seen on prod — a
 *  second of silence came back as the whole list of titles. A reply that carries the header,
 *  or is made almost entirely of two or more prompted titles, is that echo and counts as empty.
 *  One title said on its own, or inside a request, is speech. */
export function isPromptEcho(text, prompt, held = '') {
  const said = normalize(text);
  if (!said) return false;
  if (said.includes(normalize(HEADER))) return true;
  // Saying only the name of the task in hand while holding it asks nothing; on prod a second
  // of silence came back as exactly that name, the first line of the prompt.
  if (held && said === normalize(held)) return true;
  const titles = [...new Set(prompt.slice(HEADER.length).split('\n').map(normalize).filter(title => title.length >= 4))];
  // A title inside a longer one that also matched is the same words, not a second title.
  const matched = titles.filter(title => said.includes(title));
  const found = matched.filter(title => !matched.some(other => other !== title && other.includes(title)));
  const covered = found.reduce((sum, title) => sum + title.length, 0);
  return found.length >= 2 && covered >= said.replace(/ /g, '').length * 0.8;
}

export async function transcribeOpenAI({ apiKey, audio, contentType = '', prompt = '', held = '', fetchImpl = (...args) => fetch(...args) }) {
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
  const text = String(data.text || '').trim();
  return isPromptEcho(text, prompt, held) ? '' : text;
}
