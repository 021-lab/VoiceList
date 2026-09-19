import { DEFAULT_BACKEND_MODEL, DEFAULT_BACKEND_PROMPT, DEFAULT_VOICE_PROMPT } from './live-session.js';
import { fail } from './contracts.js';

export const VOICE_PROMPT_KEY = 'voicelist.live.voice-prompt.v1';
export const BACKEND_PROMPT_KEY = 'voicelist.live.backend-prompt.v1';
export const BACKEND_MODEL_KEY = 'voicelist.live.backend-model.v1';
export const PROMPT_HISTORY_KEY = 'voicelist.live.prompt-history.v1';

export const MAX_PROMPT_CHARS = 16_000;
export const MAX_HISTORY = 50;
export const PROMPT_TARGETS = ['voice', 'backend'];

const KEY = { voice: VOICE_PROMPT_KEY, backend: BACKEND_PROMPT_KEY };
const DEFAULT = { voice: DEFAULT_VOICE_PROMPT, backend: DEFAULT_BACKEND_PROMPT };

/** An empty stored value means "use the built-in default".
 *
 *  The alternative — copying the shipped text into storage the first time settings open —
 *  breaks twice. An improved default stops reaching anyone who once saved an edit, because
 *  the app reads their copy of an older version and that reads as a worse model. And
 *  returning to the default becomes impossible, because a stored copy is indistinguishable
 *  from deliberately written text and there is no reference left to restore. With the model
 *  able to rewrite these prompts, that return is the last line of rollback. */
export class LiveSettings {
  constructor(storage) { this.storage = storage; }

  async stored(target) { return (await this.storage.get(KEY[target])) || ''; }

  async prompt(target) { return (await this.stored(target)) || DEFAULT[target]; }

  async backendModel() { return (await this.storage.get(BACKEND_MODEL_KEY)) || ''; }

  async read() {
    const [voice, backend, model, history] = await Promise.all([
      this.stored('voice'), this.stored('backend'), this.backendModel(), this.history()
    ]);
    return {
      voicePrompt: voice || DEFAULT_VOICE_PROMPT,
      backendPrompt: backend || DEFAULT_BACKEND_PROMPT,
      backendModel: model || DEFAULT_BACKEND_MODEL,
      defaults: { voicePrompt: !voice, backendPrompt: !backend, backendModel: !model },
      history: history.map(({ before, after, ...rest }) => ({ ...rest, beforeChars: before.length, afterChars: after.length }))
    };
  }

  async history() { return (await this.storage.get(PROMPT_HISTORY_KEY)) || []; }

  /** Both texts are kept whole: an edit made by the model has to be readable after the fact,
   *  and the version it replaced has to be restorable. */
  async record(entry) {
    const history = await this.history();
    await this.storage.put(PROMPT_HISTORY_KEY, [{ ...entry }, ...history].slice(0, MAX_HISTORY));
  }

  async writePrompt(target, { mode = 'replace', text = '', source = 'settings' } = {}) {
    if (!PROMPT_TARGETS.includes(target)) fail('INVALID_INPUT', 'Неизвестный промпт');
    if (!['append', 'replace'].includes(mode)) fail('INVALID_INPUT', 'Неизвестный режим правки');
    const addition = String(text ?? '').trim();
    if (mode === 'append' && !addition) fail('INVALID_INPUT', 'Пустое правило');
    const before = await this.prompt(target);
    const after = mode === 'append' ? `${before}\n${addition}`.trim() : addition;
    if (after.length > MAX_PROMPT_CHARS) fail('TOO_LARGE', 'Промпт слишком длинный');
    if (after === before) return { target, changed: false, prompt: after };
    await this.storage.put(KEY[target], after);
    await this.record({ at: new Date().toISOString(), target, mode, source, before, after });
    return { target, changed: true, prompt: after, usingDefault: false };
  }

  /** Clearing the field is the return to the built-in default. */
  async resetPrompt(target, { source = 'settings' } = {}) {
    if (!PROMPT_TARGETS.includes(target)) fail('INVALID_INPUT', 'Неизвестный промпт');
    const before = await this.prompt(target);
    await this.storage.put(KEY[target], '');
    if (before !== DEFAULT[target]) await this.record({ at: new Date().toISOString(), target, mode: 'reset', source, before, after: DEFAULT[target] });
    return { target, changed: true, prompt: DEFAULT[target], usingDefault: true };
  }

  async restorePrompt(target, at, { source = 'settings' } = {}) {
    if (!PROMPT_TARGETS.includes(target)) fail('INVALID_INPUT', 'Неизвестный промпт');
    const version = (await this.history()).find(entry => entry.target === target && entry.at === at);
    if (!version) fail('NOT_FOUND', 'Версия промпта не найдена');
    return this.writePrompt(target, { mode: 'replace', text: version.before, source });
  }

  /** Free text rather than a list: a list of known models goes stale faster than it is
   *  updated. A typo surfaces at the first session as a rejection from the API, which the
   *  log records unambiguously. */
  async setBackendModel(model) {
    const value = String(model ?? '').trim();
    if (value && !/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(value)) fail('INVALID_INPUT', 'Недопустимое имя модели');
    await this.storage.put(BACKEND_MODEL_KEY, value);
    return { backendModel: value || DEFAULT_BACKEND_MODEL, usingDefault: !value };
  }
}

/** Version markers travel with the session start record, because a log entry from last week
 *  cannot be read without knowing which text drove the model then. */
export function promptVersion(text) {
  let hash = 2_166_136_261;
  for (let index = 0; index < text.length; index++) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16_777_619);
  }
  return `${(hash >>> 0).toString(36)}-${text.length}`;
}
