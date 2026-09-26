import { Agent } from 'agents';
import { DocumentRuntime } from '../../src/v2/domain/document-runtime.js';
import { safeError, fail } from '../../src/v2/domain/contracts.js';
import { CompatibilityPort } from './compatibility.js';
import { handleMcpRequest } from './mcp.js';
import { resolveOpenAI } from './model.js';
import { RuntimeStorage } from './runtime-storage.js';
import { LiveLog } from './live-log.js';
import { LiveHost } from './live-host.js';
import { GeminiHost } from './gemini-host.js';
import {
  LiveSettings, VOICE_PROMPT_KEY, BACKEND_PROMPT_KEY, BACKEND_MODEL_KEY, REASONING_KEY,
  GEMINI_PROMPT_KEY, GEMINI_MODEL_KEY, PROMPT_HISTORY_KEY
} from '../../src/v2/domain/live-settings.js';
import { taskTreeFromItems } from '../task-tree.js';
import { taskFrontierFromItems } from '../task-frontier.js';

const API_KEY = 'voicelist.openai-api-key.v1';
const GEMINI_KEY = 'voicelist.gemini-api-key.v1';
/** Carried along with the document when the object moves. */
const LIVE_SETTINGS_KEYS = [
  API_KEY, GEMINI_KEY, 'voicelist.openai-setup-used.v1',
  VOICE_PROMPT_KEY, BACKEND_PROMPT_KEY, BACKEND_MODEL_KEY, REASONING_KEY,
  GEMINI_PROMPT_KEY, GEMINI_MODEL_KEY, PROMPT_HISTORY_KEY
];
export class ListDocumentDO extends Agent {
  static options = { sendIdentityOnConnect: false };
  async onStart() {
    this.runtimeStorage = new RuntimeStorage(this.ctx.storage);
    this.liveLog = new LiveLog(this.ctx.storage);
    this.liveSettings = new LiveSettings(this.ctx.storage);
    this.initializeRuntime(this.runtimeStorage.load());
    // Durable wake-up recovers the append/enqueue gap or an exhausted SDK queue retry.
    await this.scheduleEvery(60, 'recoverPending');
  }
  initializeRuntime(initialState) {
    const resolveModel = this.env.V02_TEST_MODEL === 'local-parser' ? undefined : async context => resolveOpenAI({
      ...context,
      apiKey: this.env.OPENAI_API_KEY || await this.getOpenAIApiKey(),
      model: this.env.AGENT_MODEL || 'gpt-4.1-mini'
    });
    this.runtime = new DocumentRuntime({
      initialState, persist: state => this.runtimeStorage.save(state), scheduler: this,
      resolveModel
    });
    this.port = new CompatibilityPort(this.runtime);
  }
  async submit(input) {
    const receipt = await this.runtime.submit(input);
    if (receipt.status === 'accepted') await this.queue('processPending', {});
    return receipt;
  }
  getDocument(context) { return this.runtime.getDocument(context); }
  follow(cursor, clientKey) { return this.runtime.follow(cursor, clientKey); }
  async processPending() {
    await this.runtime.processPending();
    this.broadcastState();
    if (this.runtime.journal.pending().length) await this.queue('processPending', {});
  }
  async recoverPending() { if (this.runtime.journal.pending().length) await this.processPending(); }
  async scheduleInput(rule) {
    if (!rule || (!Number.isFinite(rule.when) && typeof rule.when !== 'string')) fail('INVALID_INPUT','Некорректное расписание');
    return this.runtime.harness.schedule(rule);
  }
  async cancelInputSchedule(id) { return this.runtime.harness.cancelSchedule(id); }
  async scheduledTrigger(input, schedule) {
    const clientKey = 'schedule:' + schedule.id, seq = Math.max(1, Number(schedule.time) || 1);
    const existing = this.runtime.journal.entries.find(e => e.key.clientKey === clientKey && e.key.seq === seq);
    if (!existing) await this.runtime.submit({ ...input, key: {clientKey,seq}, context: {...input.context,revision:this.runtime.graph.revision} });
    await this.processPending();
  }
  async onConnect(connection) {
    this.setConnectionReadonly(connection, true);
    connection.send(JSON.stringify({type:'state',state:this.port.getSnapshot()}));
  }
  async onMessage(connection, raw) {
    try {
      if (typeof raw !== 'string' || raw.length > 64000) fail('TOO_LARGE','Некорректное сообщение');
      const message = JSON.parse(raw);
      if (message.type === 'hello') { connection.send(JSON.stringify({type:'state',state:this.port.getSnapshot()})); return; }
      let ack;
      if (message.type === 'command') ack = await this.port.applyCommand(message.input, {message});
      else if (message.type === 'utterance') {
        const receipt = await this.runtime.executeAndWait({key:{clientKey:message.clientKey,seq:message.seq},
          context:{elementId:message.target ? 'task:'+message.target : 'app',view:'list',revision:this.runtime.graph.revision},text:message.transcript});
        ack = {seq:message.seq,status:receipt.error?'rejected':'applied',reason:receipt.error?.message,id:receipt.actions[0]?.id,newTarget:receipt.target || receipt.actions[0]?.target};
      } else fail('INVALID_INPUT','Неизвестный тип сообщения');
      connection.send(JSON.stringify({type:'ack',ack}));
      this.broadcastState();
    } catch (error) { connection.send(JSON.stringify({type:'ack',ack:{status:'rejected',reason:safeError(error).message}})); }
  }
  /** Building the snapshot is the most expensive thing the object does, so it is not built
   *  for nobody: with no page listening and no voice session to keep in step there is
   *  nothing to send. */
  broadcastState() {
    const connections = [...this.getConnections()];
    const live = Boolean(this.live?.active);
    if (!connections.length && !live) return;
    const payload = JSON.stringify({type:'state',state:this.port.getSnapshot()});
    for (const connection of connections) { try { connection.send(payload); } catch {} }
    // An edit made by hand has to reach the voice layer too, not only its own tool calls.
    if (this.live?.active) this.live.syncSnapshot().catch(() => {});
  }

  liveHost() {
    if (!this.live) this.live = new LiveHost({
      log: this.liveLog,
      settings: this.liveSettings,
      apiKey: this.env.OPENAI_API_KEY || '',
      store: this.env.LIVE_STORE !== 'off',
      services: {
        readItems: async () => this.runtime.graph.read().items,
        readFrontier: async () => this.getTaskFrontier(),
        applyCommand: async (command, message) => this.port.applyCommand(command, {message})
      }
    });
    return this.live;
  }
  geminiHost() {
    if (!this.gemini) this.gemini = new GeminiHost({
      log: this.liveLog,
      settings: this.liveSettings,
      apiKey: this.env.GEMINI_API_KEY || '',
      services: {
        readItems: async () => this.runtime.graph.read().items,
        readFrontier: async () => this.getTaskFrontier(),
        applyCommand: async (command, message) => this.port.applyCommand(command, {message})
      }
    });
    return this.gemini;
  }
  /** One voice session at a time: two models listening to the same microphone would both
   *  answer, and the log would interleave two conversations about one list. */
  async startGeminiSession() {
    const host = this.geminiHost();
    if (!host.apiKey) host.apiKey = await this.getGeminiApiKey();
    if (this.live?.active) await this.live.stop('replaced-by-gemini');
    const result = await host.mintToken();
    this.broadcastState();
    return result;
  }
  stopGeminiSession() { return this.gemini ? this.gemini.stop('client') : false; }
  geminiSessionStatus() { return {active:Boolean(this.gemini?.active),sessionId:this.gemini?.sessionId || ''}; }
  /** The page relays a tool call and gets back exactly what it must send to Google. */
  /** A read changes nothing, and a repeat was answered from the first result: neither is
   *  worth a snapshot. Only a call that actually applied something redraws the list. */
  async runGeminiTools(frame) {
    const host = this.geminiHost();
    const results = await host.invokeAll(frame);
    if (results.some(item => item.response?.status === 'applied')) this.broadcastState();
    return {results};
  }
  mirrorGeminiFrames(frames) { return this.geminiHost().mirror(frames); }
  async getGeminiApiKey() { return await this.ctx.storage.get(GEMINI_KEY) || ''; }
  async isGeminiKeyConfigured() { return Boolean(this.env.GEMINI_API_KEY || await this.getGeminiApiKey()); }
  /** "Configured" has to mean "works". A stored key that Google refuses is the same as no key
   *  at all for everyone using the app, and reporting it as configured hides the field that
   *  would replace it. The answer is remembered briefly so opening settings is not a round
   *  trip to Google every time. */
  async geminiKeyStatus() {
    const key = this.env.GEMINI_API_KEY || await this.getGeminiApiKey();
    if (!key) return {configured:false, setupAvailable:true};
    const fresh = this.geminiKeyChecked && Date.now() - this.geminiKeyChecked.at < 60_000 && this.geminiKeyChecked.key === key;
    const result = fresh ? this.geminiKeyChecked.result : await this.geminiHost().verifyKey(key);
    this.geminiKeyChecked = {key, at:Date.now(), result};
    return {
      configured: result.ok,
      setupAvailable: !result.ok && !this.env.GEMINI_API_KEY,
      ...(result.ok ? {} : {detail:`Сохранённый ключ отклонён Google${result.status ? ` (${result.status})` : ''}.`})
    };
  }
  /** A key is kept only once Google has confirmed it, and a stored key that no longer works
   *  does not block its replacement. Without both, a single typo closes the door: the app
   *  reports the key as configured and nothing in the interface can take it back. */
  async configureGeminiApiKey(apiKey) {
    const host = this.geminiHost();
    const candidate = await host.verifyKey(apiKey);
    if (!candidate.ok) return {configured:false, reason:'rejected', status:candidate.status || 0, detail:candidate.detail || ''};

    const stored = this.env.GEMINI_API_KEY || await this.getGeminiApiKey();
    if (stored && stored !== apiKey) {
      const existing = await host.verifyKey(stored);
      if (existing.ok) return {configured:false, reason:'already'};
      if (this.env.GEMINI_API_KEY) return {configured:false, reason:'env'};
    }
    await this.ctx.storage.put(GEMINI_KEY, apiKey);
    host.apiKey = apiKey;
    this.geminiKeyChecked = {key:apiKey, at:Date.now(), result:{ok:true}};
    return {configured:true, replaced:Boolean(stored && stored !== apiKey)};
  }
  async startLiveSession(body) {
    const host = this.liveHost();
    if (!host.apiKey) host.apiKey = await this.getOpenAIApiKey();
    if (this.gemini?.active) this.gemini.stop('replaced-by-gpt-live');
    const result = await host.start({sdp:String(body?.sdp || '')});
    this.broadcastState();
    return result;
  }
  async stopLiveSession() { return this.live ? this.live.stop('client') : false; }
  liveSessionStatus() { return {active:Boolean(this.live?.active),sessionId:this.live?.sessionId || ''}; }
  readLiveLog(query) { return {entries:this.liveLog.read(query),stats:this.liveLog.stats()}; }
  clearLiveLog() { return this.liveLog.clear(); }
  async simulateDelegation(turns) {
    const host = this.liveHost();
    if (!host.apiKey) host.apiKey = await this.getOpenAIApiKey();
    return host.simulate(turns);
  }
  async simulateVoiceTurn(turns, options) {
    const host = this.liveHost();
    if (!host.apiKey) host.apiKey = await this.getOpenAIApiKey();
    return host.simulateVoice(turns, options || {});
  }
  async repairLiveLog() {
    const host = this.liveHost();
    if (!host.apiKey) host.apiKey = await this.getOpenAIApiKey();
    return host.repairBackendInputs(this.liveLog.read({ limit: 1000 }));
  }
  async readBackendInput(responseId) {
    const host = this.liveHost();
    if (!host.apiKey) host.apiKey = await this.getOpenAIApiKey();
    return host.readBackendInput(responseId);
  }
  /** Everything the document is, for a move to another object.
   *
   *  The task graph, the journal and the private ledgers travel; so do the prompts and the
   *  keys, because a list that arrives without them looks migrated and works like a fresh
   *  install. The voice log stays behind: it is a diagnostic record of one object's sessions,
   *  not part of the document. */
  async exportEverything() {
    const runtime = this.runtime.exportState();
    const settings = {};
    for (const key of LIVE_SETTINGS_KEYS) {
      const value = await this.ctx.storage.get(key);
      if (value !== undefined) settings[key] = value;
    }
    return {
      version: 1,
      runtime,
      settings,
      counts: { items: runtime.graph.items.length, entries: runtime.entries.length, settings: Object.keys(settings).length }
    };
  }

  /** Refuses a target that already holds a document. A move that silently overwrites is a
   *  move that can destroy the thing it was copying. */
  async importEverything(payload) {
    if (payload?.version !== 1) fail('INVALID_INPUT','Неизвестный формат переноса');
    const existing = this.runtimeStorage.load();
    if (existing && existing.graph?.items?.length > 1) fail('CONFLICT','В целевом объекте уже есть документ');
    await this.runtimeStorage.save(payload.runtime);
    for (const [key, value] of Object.entries(payload.settings || {})) await this.ctx.storage.put(key, value);
    this.initializeRuntime(this.runtimeStorage.load());
    const loaded = this.runtime.exportState();
    return { items: loaded.graph.items.length, entries: loaded.entries.length, revision: loaded.graph.revision };
  }
  /** One command applied the way every other path applies one; used by the benchmark to undo
   *  what it measured. */
  /** What a broadcast has to build before it can send anything. */
  benchSnapshot() { return JSON.stringify(this.port.getSnapshot()).length; }
  async applyTaskCommand(command) {
    const ack = await this.port.applyCommand(command, {message:{clientKey:'bench:'+crypto.randomUUID(), seq:1}});
    this.broadcastState();
    return ack;
  }
  listLiveSessions(limit) { return {sessions:this.liveLog.sessions(limit),stats:this.liveLog.stats()}; }
  readLiveSettings() { return this.liveSettings.read(); }
  writeLivePrompt(target, body) { return this.liveSettings.writePrompt(target,{mode:body?.mode,text:body?.text,source:'settings'}); }
  resetLivePrompt(target) { return this.liveSettings.resetPrompt(target); }
  restoreLivePrompt(target, at) { return this.liveSettings.restorePrompt(target,at); }
  livePromptHistory() { return this.liveSettings.history(); }
  setLiveBackendModel(model) { return this.liveSettings.setBackendModel(model); }
  setLiveReasoningEffort(effort) { return this.liveSettings.setReasoningEffort(effort); }
  setGeminiModel(model) { return this.liveSettings.setGeminiModel(model); }
  async onRequest(request) {
    if (new URL(request.url).pathname === '/mcp') {
      const response = await handleMcpRequest(request, this.port); this.broadcastState(); return response;
    }
    return new Response('Not found',{status:404});
  }
  async mcpRequest(request) {
    const response = await handleMcpRequest(request, this.port);
    this.broadcastState(); return response;
  }
  async getOpenAIApiKey() { return await this.ctx.storage.get(API_KEY) || ''; }
  async isOpenAIKeyConfigured() { return Boolean(this.env.OPENAI_API_KEY || await this.getOpenAIApiKey()); }
  async configureOpenAIApiKey(apiKey) {
    return this.ctx.storage.transaction(async txn => {
      if (await txn.get('voicelist.openai-setup-used.v1')) return false;
      await txn.put({[API_KEY]:apiKey,'voicelist.openai-setup-used.v1':true}); return true;
    });
  }
  async getOpenAISystemPrompt() { return await this.ctx.storage.get('voicelist.openai-system-prompt.v1') || ''; }
  async configureOpenAISystemPrompt(prompt) { await this.ctx.storage.put('voicelist.openai-system-prompt.v1',prompt); return true; }
  getTaskTree() { return taskTreeFromItems(this.runtime.graph.read().items); }
  getTaskFrontier() { return taskFrontierFromItems(this.runtime.graph.read().items); }
  getTaskTitleTreeText() { return this.port.getTaskTitleTreeText(); }
  getTaskItem(id) { return this.port.getTaskTree({id}); }
  async reset() {
    if (this.runtime.processing) await this.runtime.processing;
    await this.runtime.tail;
    this.runtimeStorage.clear(); this.initializeRuntime();
    await this.runtimeStorage.save(this.runtime.exportState());
    this.broadcastState(); return this.port.getSnapshot();
  }
  async recordRealtimeDiagnostics(entry) {
    // Diagnostics deliberately exclude arbitrary client strings (and never include credentials).
    const safe = {at:new Date().toISOString(),outcome:['active','cancelled','error'].includes(entry.outcome)?entry.outcome:'error'};
    for (const key of ['totalMs','microphoneMs','offerMs','sessionRequestMs','remoteDescriptionMs','dataChannelMs']) if(Number.isFinite(entry[key])) safe[key]=entry[key];
    const values = await this.ctx.storage.get('diagnostics') || [];
    await this.ctx.storage.put('diagnostics',[safe,...values].slice(0,50)); return safe;
  }
  async getRealtimeDiagnostics() { return await this.ctx.storage.get('diagnostics') || []; }
}
