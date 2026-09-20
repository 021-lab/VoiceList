import { Agent } from 'agents';
import { DocumentRuntime } from '../../src/v2/domain/document-runtime.js';
import { safeError, fail } from '../../src/v2/domain/contracts.js';
import { CompatibilityPort } from './compatibility.js';
import { handleMcpRequest } from './mcp.js';
import { resolveOpenAI } from './model.js';
import { RuntimeStorage } from './runtime-storage.js';
import { LiveLog } from './live-log.js';
import { LiveHost } from './live-host.js';
import { LiveSettings } from '../../src/v2/domain/live-settings.js';
import { taskTreeFromItems } from '../task-tree.js';
import { taskFrontierFromItems } from '../task-frontier.js';

const API_KEY = 'voicelist.openai-api-key.v1';
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
  broadcastState() {
    const payload = JSON.stringify({type:'state',state:this.port.getSnapshot()});
    for (const connection of this.getConnections()) { try { connection.send(payload); } catch {} }
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
  async startLiveSession(body) {
    const host = this.liveHost();
    if (!host.apiKey) host.apiKey = await this.getOpenAIApiKey();
    const result = await host.start({sdp:String(body?.sdp || '')});
    this.broadcastState();
    return result;
  }
  async stopLiveSession() { return this.live ? this.live.stop('client') : false; }
  liveSessionStatus() { return {active:Boolean(this.live?.active),sessionId:this.live?.sessionId || ''}; }
  readLiveLog(query) { return {entries:this.liveLog.read(query),stats:this.liveLog.stats()}; }
  clearLiveLog() { return this.liveLog.clear(); }
  listLiveSessions(limit) { return {sessions:this.liveLog.sessions(limit),stats:this.liveLog.stats()}; }
  readLiveSettings() { return this.liveSettings.read(); }
  writeLivePrompt(target, body) { return this.liveSettings.writePrompt(target,{mode:body?.mode,text:body?.text,source:'settings'}); }
  resetLivePrompt(target) { return this.liveSettings.resetPrompt(target); }
  restoreLivePrompt(target, at) { return this.liveSettings.restorePrompt(target,at); }
  livePromptHistory() { return this.liveSettings.history(); }
  setLiveBackendModel(model) { return this.liveSettings.setBackendModel(model); }
  setLiveReasoningEffort(effort) { return this.liveSettings.setReasoningEffort(effort); }
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
