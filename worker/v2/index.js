import { getAgentByName } from 'agents';
import { LIST_MANAGER_HTML } from '../generated-html.js';
import { LIVE_LOG_PAGE } from './live-log-page.js';
import { BENCH_PAGE } from './bench-page.js';
import { ListDocumentDO } from './document-do.js';
import { handleOpenAIKeySetup, handleOpenAIKeyStatus } from '../openai-key-setup.js';
import { isMcpHostAllowed } from './mcp.js';
import { PROMPT_TARGETS } from '../../src/v2/domain/live-settings.js';
import { readBoundedJson } from './model.js';
import { safeError } from '../../src/v2/domain/contracts.js';
export { ListDocumentDO };
const json = (data,status=200) => Response.json(data,{status,headers:{'Cache-Control':'no-store'}});
/** Which object holds the document, and where it lives.
 *
 *  A Durable Object is created once and stays in the region it was created in; the hint is
 *  read only at creation. Moving the document therefore means a new object under a new name,
 *  which is why the name is configuration rather than a constant — flipping it back is the
 *  way back. */
const documentStub = (env) => getAgentByName(env.LIST_DOCUMENT, env.DOCUMENT_NAME || 'main',
  env.DO_LOCATION_HINT ? { locationHint: env.DO_LOCATION_HINT } : undefined);
const missing = () => new Response('Not found',{status:404});
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === '/health') return new Response('ok\n',{headers:{'Cache-Control':'no-store'}});
    if (url.pathname === '/live-log') return new Response(LIVE_LOG_PAGE,{headers:{'Content-Type':'text/html;charset=utf-8','Cache-Control':'no-store'}});
    if (url.pathname === '/bench') return new Response(BENCH_PAGE,{headers:{'Content-Type':'text/html;charset=utf-8','Cache-Control':'no-store'}});
    if (['/','/index.html','/list-manager.html'].includes(url.pathname)) return new Response(LIST_MANAGER_HTML,{headers:{'Content-Type':'text/html;charset=utf-8','Cache-Control':'no-store'}});
    if (request.method !== 'GET' && request.method !== 'OPTIONS') {
      const origin = request.headers.get('Origin');
      if (origin && origin !== url.origin) return json({error:'Origin not allowed'},403);
      if (url.pathname.startsWith('/api/v2/') && !request.headers.get('Content-Type')?.includes('application/json')) return json({error:'JSON required'},415);
    }
    try {
      if (!['/ws','/mcp','/reset'].includes(url.pathname) && !url.pathname.startsWith('/api/')) return missing();
      if (url.pathname === '/mcp' && !isMcpHostAllowed(request,env)) return missing();
      const stub = await documentStub(env);
      if (url.pathname === '/api/v2/input' && request.method === 'POST') return json(await stub.submit(await readBoundedJson(request,64000)),202);
      if (url.pathname === '/api/v2/document' && request.method === 'GET') return json(await stub.getDocument(Object.fromEntries(url.searchParams)));
      if (url.pathname === '/api/v2/updates' && request.method === 'GET') return json(await stub.follow(Number(url.searchParams.get('cursor')||0),url.searchParams.get('clientKey')||''));
      if (url.pathname === '/ws') return stub.fetch(request);
      if (url.pathname === '/mcp') return stub.mcpRequest(request);
      if (url.pathname === '/api/live/key/status') return handleOpenAIKeyStatus({configured:await stub.isOpenAIKeyConfigured(),setupAvailable:!await stub.isOpenAIKeyConfigured()});
      if (url.pathname === '/api/live/key') return handleOpenAIKeySetup(request,{configureKey:key=>stub.configureOpenAIApiKey(key)});
      if (url.pathname === '/api/live/session') {
        if (request.method === 'POST') return json(await stub.startLiveSession(await readBoundedJson(request,200000)),201);
        if (request.method === 'GET') return json(await stub.liveSessionStatus());
      }
      if (url.pathname === '/api/live/session/stop' && request.method === 'POST') return json({stopped:await stub.stopLiveSession()});
      if (url.pathname === '/api/live/settings') {
        if (request.method === 'GET') return json(await stub.readLiveSettings());
        if (request.method === 'PUT') {
          const body = await readBoundedJson(request,32000);
          if (typeof body.backendModel === 'string') return json(await stub.setLiveBackendModel(body.backendModel));
          if (typeof body.geminiModel === 'string') return json(await stub.setGeminiModel(body.geminiModel));
          if (typeof body.reasoningEffort === 'string') return json(await stub.setLiveReasoningEffort(body.reasoningEffort));
          if (!PROMPT_TARGETS.includes(body.target)) return json({error:'Unknown prompt target'},400);
          if (body.action === 'reset') return json(await stub.resetLivePrompt(body.target));
          if (body.action === 'restore') return json(await stub.restoreLivePrompt(body.target,String(body.at||'')));
          return json(await stub.writeLivePrompt(body.target,body));
        }
      }
      // Gemini Live: the browser holds the socket to Google, so what lives here is the key,
      // the session the token seals, the execution of every tool call, and the log.
      if (url.pathname === '/api/live/gemini/key/status') return json(await stub.geminiKeyStatus());
      if (url.pathname === '/api/live/gemini/key' && request.method === 'POST') {
        const body = await readBoundedJson(request,2048);
        // Pasted keys arrive with whitespace far more often than they arrive malformed.
        const apiKey = String(body.apiKey || '').trim();
        // AI Studio issues keys in two shapes: AIza… and AQ.… — the dot is part of the key.
        if (!/^[A-Za-z0-9._-]{20,200}$/.test(apiKey)) return json({error:'Ключ не похож на ключ Gemini API.'},400);
        const result = await stub.configureGeminiApiKey(apiKey);
        if (result.configured) return json(result);
        if (result.reason === 'already') return json({error:'Ключ Gemini уже настроен и работает.'},409);
        if (result.reason === 'env') return json({error:'Ключ Gemini задан секретом воркера; замените его там.'},409);
        return json({error:`Google отклонил ключ${result.status ? ` (${result.status})` : ''}: ${result.detail || 'нет подробностей'}`},400);
      }
      if (url.pathname === '/api/live/gemini/session') {
        if (request.method === 'POST') return json(await stub.startGeminiSession(),201);
        if (request.method === 'GET') return json(await stub.geminiSessionStatus());
      }
      if (url.pathname === '/api/live/gemini/session/stop' && request.method === 'POST') return json({stopped:await stub.stopGeminiSession()});
      if (url.pathname === '/api/live/gemini/tools' && request.method === 'POST') return json(await stub.runGeminiTools(await readBoundedJson(request,64000)));
      if (url.pathname === '/api/live/gemini/frames' && request.method === 'POST') { const body = await readBoundedJson(request,256000); return json(await stub.mirrorGeminiFrames(body.frames)); }
      if (url.pathname === '/api/live/settings/history' && request.method === 'GET') return json({history:await stub.livePromptHistory()});
      if (url.pathname === '/api/live/input' && request.method === 'GET') { const id=url.searchParams.get('response'); if(!id) return json({error:'response required'},400); return json(await stub.readBackendInput(id)); }
      // The log reads openly, by decision: it is the working record of what the framework
      // returned, and gating it behind a token got in the way of reading it.
      // Clearing is a POST, so the same-origin check above applies to it; reading is open.
      if (url.pathname === '/api/live/log/clear' && request.method === 'POST') return json(await stub.clearLiveLog());
      if (url.pathname === '/api/live/log/repair' && request.method === 'POST') return json(await stub.repairLiveLog());
      // Replays a written dialogue, so both halves are testable without a microphone:
      // layer=voice shows which task the voice layer picks and what it says aloud,
      // layer=backend drives one delegation. Runs the model only; no task is changed.
      if (url.pathname === '/api/live/simulate' && request.method === 'POST') {
        const body = await readBoundedJson(request,32000);
        const turns = Array.isArray(body.turns) ? body.turns : [{role:'user',text:String(body.text||'')}];
        if (!turns.length || !turns.every(turn => turn && typeof turn.text === 'string' && turn.text.trim())) return json({error:'turns required'},400);
        const dialogue = turns.map(turn => ({role:turn.role === 'assistant' ? 'assistant' : 'user', text:turn.text}));
        return json(body.layer === 'voice' ? await stub.simulateVoiceTurn(dialogue,{prompt:body.prompt === 'default' ? 'default' : 'current'}) : await stub.simulateDelegation(dialogue));
      }
      if (url.pathname.startsWith('/api/live/log') && request.method === 'GET') {
        if (url.pathname === '/api/live/log/sessions') return json(await stub.listLiveSessions(Number(url.searchParams.get('limit')||50)));
        if (url.pathname === '/api/live/log') return json(await stub.readLiveLog({sessionId:url.searchParams.get('session')||'',afterSeq:Number(url.searchParams.get('after')||0),limit:Number(url.searchParams.get('limit')||200)}));
      }
      // The same measurements, taken inside the worker. Run from the client they include the
      // client's own network; run here they are edge-to-object and back, which is the part
      // moving the object was meant to change. Cleans up the tasks it creates.
      if (url.pathname === '/api/v2/bench') {
        const mark = 'bench-' + Date.now().toString(36);
        const phases = [];
        const took = async (name, run) => { const at = Date.now(); const value = await run(); phases.push({name, ms: Date.now() - at}); return value; };
        const call = (calls) => stub.runGeminiTools({toolCall:{functionCalls:calls}});
        // The parts of one tool call, so the cost can be attributed rather than guessed.
        await took('пустой вызов объекта', () => stub.liveSessionStatus());
        const snapshot = await took('снимок для рассылки', () => stub.benchSnapshot());
        await took('чтение фронтира', () => stub.getTaskFrontier());
        await took('вызов getFrontier целиком', () => call([{id:mark+'f',name:'getFrontier',args:{}}]));
        await took('чтение документа', () => stub.getDocument({}));
        const one = await took('одна задача', () => call([{id:mark+'1',name:'addItem',args:{line1:mark+' 1'}}]));
        const two = await took('две задачи', () => call([{id:mark+'2',name:'addItem',args:{line1:mark+' 2'}},{id:mark+'3',name:'addItem',args:{line1:mark+' 3'}}]));
        const made = [...one.results, ...two.results].map(item => item.response?.target).filter(Boolean);
        await took('уборка', async () => { for (const id of made) await stub.applyTaskCommand({command:'deleteItem',actId:id,actType:'task',payload:{}}); });
        return json({edge: request.cf?.colo || 'unknown', object: env.DOCUMENT_NAME || 'main', snapshot, phases, created: made.length});
      }
      // How far the object is, measured inside the worker so the client's own network is not
      // part of the number. A Durable Object is pinned to one region; this is the only way to
      // see which side of the planet it ended up on.
      if (url.pathname === '/api/v2/where') {
        const colo = request.cf?.colo || 'unknown';
        const probes = (url.searchParams.get('names') || (env.DOCUMENT_NAME || 'main')).split(',').slice(0,6);
        const results = [];
        for (const entry of probes) {
          const [name, hint] = entry.split(':');
          const started = Date.now();
          try {
            const target = await getAgentByName(env.LIST_DOCUMENT, name, hint ? {locationHint:hint} : undefined);
            await target.liveSessionStatus();
            results.push({name, hint: hint || null, ms: Date.now() - started});
          } catch (error) { results.push({name, hint: hint || null, error: safeError(error).message}); }
        }
        return json({edge: colo, object: env.DOCUMENT_NAME || 'main', probes: results});
      }
      // Copies the document into another object, which is how it changes region. The source
      // is left untouched, so the move is undone by pointing DOCUMENT_NAME back.
      if (url.pathname === '/api/v2/relocate' && request.method === 'POST') {
        const body = await readBoundedJson(request,2048);
        const target = String(body.target || '').trim();
        const hint = String(body.locationHint || '').trim();
        if (!/^[a-z0-9][a-z0-9-]{0,40}$/.test(target)) return json({error:'Некорректное имя объекта'},400);
        if (target === (env.DOCUMENT_NAME || 'main')) return json({error:'Целевой объект совпадает с текущим'},400);
        const payload = await stub.exportEverything();
        const destination = await getAgentByName(env.LIST_DOCUMENT, target, hint ? {locationHint:hint} : undefined);
        const written = await destination.importEverything(payload);
        return json({from:env.DOCUMENT_NAME || 'main', target, locationHint:hint || null, exported:payload.counts, imported:written});
      }
      if (url.pathname === '/api/tasks/tree.json') return json({tasks:await stub.getTaskTree()});
      if (url.pathname === '/api/tasks/frontier.json') return json({frontier:await stub.getTaskFrontier()});
      if (url.pathname === '/api/tasks/tree.txt') return new Response(await stub.getTaskTitleTreeText(),{headers:{'Content-Type':'text/plain;charset=utf-8','Cache-Control':'no-store'}});
      if (url.pathname === '/api/tasks/item') { const id=url.searchParams.get('id');if(!id)return json({error:'id required'},400);const result=await stub.getTaskItem(id);return json(result,result.status==='found'?200:404); }
      if (url.pathname === '/reset' && request.method === 'POST' && env.TEST_RESET_TOKEN && request.headers.get('X-VoiceList-Test-Reset') === env.TEST_RESET_TOKEN) return json({state:await stub.reset()});
      return missing();
    } catch (error) { return json({error:safeError(error)},error.code==='NOT_FOUND'?404:error.code==='CONFLICT'?409:error.code?400:500); }
  }
};
