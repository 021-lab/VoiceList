import { getAgentByName } from 'agents';
import { LIST_MANAGER_HTML } from '../generated-html.js';
import { LIVE_LOG_PAGE } from './live-log-page.js';
import { ListDocumentDO } from './document-do.js';
import { handleOpenAIKeySetup, handleOpenAIKeyStatus } from '../openai-key-setup.js';
import { isMcpHostAllowed } from './mcp.js';
import { PROMPT_TARGETS } from '../../src/v2/domain/live-settings.js';
import { readBoundedJson } from './model.js';
import { safeError } from '../../src/v2/domain/contracts.js';
export { ListDocumentDO };
const json = (data,status=200) => Response.json(data,{status,headers:{'Cache-Control':'no-store'}});
const missing = () => new Response('Not found',{status:404});
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === '/health') return new Response('ok\n',{headers:{'Cache-Control':'no-store'}});
    if (url.pathname === '/live-log') return new Response(LIVE_LOG_PAGE,{headers:{'Content-Type':'text/html;charset=utf-8','Cache-Control':'no-store'}});
    if (['/','/index.html','/list-manager.html'].includes(url.pathname)) return new Response(LIST_MANAGER_HTML,{headers:{'Content-Type':'text/html;charset=utf-8','Cache-Control':'no-store'}});
    if (request.method !== 'GET' && request.method !== 'OPTIONS') {
      const origin = request.headers.get('Origin');
      if (origin && origin !== url.origin) return json({error:'Origin not allowed'},403);
      if (url.pathname.startsWith('/api/v2/') && !request.headers.get('Content-Type')?.includes('application/json')) return json({error:'JSON required'},415);
    }
    try {
      if (!['/ws','/mcp','/reset'].includes(url.pathname) && !url.pathname.startsWith('/api/')) return missing();
      if (url.pathname === '/mcp' && !isMcpHostAllowed(request,env)) return missing();
      const stub = await getAgentByName(env.LIST_DOCUMENT,'main');
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
        if (!/^[A-Za-z0-9_-]{20,128}$/.test(apiKey)) return json({error:'Ключ не похож на ключ Gemini API: ожидаются 20–128 символов из латиницы, цифр, дефиса и подчёркивания.'},400);
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
      if (url.pathname === '/api/tasks/tree.json') return json({tasks:await stub.getTaskTree()});
      if (url.pathname === '/api/tasks/frontier.json') return json({frontier:await stub.getTaskFrontier()});
      if (url.pathname === '/api/tasks/tree.txt') return new Response(await stub.getTaskTitleTreeText(),{headers:{'Content-Type':'text/plain;charset=utf-8','Cache-Control':'no-store'}});
      if (url.pathname === '/api/tasks/item') { const id=url.searchParams.get('id');if(!id)return json({error:'id required'},400);const result=await stub.getTaskItem(id);return json(result,result.status==='found'?200:404); }
      if (url.pathname === '/reset' && request.method === 'POST' && env.TEST_RESET_TOKEN && request.headers.get('X-VoiceList-Test-Reset') === env.TEST_RESET_TOKEN) return json({state:await stub.reset()});
      return missing();
    } catch (error) { return json({error:safeError(error)},error.code==='NOT_FOUND'?404:error.code==='CONFLICT'?409:error.code?400:500); }
  }
};
