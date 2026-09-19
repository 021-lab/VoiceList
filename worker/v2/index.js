import { getAgentByName } from 'agents';
import { LIST_MANAGER_HTML } from '../generated-html.js';
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
          if (!PROMPT_TARGETS.includes(body.target)) return json({error:'Unknown prompt target'},400);
          if (body.action === 'reset') return json(await stub.resetLivePrompt(body.target));
          if (body.action === 'restore') return json(await stub.restoreLivePrompt(body.target,String(body.at||'')));
          return json(await stub.writeLivePrompt(body.target,body));
        }
      }
      if (url.pathname === '/api/live/settings/history' && request.method === 'GET') return json({history:await stub.livePromptHistory()});
      // The log holds spoken transcripts, so reading it is closed by a token and fails shut
      // when none is configured.
      if (url.pathname.startsWith('/api/live/log') && request.method === 'GET') {
        if (!env.LIVE_LOG_TOKEN || request.headers.get('X-VoiceList-Log-Token') !== env.LIVE_LOG_TOKEN) return json({error:'Log token required'},403);
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
