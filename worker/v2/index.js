import { getAgentByName } from 'agents';
import { LIST_MANAGER_HTML } from '../generated-html.js';
import { ListDocumentDO } from './document-do.js';
import { handleOpenAIKeySetup, handleOpenAIKeyStatus } from '../openai-key-setup.js';
import { getDefaultRealtimeSystemPrompt, handleOpenAIRealtimeSession } from '../openai-realtime.js';
import { isMcpHostAllowed } from './mcp.js';
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
      if (url.pathname === '/api/realtime/key/status') return handleOpenAIKeyStatus({configured:await stub.isOpenAIKeyConfigured(),setupAvailable:!await stub.isOpenAIKeyConfigured()});
      if (url.pathname === '/api/realtime/key') return handleOpenAIKeySetup(request,{configureKey:key=>stub.configureOpenAIApiKey(key)});
      if (url.pathname === '/api/realtime/prompt') {
        if (request.method === 'GET') return json({prompt:await stub.getOpenAISystemPrompt() || getDefaultRealtimeSystemPrompt()});
        if (request.method === 'POST') {const body=await readBoundedJson(request,16000);if(typeof body.prompt!=='string')return json({error:'Invalid prompt'},400);await stub.configureOpenAISystemPrompt(body.prompt);return json({configured:true});}
      }
      if (url.pathname === '/api/realtime/session') return handleOpenAIRealtimeSession(request,env,{apiKey:env.OPENAI_API_KEY||await stub.getOpenAIApiKey(),systemPrompt:await stub.getOpenAISystemPrompt()});
      if (url.pathname === '/api/realtime/diagnostics' && request.method === 'POST') {await stub.recordRealtimeDiagnostics(await readBoundedJson(request,4096));return json({recorded:true},202);}
      if (url.pathname === '/api/realtime/diagnostics' && request.method === 'GET' && env.REALTIME_DIAGNOSTICS_TOKEN && request.headers.get('X-VoiceList-Diagnostics-Token') === env.REALTIME_DIAGNOSTICS_TOKEN) return json({entries:await stub.getRealtimeDiagnostics()});
      if (url.pathname === '/api/tasks/tree.json') return json({tasks:await stub.getTaskTree()});
      if (url.pathname === '/api/tasks/frontier.json') return json({frontier:await stub.getTaskFrontier()});
      if (url.pathname === '/api/tasks/tree.txt') return new Response(await stub.getTaskTitleTreeText(),{headers:{'Content-Type':'text/plain;charset=utf-8','Cache-Control':'no-store'}});
      if (url.pathname === '/api/tasks/item') { const id=url.searchParams.get('id');if(!id)return json({error:'id required'},400);const result=await stub.getTaskItem(id);return json(result,result.status==='found'?200:404); }
      if (url.pathname === '/reset' && request.method === 'POST' && env.TEST_RESET_TOKEN && request.headers.get('X-VoiceList-Test-Reset') === env.TEST_RESET_TOKEN) return json({state:await stub.reset()});
      return missing();
    } catch (error) { return json({error:safeError(error)},error.code==='NOT_FOUND'?404:error.code==='CONFLICT'?409:error.code?400:500); }
  }
};
