import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
const base = process.env.V02_BASE_URL || 'http://127.0.0.1:4511';
const clientKey = 'integration:' + randomUUID(); let seq = 0;
const request = async (path, options) => {
  const response = await fetch(base + path, options); assert.equal(response.ok, true, `${path}: ${response.status}`);
  return response.json();
};
const doc = () => request('/api/v2/document');
const initial = await doc(); assert.equal(initial.schemaVersion, 1);
const post = input => request('/api/v2/input',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(input)});
async function run(command, context={}, text) {
  const input={key:{clientKey,seq:++seq},context:{elementId:'app',view:'list',revision:(await doc()).revision,...context},...(text?{text}:{command})};
  let receipt=await post(input);
  for(let i=0;receipt.status==='accepted'&&i<100;i++) {await new Promise(r=>setTimeout(r,100));receipt=await post(input);}
  assert.equal(receipt.status,'completed'); assert.equal(receipt.error,null,JSON.stringify(receipt.error));
  return {input,receipt};
}
const title='V02 integration '+clientKey;
const created=await run({command:'addItem',actId:'list',payload:{line1:title,line2:'durable test'}});
const id=created.receipt.actions[0].target; assert.ok(id);
const duplicate=await post(created.input); assert.equal(duplicate.requestId,created.receipt.requestId);
const nodes=()=>doc().then(d=>d.root.children[1].children);
assert.equal((await nodes()).filter(n=>n.props.line1===title).length,1);
const focused=await run(null,{elementId:'task:'+id},'это фокус');
assert.equal((await nodes()).find(n=>n.props.taskId===id).props.status,'Focus');
const actionId=focused.receipt.actions[0].id;
await run(null,{elementId:'action:'+actionId,actionId,view:'action'},'это сделано');
assert.equal((await nodes()).find(n=>n.props.taskId===id).props.status,'Done');
const details=await request('/api/v2/document?view=action&actionId='+actionId);
assert.equal(details.root.children[1].props.messages.length,4);
await run({command:'rollbackAction',actId:actionId,payload:{}},{elementId:'action:'+actionId,actionId,view:'action'});
assert.equal((await nodes()).find(n=>n.props.taskId===id).props.status,'Open');
const events=await request('/api/v2/updates?cursor='+initial.cursor+'&clientKey='+encodeURIComponent(clientKey));
assert.ok(events.actions.some(a=>a.id===actionId&&a.rolledBack));
const invalid=await fetch(base+'/api/v2/input',{method:'POST',headers:{'Content-Type':'application/json','Origin':'https://untrusted.invalid'},body:JSON.stringify(created.input)});
assert.equal(invalid.status,403);
const status=await request('/api/realtime/key/status');assert.equal(typeof status.configured,'boolean');assert.equal('apiKey' in status,false);
const socket=new WebSocket(base.replace(/^http/,'ws')+'/ws');
await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('WebSocket timeout')),5000);socket.onmessage=event=>{const value=JSON.parse(event.data);if(value.type==='state'){clearTimeout(timer);assert.ok(value.state.content.snapshot.items.find(x=>x.id===id));resolve();}};socket.onerror=reject;});socket.close();
if(new URL(base).hostname==='127.0.0.1') {
  const mcp=await request('/mcp',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({jsonrpc:'2.0',id:1,method:'tools/list',params:{}})});
  assert.ok(mcp.result.tools.some(t=>t.name==='voicelist_get_task_tree'));
}
await run({command:'deleteItem',actId:id,payload:{}},{elementId:'task:'+id});
assert.equal((await nodes()).some(n=>n.props.taskId===id),false);
console.log('PASS Workers runtime: persisted create, dedup, speech, correction chain, rollback, details, events, origin gate, key status, WebSocket, MCP, cleanup');
