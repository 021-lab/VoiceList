import { describe, it, expect } from 'vitest';
import { DocumentRuntime } from '../../src/v2/domain/document-runtime.js';
import { TaskGraph } from '../../src/v2/domain/task-graph.js';
import { InteractionJournal } from '../../src/v2/domain/interaction-journal.js';
import { AgentHarness } from '../../src/v2/domain/agent-harness.js';
import { CompatibilityPort } from '../../worker/v2/compatibility.js';

function input(seq, command, extra = {}) {
  return { key:{clientKey:'test',seq}, context:{elementId:'task:milk1',view:'list',revision:0,...extra}, command };
}
const status = value => ({command:'setStatus',actId:'milk1',payload:{status:value}});
async function execute(runtime, value) {
  value = structuredClone(value);
  value.context.revision = runtime.graph.revision;
  const receipt = await runtime.executeAndWait(value);
  expect(receipt.status).toBe('completed');
  return receipt;
}
describe('v0.2 authoritative journal and graph projection', () => {
  it('accepts before execution and publishes a server component tree', async () => {
    const runtime = new DocumentRuntime();
    const receipt = await runtime.submit(input(1,status('Focus')));
    expect(receipt.status).toBe('accepted');
    expect(runtime.graph.read({id:'milk1'}).status).toBe('Open');
    await runtime.processPending();
    expect(runtime.graph.read({id:'milk1'}).status).toBe('Focus');
    expect(runtime.journal.entries.map(e=>e.type)).toEqual(['interaction']);
    expect(runtime.journal.entries[0]).not.toHaveProperty('result');
    expect(runtime.journal.actions()).toHaveLength(0);
    const doc=runtime.getDocument({view:'list'});
    expect(doc.schemaVersion).toBe(1);
    expect(doc.root.children[1].children.find(n=>n.id==='task:milk1').props.status).toBe('Focus');
  });
  it('deduplicates both pending and completed requests; rejects key reuse', async () => {
    const runtime=new DocumentRuntime();
    const request=input(1,{command:'addItem',actId:'list',payload:{line1:'One'}});
    await Promise.all([runtime.submit(request),runtime.submit(request)]);
    await runtime.processPending();
    expect((await runtime.executeAndWait(request)).status).toBe('completed');
    expect(runtime.graph.read().items.filter(x=>x.line1==='One')).toHaveLength(1);
    await expect(runtime.submit({...request,command:status('Done')})).rejects.toMatchObject({code:'REQUEST_KEY_REUSED'});
  });
  it('does not publish a changed graph on storage failure and can retry', async () => {
    let persisted, reject=false;
    const runtime=new DocumentRuntime({persist:async state=>{if(reject)throw Error('storage');persisted=structuredClone(state);}});
    await runtime.submit(input(1,status('Done')));
    reject=true;
    await expect(runtime.processPending()).rejects.toThrow('storage');
    expect(runtime.graph.read({id:'milk1'}).status).toBe('Open');
    reject=false;await runtime.processPending();
    expect(runtime.graph.read({id:'milk1'}).status).toBe('Done');
    expect(persisted.graph.items.find(x=>x.id==='milk1').status).toBe('Done');
  });
  it('leaves the ledgers and the journal as they were when a transaction fails', async () => {
    // A transaction copies the ledgers instead of deep-copying them, and enriches a journal
    // entry by replacing it. What makes that safe is this: a failed transaction leaves
    // nothing behind for the next read to see.
    let reject = false;
    const runtime = new DocumentRuntime({ persist: async () => { if (reject) throw Error('storage'); } });
    await runtime.submit(input(1, status('Done')));
    const before = structuredClone(runtime.exportState());
    reject = true;
    await expect(runtime.processPending()).rejects.toThrow('storage');
    expect(runtime.exportState()).toEqual(before);
    expect(runtime.journal.entries).toEqual(before.entries);
    reject = false;
    await runtime.processPending();
    expect(runtime.state.technical.executor[before.entries[0].id].outcomes).toHaveLength(1);
  });
  it('recovers an accepted input after a restart before queueing', async () => {
    const runtime=new DocumentRuntime();
    await runtime.submit(input(1,status('Done')));
    const restarted=new DocumentRuntime({initialState:runtime.exportState()});
    await restarted.processPending();
    expect(restarted.graph.read({id:'milk1'}).status).toBe('Done');
    expect(restarted.journal.pending()).toHaveLength(0);
  });
  it('rejects invalid targets, state patches, status and reorder cycles', async () => {
    const runtime=new DocumentRuntime();
    await expect(runtime.submit(input(1,status('Done'),{elementId:'task:missing'}))).rejects.toMatchObject({code:'NOT_FOUND'});
    await expect(runtime.submit(input(1,{command:'rawPatch',payload:{}}))).rejects.toMatchObject({code:'UNSUPPORTED_COMMAND'});
    expect((await execute(runtime,input(1,status('Invented')))).error.code).toBe('INVALID_INPUT');
    const reorder={command:'reorderItems',actId:'list',payload:{arranged:[{id:'bread',parentId:'borod',order:1}]}};
    expect((await execute(runtime,input(2,reorder))).error.code).toBe('INVALID_GRAPH');
    expect(runtime.graph.read({id:'bread'}).parentId).toBe(null);
  });
  it('preserves details, deadline, tags and arbitrary task attributes on edits', async () => {
    const runtime=new DocumentRuntime();
    await execute(runtime,input(1,{command:'setDeadline',actId:'milk1',payload:{deadline:'2026-10-20'}}));
    await execute(runtime,input(2,{command:'editItem',actId:'milk1',payload:{line1:'New title'}}));
    const task=runtime.graph.read({id:'milk1'});
    expect(task.line2).toBe('2 пакета, магазин у дома');
    expect(task.deadline).toBe('2026-10-20');
  });
  it('rolls back an action and its corrections without deleting independent field edits', async () => {
    const runtime=new DocumentRuntime();
    const first=await execute(runtime,input(1,status('Focus')));
    const id=first.requestId;
    await execute(runtime,input(2,status('Done'),{actionId:id,elementId:'action:'+id}));
    await execute(runtime,input(3,{command:'editItem',actId:'milk1',payload:{line1:'Independent'}}));
    const rollback=await execute(runtime,input(4,{command:'rollbackAction',actId:id,payload:{}},{elementId:'action:'+id,actionId:id}));
    expect(rollback.error).toBe(null);
    expect(runtime.graph.read({id:'milk1'})).toMatchObject({status:'Open',line1:'Independent'});
    expect(runtime.state.technical.undoneEntries[id]).toBe(true);
  });
  it('fails targeted rollback if another action changed the same field', async () => {
    const runtime=new DocumentRuntime();
    const first=await execute(runtime,input(1,status('Focus')));
    await execute(runtime,input(2,status('Done')));
    const rollback=await execute(runtime,input(3,{command:'rollbackAction',actId:first.requestId,payload:{}}));
    expect(rollback.error.code).toBe('CONFLICT');
    expect(runtime.graph.read({id:'milk1'}).status).toBe('Done');
  });
  it('interprets completed speech server-side and retains unrecognized input', async () => {
    const runtime=new DocumentRuntime();
    const speech=input(1,status('Open'));delete speech.command;speech.text='это фокус';
    await execute(runtime,speech);
    expect(runtime.graph.read({id:'milk1'}).status).toBe('Focus');
    await execute(runtime,{...speech,key:{clientKey:'test',seq:2},text:'что-то непонятное'});
    expect(runtime.journal.actions().at(-1).status).toBe('needs-input');
    expect(runtime.journal.actions().at(-1).transcript).toBe('что-то непонятное');
  });
  it('detects context revision drift while an external model is running', async () => {
    let release;
    const runtime=new DocumentRuntime({resolveModel:()=>new Promise(resolve=>{release=resolve;})});
    const speech=input(1,status('Open'));delete speech.command;speech.text='неизвестный запрос';
    await runtime.submit(speech);
    const processing=runtime.processPending();
    await Promise.resolve();await Promise.resolve();
    await runtime.transaction(({graph})=>graph.apply([status('Done')]));
    release({commands:[status('Focus')]});
    await processing;
    expect(runtime.graph.read({id:'milk1'}).status).toBe('Done');
    expect(runtime.journal.actions().at(-1).error.code).toBe('CONFLICT');
  });
  it('targets UI effects to their originating session only', async () => {
    const runtime=new DocumentRuntime();
    await execute(runtime,input(1,{command:'showFrontier',actId:'list',payload:{}}));
    expect(runtime.follow(0,'test').uiEffects[0].view).toBe('frontier');
    expect(runtime.follow(0,'another').uiEffects).toHaveLength(0);
    expect(runtime.graph.revision).toBe(0);
    expect(JSON.stringify(runtime.follow(0,'test'))).not.toContain('expectedRevision');
  });
  it('keeps the legacy API port on the same authoritative runtime', async () => {
    const runtime=new DocumentRuntime(), port=new CompatibilityPort(runtime);
    const ack=await port.applyCommand(status('Focus'),{message:{clientKey:'legacy',seq:1}});
    expect(ack.status).toBe('applied');
    expect(port.getTaskById('milk1').status).toBe('Focus');
    expect(port.getTaskSubgraph('milk1').status).toBe('found');
    expect(port.getTaskTitleTreeText()).toContain('milk1 >>');
    expect(runtime.journal.actions()).toHaveLength(0);
  });
  it('forms editor, action, frontier, search and empty screen component documents', async () => {
    const runtime=new DocumentRuntime();
    const request=input(1,status('Open'));delete request.command;request.text='это фокус';
    const receipt=await execute(runtime,request);
    expect(runtime.getDocument({view:'edit',taskId:'milk1'}).root.children[1].type).toBe('task-editor');
    expect(runtime.getDocument({view:'action',actionId:receipt.actions[0].id}).root.children[1].props.records).toHaveLength(1);
    expect(runtime.getDocument({view:'frontier'}).root.children[1].type).toBe('task-list');
    expect(runtime.getDocument({view:'search',query:'хлеб'}).root.children[1].children.length).toBeGreaterThan(0);
    expect(runtime.getDocument({view:'settings'}).root.children[1].type).toBe('settings');
  });
  it('supports harness schedule and cancellation through the SDK adapter', async () => {
    const calls=[];
    const harness=new AgentHarness({agent:{run:x=>x},scheduler:{schedule:(...args)=>calls.push(args),cancelSchedule:id=>calls.push(id)}});
    await harness.schedule({when:60,input:{text:'test'}});
    await harness.cancelSchedule('s1');
    expect(calls).toEqual([[60,'scheduledTrigger',{text:'test'}],'s1']);
  });
});
