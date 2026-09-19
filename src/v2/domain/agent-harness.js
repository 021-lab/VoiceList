export const HARNESS_VERSIONS = Object.freeze({ contextBuilder: 'context-v0.2.1', prompt: 'prompt-v0.2.1', parser: 'parser-v0.2.1' });

export class AgentHarness {
  constructor({ agent, scheduler }) { this.agent = agent; this.scheduler = scheduler; }
  buildContext(trigger) { return this.agent.buildContext(trigger); }
  invoke(modelContext) { return this.agent.invoke(modelContext); }
  parse(raw, modelContext) { return this.agent.parse(raw, modelContext); }
  schedule(rule) { return this.scheduler.schedule(rule.when, 'scheduledTrigger', rule.input); }
  cancelSchedule(id) { return this.scheduler.cancelSchedule(id); }
}
