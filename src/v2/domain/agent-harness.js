export class AgentHarness {
  constructor({ agent, scheduler }) { this.agent = agent; this.scheduler = scheduler; }
  handle(trigger) { return this.agent.run(trigger); }
  schedule(rule) { return this.scheduler.schedule(rule.when, 'scheduledTrigger', rule.input); }
  cancelSchedule(id) { return this.scheduler.cancelSchedule(id); }
}
