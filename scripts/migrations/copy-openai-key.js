import { getAgentByName } from 'agents';

/** One-time, route-less, expiring job. Never log, return, or locally serialize the key. */
export default {
  async scheduled(_event, env) {
    if (Date.now() > Date.parse(env.EXPIRES_AT)) return;
    const target = await getAgentByName(env.TARGET, 'main');
    if (await target.isOpenAIKeyConfigured()) return;
    const source = env.SOURCE.getByName('main');
    const key = await source.getOpenAIApiKey();
    if (typeof key !== 'string' || !key.startsWith('sk-')) throw new Error('Source key unavailable');
    if (!await target.configureOpenAIApiKey(key)) throw new Error('Target key not configured');
  },
  fetch() { return new Response('Not found', { status: 404 }); }
};
