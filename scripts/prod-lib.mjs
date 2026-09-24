/** Shared plumbing for the production scripts.
 *
 *  Production is a worker of its own (wrangler.prod.jsonc). The config name is not a
 *  parameter anywhere: a flag that can point these scripts at another worker is the one
 *  mistake that cannot be rolled back, and v1 next door is deployed by someone else. */
import { execFileSync } from 'node:child_process';

export const PROD_CONFIG = 'wrangler.prod.jsonc';
export const PROD_HOST = 'vlist-v02-prod.smileme.ai';

export function run(command, args, { capture = false } = {}) {
  return execFileSync(command, args, {
    encoding: 'utf8',
    stdio: capture ? ['inherit', 'pipe', 'inherit'] : 'inherit',
    env: process.env
  });
}

export const wrangler = (args, options) => run('npx', ['wrangler', ...args, '--config', PROD_CONFIG], options);

/** Wrangler prints a proxy warning before the payload, so the JSON is found rather than parsed
 *  from the first byte. */
export function wranglerJson(args) {
  const output = wrangler([...args, '--json'], { capture: true });
  const start = output.search(/[[{]/);
  if (start < 0) throw new Error(`Не удалось прочитать ответ wrangler: ${output.slice(0, 200)}`);
  return JSON.parse(output.slice(start));
}

export function git(args) {
  return run('git', args, { capture: true }).trim();
}

/** What is live now, and what was live before it. A deployment can hold several versions at
 *  once during a gradual rollout; these scripts always deploy one at 100%, so the first entry
 *  of each deployment is the whole of it. */
export function deployedVersions() {
  const deployments = wranglerJson(['deployments', 'list']);
  const ordered = [...deployments].sort((a, b) => new Date(a.created_on) - new Date(b.created_on));
  const ids = [];
  for (const deployment of ordered) {
    const id = deployment.versions?.[0]?.version_id;
    if (id && ids.at(-1) !== id) ids.push(id);
  }
  return { current: ids.at(-1) || null, previous: ids.at(-2) || null, history: ids };
}

export function versionIndex() {
  const versions = wranglerJson(['versions', 'list']);
  return new Map(versions.map(version => [version.id, {
    id: version.id,
    number: version.number,
    createdOn: version.metadata?.created_on || '',
    tag: version.annotations?.['workers/tag'] || '',
    message: version.annotations?.['workers/message'] || ''
  }]));
}

export function describeVersion(id, index = versionIndex()) {
  const version = index.get(id);
  if (!version) return id;
  const when = version.createdOn ? version.createdOn.slice(0, 19).replace('T', ' ') : '';
  return [`#${version.number}`, id, when, version.tag, version.message].filter(Boolean).join('  ');
}

/** The deploy is not finished when wrangler returns: the route has to answer. */
export async function checkHealth(attempts = 10) {
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      const response = await fetch(`https://${PROD_HOST}/health`, { headers: { 'Cache-Control': 'no-cache' } });
      if (response.ok) return (await response.text()).trim();
    } catch { /* the custom domain can take a moment on first publish */ }
    await new Promise(resolve => setTimeout(resolve, 3_000));
  }
  return null;
}
