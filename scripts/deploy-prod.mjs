/** Publishes the current tree to production as a new version and sends all traffic to it.
 *
 *  Upload and activation are two steps on purpose. The uploaded version keeps its own id and
 *  stays on Cloudflare's side, so returning to yesterday's build is a single activation of an
 *  id that already exists — no rebuild, no checkout, and it works when the repository has
 *  moved on. Each version carries the commit it was built from as its tag.
 *
 *  Usage: npm run deploy:prod [-- --allow-dirty] [-- --skip-tests] */
import { PROD_CONFIG, PROD_HOST, checkHealth, deployedVersions, describeVersion, git, run, wrangler } from './prod-lib.mjs';

const flags = new Set(process.argv.slice(2));
const stop = (message) => { console.error(`\n✗ ${message}`); process.exit(1); };

const status = git(['status', '--porcelain']);
if (status && !flags.has('--allow-dirty')) {
  stop(`Рабочее дерево не чистое — непонятно, что именно уедет на прод:\n${status}\nЗакоммитьте или запустите с --allow-dirty.`);
}

const sha = git(['rev-parse', '--short', 'HEAD']);
const branch = git(['rev-parse', '--abbrev-ref', 'HEAD']);
const subject = git(['log', '-1', '--pretty=%s']).slice(0, 100);
const tag = status ? `${sha}-dirty` : sha;

console.log(`\n▸ Прод: ${PROD_HOST} (${PROD_CONFIG})`);
console.log(`▸ Сборка: ${branch} ${tag} — ${subject}\n`);

if (!flags.has('--skip-tests')) {
  console.log('▸ Тесты');
  try { run('npx', ['vitest', 'run']); } catch { stop('Тесты не прошли — выката нет.'); }
}

console.log('\n▸ Сборка клиента');
run('npm', ['run', 'build:v2']);

console.log('\n▸ Загрузка версии');
const before = deployedVersions();
const uploaded = wrangler(['versions', 'upload', '--tag', tag, '--message', `${branch}: ${subject}`], { capture: true });
process.stdout.write(uploaded);
const versionId = uploaded.match(/Worker Version ID:\s*([0-9a-f-]{36})/i)?.[1];
if (!versionId) stop('Wrangler не вернул идентификатор версии; ничего не активировано.');

console.log('\n▸ Перевод трафика на новую версию');
wrangler(['versions', 'deploy', `${versionId}@100%`, '--yes']);

const health = await checkHealth();
console.log(`\n▸ Здоровье: ${health === 'ok' ? 'ok' : `НЕ ОТВЕЧАЕТ (${health ?? 'нет ответа'})`}`);

console.log(`\n✓ На проде версия ${describeVersion(versionId)}`);
if (before.current) console.log(`  Предыдущая: ${describeVersion(before.current)}`);
console.log(`  Откат: npm run rollback:prod${before.current ? '' : ' (откатываться пока некуда)'}`);

if (health !== 'ok') process.exit(2);
