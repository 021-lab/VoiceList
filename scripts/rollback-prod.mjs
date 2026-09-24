/** Returns production to a version that is already uploaded.
 *
 *  Only the code moves. The Durable Object holding the task graph belongs to the worker, not
 *  to the version, so a rollback leaves the data exactly where it was — which also means a
 *  rollback past a storage change is a code-only remedy and the data keeps the newer shape.
 *
 *  Usage: npm run rollback:prod            — to the version that was live before this one
 *         npm run rollback:prod -- --list  — show what can be returned to
 *         npm run rollback:prod -- <id>    — to that exact version */
import { PROD_HOST, checkHealth, deployedVersions, describeVersion, versionIndex, wrangler } from './prod-lib.mjs';

const args = process.argv.slice(2).filter(arg => arg !== '--');
const stop = (message) => { console.error(`\n✗ ${message}`); process.exit(1); };

const { current, previous, history } = deployedVersions();
const index = versionIndex();

if (args.includes('--list') || args.includes('-l')) {
  console.log(`\nПрод: ${PROD_HOST}\nБыло на проде, от старого к новому:\n`);
  for (const id of history) console.log(`  ${id === current ? '→' : ' '} ${describeVersion(id, index)}`);
  console.log('\nОткат: npm run rollback:prod -- <version-id>');
  process.exit(0);
}

const requested = args.find(arg => /^[0-9a-f-]{36}$/i.test(arg));
const target = requested || previous;
if (!target) stop('Некуда откатываться: на проде была только одна версия. Список: npm run rollback:prod -- --list');
if (target === current) stop('Эта версия и так на проде.');
if (requested && !index.has(requested)) stop(`Версия ${requested} не найдена у этого воркера.`);

console.log(`\n▸ Сейчас:  ${describeVersion(current, index)}`);
console.log(`▸ Вернуть: ${describeVersion(target, index)}\n`);

wrangler(['versions', 'deploy', `${target}@100%`, '--yes']);

const health = await checkHealth();
console.log(`\n▸ Здоровье: ${health === 'ok' ? 'ok' : `НЕ ОТВЕЧАЕТ (${health ?? 'нет ответа'})`}`);
console.log(`\n✓ На проде версия ${describeVersion(target, index)}`);
console.log(`  Вернуться обратно: npm run rollback:prod -- ${current}`);

if (health !== 'ok') process.exit(2);
