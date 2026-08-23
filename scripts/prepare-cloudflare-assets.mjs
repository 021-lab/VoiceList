import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');
const html = await readFile(path.join(repoRoot, 'list-manager.html'), 'utf8');

await writeFile(
  path.join(repoRoot, 'worker/generated-html.js'),
  `export const LIST_MANAGER_HTML = ${JSON.stringify(html)};\n`
);
