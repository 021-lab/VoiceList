import { mkdir, readFile, writeFile } from 'node:fs/promises';

const html = await readFile(new URL('../list-manager.html', import.meta.url), 'utf8');
const outputDir = new URL('../worker/', import.meta.url);

await mkdir(outputDir, { recursive: true });
await writeFile(
  new URL('./generated-list-manager.js', outputDir),
  `export const LIST_MANAGER_HTML = ${JSON.stringify(html)};\n`
);
