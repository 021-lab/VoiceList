import { build } from 'esbuild';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');
const outputPath = path.join(repoRoot, 'list-manager.html');
const templatePath = path.join(repoRoot, 'list-manager.template.html');
const cssPath = path.join(repoRoot, 'list-manager.css');
const entryPath = path.join(repoRoot, 'src/list-preview-entry.js');

function generateBuildHash() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function escapeInlineScript(value) {
  return value.replaceAll('</script>', '<\\/script>');
}

async function readExistingBuildHash() {
  try {
    const currentHtml = await readFile(outputPath, 'utf8');
    const match = currentHtml.match(/<meta name="preview-build-hash" content="([^"]+)">/);
    return match?.[1] || null;
  } catch {
    return null;
  }
}

const buildHash = process.env.PREVIEW_BUILD_HASH || await readExistingBuildHash() || generateBuildHash();
const [template, css] = await Promise.all([
  readFile(templatePath, 'utf8'),
  readFile(cssPath, 'utf8')
]);

const bundle = await build({
  entryPoints: [entryPath],
  bundle: true,
  format: 'iife',
  platform: 'browser',
  target: ['es2022'],
  write: false,
  define: {
    __PREVIEW_BUILD_HASH__: JSON.stringify(buildHash)
  }
});

const inlineJs = bundle.outputFiles[0].text;
const html = template
  .replaceAll('__PREVIEW_BUILD_HASH__', buildHash)
  .replace('__INLINE_CSS__', css.trim())
  .replace('__INLINE_JS__', escapeInlineScript(inlineJs.trim()));

await writeFile(outputPath, `${html}\n`);
process.stdout.write(`${buildHash}\n`);
