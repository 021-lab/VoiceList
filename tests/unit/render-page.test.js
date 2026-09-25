import { describe, expect, it } from 'vitest';
import { escapeInlineScript, renderPage } from '../../scripts/lib/render-page.mjs';

const template = '<html><head><style>__INLINE_CSS__</style></head><body data-build="__PREVIEW_BUILD_HASH__">' +
  '<button id="go">go</button><script>__INLINE_JS__</script></body></html>';

describe('assembling the page', () => {
  it('inlines the bundle and the stylesheet and stamps the build', () => {
    const html = renderPage({ template, css: ' body {} ', js: ' console.log(1); ', buildHash: 'v02-abc' });
    expect(html).toContain('<script>console.log(1);</script>');
    expect(html).toContain('body {}');
    expect(html).toContain('data-build="v02-abc"');
  });

  /** This is the bug that broke the page once: zod builds its regexes from template literals
   *  that end in `$`, so the bundle carried `$` immediately before a backtick. Passed to
   *  String.replace as a string, that is the "everything before the match" pattern, and the
   *  whole document head was spliced into the middle of the script sixteen times over. */
  it('does not expand a $-sequence in the bundle into the surrounding html', () => {
    const js = 'const uuid = new RegExp(`^[0-9a-f]{8}$`); const all = "$&"; const before = "$\'";';
    const html = renderPage({ template, css: '', js, buildHash: 'v02-abc' });

    expect(html).toContain(js);
    expect(html.match(/<script>/g)).toHaveLength(1);
    expect(html.match(/<\/script>/g)).toHaveLength(1);
    expect(html).not.toContain('<button id="go">go</button><html>');
  });

  it('keeps a closing script tag inside the bundle from ending the script', () => {
    const html = renderPage({ template, css: '', js: 'const s = "</script>";', buildHash: 'v02-abc' });
    expect(html.match(/<\/script>/g)).toHaveLength(1);
    expect(html).toContain('<\\/script>');
  });

  it('escapes every closing tag, not only the first', () => {
    expect(escapeInlineScript('a</script>b</script>c')).toBe('a<\\/script>b<\\/script>c');
  });
});
