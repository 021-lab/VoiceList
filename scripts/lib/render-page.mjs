/** Assembles the single-file page: template plus inlined stylesheet and bundle. */

export function escapeInlineScript(value) {
  return value.replaceAll('</script>', '<\\/script>');
}

/** Every substitution takes a function rather than a string.
 *
 *  A string replacement expands $&, $` and $' as patterns, so a bundle that happens to
 *  contain one — a template literal ending in $ is enough, and a validation library is full
 *  of them — splices the surrounding HTML into the middle of the script. The page then dies
 *  at parse time with nothing in the build output to say why. */
export function renderPage({ template, css, js, buildHash }) {
  return template
    .replaceAll('__PREVIEW_BUILD_HASH__', () => buildHash)
    .replace('__INLINE_CSS__', () => css.trim())
    .replace('__INLINE_JS__', () => escapeInlineScript(js.trim()));
}
