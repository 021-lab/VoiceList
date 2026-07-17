# VoiceList — Project Rules

## Before reporting work done
1. Verify the current page loads and shows expected content (fetch the live URL or validate locally).
2. End every message with a working link to the live page.

## Live page link format
Generate the link with these commands and include it at the end of every message:
```bash
SHA=$(git rev-parse HEAD)
ORIGIN=$(git remote get-url origin)
OWNER_REPO=$(printf '%s\n' "$ORIGIN" | sed -E 's#^git@github.com:##; s#^https://github.com/##; s#\.git$##')
echo "https://htmlpreview.github.io/?https://raw.githubusercontent.com/${OWNER_REPO}/${SHA}/list-manager.html"
```

## Development branch
- Work in `codex-` branches.
- Keep the repo fork-friendly: avoid hardcoded owner/repo names in preview or CI paths.
