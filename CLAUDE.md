# SearchMyData — Project Rules

## Before reporting work done
1. Verify the current page loads and shows expected content (fetch the live URL or validate locally).
2. End every message with a working link to the live page.

## Live page link format
Generate the link with these commands and include it at the end of every message:
```bash
BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT=$(git rev-parse --short HEAD)
echo "https://htmlpreview.github.io/?https://raw.githubusercontent.com/021-lab/searchmydata/${BRANCH}/list-manager.html?v=${COMMIT}"
```

## Development branch
- Feature branch: `claude/nested-list-structure-8hrDx`
- Remote: `021-lab/SearchMyData`
