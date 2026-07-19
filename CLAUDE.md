# VoiceList — Project Rules

## Ветка и точка входа для интеграции с taosmd

Если ты в ветке `taosmd-backend` — твоя задача описана здесь, читать в этом порядке:
1. `docs/CODER_TASK_sync-module.md` — что делать (модуль синхронизации localStorage ↔ taosmd);
2. `docs/INTEGRATION_taosmd.md` — архитектура доступа (Cloudflare Pages + Worker-proxy + Tunnel) и правила;
3. Канон спецификаций бэкенда (контракт API, жизненный цикл задач): https://github.com/021-lab/TaOS/tree/feat/user-statuses/docs/specs — начинать с `00-INDEX-focus-harness.md`.

Для интеграционных работ работай в `taosmd-backend` (или ветках от неё); правило `codex-*` ниже относится к остальной разработке фронта.

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
- Work in `codex-` branches (кроме интеграции с taosmd — см. выше).
- Keep the repo fork-friendly: avoid hardcoded owner/repo names in preview or CI paths.
