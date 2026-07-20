# VoiceList — Project Rules

## Ветка и точка входа для интеграции с taosmd

Если ты в ветке `taosmd-backend` — твоя задача описана здесь, читать в этом порядке:
1. `docs/CODER_TASK_storage-module.md` — что делать (storage-модуль: localStorage + write-through в taosmd `/tasks`, bootstrap из бэкенда, удаление `list-data.js`);
2. `docs/INTEGRATION_taosmd.md` — архитектура доступа (Cloudflare Pages + Worker-proxy + Tunnel), подмножество API и правила;
3. Канон спецификаций бэкенда: https://github.com/021-lab/TaOS/tree/codex-voicelist-task-provenance/docs/specs — начинать с `TAOSMD_API_CONTRACT.md`.

Для интеграционных работ работай в `taosmd-backend` (или ветках от неё); правило `codex-*` ниже относится к остальной разработке фронта.

## Before reporting work done
1. Verify the current page loads and shows expected content (fetch the live URL or validate locally).
2. End every message with a working link to the live page.

## Live page link format
For taosmd integration, report the deployed Cloudflare Worker URL after `npm run build:worker` and Worker deployment. Do not use htmlpreview for this branch.

## Development branch
- Work in `codex-` branches (кроме интеграции с taosmd — см. выше).
- Keep the repo fork-friendly: avoid hardcoded owner/repo names in preview or CI paths.
