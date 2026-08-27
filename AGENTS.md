# Repository Guidelines

## Project Structure & Module Organization

VoiceList is a browser task manager backed by a Cloudflare Worker and a single Durable Object. Browser modules live in `src/`; `list-preview-entry.js` wires the UI, sync client, and Realtime voice button. `worker/index.js` exposes HTTP and WebSocket routes, while `worker/list-document-core.js` owns persisted tasks and action-log entries. `list-manager.html` and `worker/generated-html.js` are generated assets: edit their source modules, then run `npm run build:cloudflare`.

## Build, Test, and Development Commands

- `npm test` runs the Vitest unit suite.
- `npm run test:voice` runs command-resolution and voice interaction checks.
- `npm run test:e2e` runs Playwright; `npm run test:e2e -- --grep "text"` runs one scenario.
- `npm run build:cloudflare` rebuilds the preview and Worker HTML assets.
- `npm run deploy:cloudflare` deploys production. Issue worktrees use only their local `.wrangler.dev.jsonc` and `vlist-dev.smileme.ai` until merged.

## Bug Reports from the Action Log

When the user says “в журнале баг” or asks to fix a bug in the journal, inspect comments attached to action-log entries, not the action labels, transcripts, or generic "Нераспознано" rows. A comment on an entry is the authoritative bug report. Read its exact text, identify the entry it annotates, and derive expected versus actual behavior from that comment before changing code. Do not infer a bug from an unannotated log row.

Fix the reported bug through the normal reversible workflow: create an isolated worktree and branch from current `main`, preserve unrelated changes, add a regression test, rebuild generated assets when source changes, run relevant tests, deploy to `vlist-dev.smileme.ai`, then commit, push, and merge with a merge commit. Deploy production only from updated `main`. Record the merge SHA; rollback uses `git revert -m 1 <merge-commit>` followed by a production deploy.

## Testing and Commit Guidelines

For gesture or voice changes, verify the exact user gesture and persisted behavior after rerender or reload; visible DOM alone is insufficient. Keep OpenAI keys server-side. Commit messages follow `fix:`, `feat:`, `perf:`, `docs:`, and `merge:` prefixes. Stage only task-related files and never merge or deploy from a dirty stable worktree.
