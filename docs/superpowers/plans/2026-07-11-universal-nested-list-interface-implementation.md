# Universal Nested List Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `list-manager.html` into a modular nested-list app with flat state, command interpreter, local store, action log, status actions, htmlpreview cache-busted deploy, and Playwright tests against the deployed page.

**Architecture:** Keep the current DOM gesture mechanics and current full-render strategy, but move all business changes behind `dispatchUserInput -> interpreter -> store -> renderer -> sync`. Introduce ES modules, snapshot plus action log persistence, flat `parentId/order` state, and deploy-first Playwright coverage.

**Tech Stack:** Static HTML, browser ES modules, Node.js tooling, Playwright, GitHub Actions, htmlpreview.github.io

---

### Task 1: Tooling And Preview Pipeline

**Files:**
- Create: `package.json`
- Create: `package-lock.json`
- Create: `playwright.config.js`
- Create: `scripts/prepare-preview.mjs`
- Create: `.github/workflows/deploy-test.yml`
- Modify: `list-manager.html`

- [ ] Add Node scripts for preview preparation and Playwright execution.
- [ ] Add a preview-prep script that writes a build hash and rewrites HTML asset references with `?v=<hash>`.
- [ ] Add a GitHub Actions workflow that computes the branch preview URL, verifies raw asset `200`s, and runs Playwright against htmlpreview.
- [ ] Verify locally with `npm install`, `node scripts/prepare-preview.mjs`, and `npx playwright test --list`.

### Task 2: Seed Data And Shared App State

**Files:**
- Modify: `list-data.js`
- Create: `src/list-seed.js`
- Create: `src/list-types.js`

- [ ] Write a failing test for seed-state shape and id generation expectations.
- [ ] Convert the current nested demo data into flat `snapshot.items` with `parentId`, `order`, `status`, `collapsed`, and `tags`.
- [ ] Remove `nextId` and add random 5-character id generation with collision retry.
- [ ] Verify the state shape test passes.

### Task 3: Store, Interpreter, And App Controller

**Files:**
- Create: `src/list-store.js`
- Create: `src/list-interpreter.js`
- Create: `src/list-app.js`
- Create: `tests/unit/list-store.test.js`
- Create: `tests/unit/list-interpreter.test.js`

- [ ] Write failing unit tests for `addItem`, `addChild`, `setStatus`, `toggleCollapse`, and action-log append behavior.
- [ ] Implement a localStorage-backed store for `{ snapshot, actionLog }`.
- [ ] Implement interpreter commands returning JSON Patch and action-log entries.
- [ ] Implement controller dispatch that loads state, applies interpreter output, persists, re-renders, and triggers sync.
- [ ] Verify unit tests pass.

### Task 4: Renderer And Action Log UI

**Files:**
- Create: `src/list-renderer.js`
- Modify: `list-manager.html`
- Modify: `list-manager.css`
- Create: `tests/unit/list-renderer.test.js`

- [ ] Write failing renderer tests for task rendering, nested rendering from flat state, status badge rendering, and action-log rendering.
- [ ] Keep current full-render list strategy while rebuilding visible nested order from flat `parentId/order` data.
- [ ] Add `data-act-id`, `data-act-type`, and command metadata to active DOM nodes.
- [ ] Add an action-log tab/panel and compact status badge UI.
- [ ] Remove PTT button, diagnostics panel, and related UI.
- [ ] Verify renderer tests pass.

### Task 5: UI Dispatch Layer And Gesture Preservation

**Files:**
- Create: `src/list-ui.js`
- Modify: `list-manager.html`
- Create: `tests/browser/list-ui-smoke.spec.js`

- [ ] Write a failing browser test that proves add task, add subtask, and set status travel through the page and update rendered output.
- [ ] Extract current gesture code into the UI module with minimal geometric changes.
- [ ] Replace direct state mutation calls with `dispatchUserInput(...)`.
- [ ] Keep current drag finalization flow, but emit a semantic move command instead of rebuilding the legacy tree directly.
- [ ] Verify the browser smoke test passes locally.

### Task 6: Deployed Playwright Coverage

**Files:**
- Create: `tests/e2e/list-manager-preview.spec.js`
- Modify: `playwright.config.js`
- Modify: `.github/workflows/deploy-test.yml`

- [ ] Write Playwright coverage for create task, create subtask, and status transitions on the preview URL.
- [ ] Clear localStorage per test and assert rendered DOM after each mutation.
- [ ] Assert the action log shows corresponding commands.
- [ ] Verify locally against a served preview-equivalent page and in CI against htmlpreview.

### Task 7: Final Integration And Docs Sync

**Files:**
- Modify: `docs/superpowers/specs/2026-07-11-universal-nested-list-interface-design.md`
- Modify: `docs/superpowers/specs/2026-07-11-universal-nested-list-interface-tests.md`
- Modify: `list-manager-docs.md`

- [ ] Update docs if implementation details diverge from the agreed spec.
- [ ] Run the full local verification set.
- [ ] Commit implementation in focused commits or one cohesive final commit if the branch remains tightly scoped.

### Verification Commands

- [ ] `npm install`
- [ ] `npm test`
- [ ] `npx playwright test`
- [ ] `node scripts/prepare-preview.mjs`
- [ ] `git status --short`

