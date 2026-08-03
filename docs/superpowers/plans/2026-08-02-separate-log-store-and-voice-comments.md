# Separate Log Store And Voice Comments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move log entries out of the task snapshot store, create each log record before applying task mutations, and add voice comments on log rows.

**Architecture:** Keep the task snapshot as the source of truth for tasks in `localStorage`, but persist log entries in a separate per-entry store keyed by log id. The app composes `{ snapshot, actionLog }` in memory for rendering, creates log entries before applying patches, and routes log create/update through a transport queue so remote sync can be added later without changing UI flow.

**Tech Stack:** Vanilla ES modules, `localStorage`, Playwright, Vitest

---

### Task 1: Split task snapshot storage from log entry storage

**Files:**
- Modify: `src/list-store.js`
- Create: `src/list-log-store.js`
- Test: `tests/unit/list-store.test.js`
- Test: `tests/unit/list-log-store.test.js`

- [ ] Add failing unit tests for task-only snapshot persistence and per-entry log persistence.
- [ ] Implement `createLogStore()` with separate entry keys plus ordered id index.
- [ ] Update `createStore()` to persist only task snapshot state.
- [ ] Add legacy `actionLog` migration hook for old combined state.

### Task 2: Create log entries before task mutation application

**Files:**
- Modify: `src/list-interpreter.js`
- Modify: `src/list-app.js`
- Modify: `src/list-preview-entry.js`
- Test: `tests/unit/list-interpreter.test.js`
- Test: `tests/unit/list-app.test.js`

- [ ] Add failing tests proving log entry draft includes transcript, command, patch, and is persisted before snapshot mutation.
- [ ] Change interpreter output to return a log draft instead of mutating log storage directly.
- [ ] Update app flow to persist the log entry first, apply the snapshot patch second, then re-render composite state.

### Task 3: Add transport-backed log entry status updates and comment updates

**Files:**
- Modify: `src/list-sync.js`
- Modify: `src/list-renderer.js`
- Test: `tests/unit/list-sync.test.js`
- Test: `tests/unit/list-renderer.test.js`

- [ ] Add failing tests for log create/update transport queue behavior.
- [ ] Update sync queue to operate on log entry create/update rather than task snapshot action-log mutation.
- [ ] Render transcript, interpreted command, and comments from separate log entries.

### Task 4: Add voice comments on log rows

**Files:**
- Modify: `src/list-ui.js`
- Modify: `list-manager.css`
- Test: `tests/unit/list-ui.test.js`
- Test: `tests/e2e/list-manager-preview.spec.js`

- [ ] Add failing tests for long-press voice comment on a log row.
- [ ] Implement log-row voice capture with press / speak / release comment append behavior.
- [ ] Reuse overlay styling without command-candidate selection for log comments.

### Task 5: Rebuild preview and verify deployment behavior

**Files:**
- Modify: `list-manager.html`

- [ ] Run `npm test`
- [ ] Run `npm run test:voice`
- [ ] Run `npm run test:e2e`
- [ ] Run `npm run prepare-preview`
- [ ] Verify `https://viewterminus.smileme.ai/`
