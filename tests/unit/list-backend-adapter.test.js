import { describe, expect, test, vi } from 'vitest';

import {
  createBackendAdapter,
  createBackupTargets,
  defaultTarget,
  describeFirestoreAccess,
  firestoreCollections
} from '../../src/list-backend-adapter.js';

const sampleState = {
  snapshot: {
    items: [
      {
        id: 'abc12',
        parentId: null,
        order: 10,
        status: 'Open',
        line1: 'Task',
        line2: '',
        collapsed: false,
        tags: []
      }
    ]
  },
  actionLog: []
};

describe('list backend adapter', () => {
  test('describes current and backup collection access', () => {
    const access = describeFirestoreAccess(defaultTarget, new Date('2026-07-12T07:08:00.000Z'));

    expect(access.current.path).toBe('lists/main');
    expect(access.collections.state).toBe(firestoreCollections.state);
    expect(access.collections.history).toBe(firestoreCollections.history);
    expect(access.collections.backups).toBe(firestoreCollections.backups);
    expect(access.backups.history.path).toBe('list_state_history/main--minute--2026-07-12T07:08');
    expect(access.backups.day.path).toBe('list_state_backups/main--day--2026-07-12');
    expect(access.backups.week.path).toMatch(/^list_state_backups\/main--week--2026-W\d{2}$/);
    expect(access.backups.month.path).toBe('list_state_backups/main--month--2026-07');
  });

  test('creates day week month and minute backup targets', () => {
    const targets = createBackupTargets(defaultTarget, new Date('2026-07-12T07:08:00.000Z'));

    expect(targets.history.id).toBe('main--minute--2026-07-12T07:08');
    expect(targets.day.id).toBe('main--day--2026-07-12');
    expect(targets.week.id).toMatch(/^main--week--2026-W\d{2}$/);
    expect(targets.month.id).toBe('main--month--2026-07');
  });

  test('writes live state and backup documents on autosave', async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      status: 200,
      async json() {
        return {};
      }
    }));

    const adapter = createBackendAdapter({
      fetchImpl,
      now: () => new Date('2026-07-12T07:08:00.000Z')
    });

    await adapter.save(sampleState, { reason: 'autosave', createBackup: true });

    expect(fetchImpl).toHaveBeenCalledTimes(5);
    expect(fetchImpl.mock.calls[0][0]).toContain('/documents/lists/main?key=');
    expect(fetchImpl.mock.calls[1][0]).toContain('/documents/list_state_history/main--minute--2026-07-12T07:08?key=');
    expect(fetchImpl.mock.calls[2][0]).toContain('/documents/list_state_backups/main--day--2026-07-12?key=');
    expect(fetchImpl.mock.calls[3][0]).toContain('/documents/list_state_backups/main--week--2026-W');
    expect(fetchImpl.mock.calls[4][0]).toContain('/documents/list_state_backups/main--month--2026-07?key=');

    const payload = JSON.parse(fetchImpl.mock.calls[1][1].body);
    expect(payload.fields.reason.stringValue).toBe('autosave');
    expect(payload.fields.backupKind.stringValue).toBe('minute');
  });
});
