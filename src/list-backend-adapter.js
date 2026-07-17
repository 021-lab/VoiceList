const firebaseConfig = {
  apiKey: 'AIzaSyBtttOmje8yQvU1mf1-zDbOq5OlBHLt6Ic',
  projectId: 'ai-labg'
};

export const firestoreCollections = Object.freeze({
  state: 'lists',
  history: 'list_state_history',
  backups: 'list_state_backups'
});

export const defaultTarget = Object.freeze({
  collection: firestoreCollections.state,
  id: 'main'
});

async function fetchWithTimeout(fetchImpl, url, options, timeoutMs) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetchImpl(url, {
      ...options,
      signal: controller.signal
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

function pad(value) {
  return String(value).padStart(2, '0');
}

function formatMinuteStamp(date) {
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}T${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}`;
}

function formatDayStamp(date) {
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}`;
}

function formatMonthStamp(date) {
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}`;
}

function formatWeekStamp(date) {
  const utcDate = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
  const day = utcDate.getUTCDay() || 7;
  utcDate.setUTCDate(utcDate.getUTCDate() + 4 - day);
  const yearStart = new Date(Date.UTC(utcDate.getUTCFullYear(), 0, 1));
  const weekNumber = Math.ceil((((utcDate - yearStart) / 86400000) + 1) / 7);
  return `${utcDate.getUTCFullYear()}-W${pad(weekNumber)}`;
}

function buildDocumentUrl(target) {
  const path = `projects/${firebaseConfig.projectId}/databases/(default)/documents/${target.collection}/${target.id}`;
  return `https://firestore.googleapis.com/v1/${path}?key=${encodeURIComponent(firebaseConfig.apiKey)}`;
}

function extractState(snapshotData) {
  if (!snapshotData || typeof snapshotData !== 'object') return null;
  if (snapshotData.snapshot?.items && Array.isArray(snapshotData.actionLog)) return snapshotData;
  return null;
}

function encodeDocument(state, target, metadata) {
  return {
    fields: {
      payload: {
        stringValue: JSON.stringify({
          snapshot: state.snapshot,
          actionLog: state.actionLog
        })
      },
      target: {
        stringValue: `${target.collection}/${target.id}`
      },
      reason: {
        stringValue: metadata.reason
      },
      backupKind: {
        stringValue: metadata.backupKind
      },
      updatedAtMs: {
        integerValue: String(metadata.updatedAtMs)
      }
    }
  };
}

function decodeDocument(documentData) {
  const payload = documentData?.fields?.payload?.stringValue;
  if (!payload) return null;

  try {
    return extractState(JSON.parse(payload));
  } catch (error) {
    console.warn('Firestore payload parse skipped', error);
    return null;
  }
}

export function createBackupTargets(target = defaultTarget, currentDate = new Date()) {
  const baseId = target.id;

  return {
    history: {
      collection: firestoreCollections.history,
      id: `${baseId}--minute--${formatMinuteStamp(currentDate)}`
    },
    day: {
      collection: firestoreCollections.backups,
      id: `${baseId}--day--${formatDayStamp(currentDate)}`
    },
    week: {
      collection: firestoreCollections.backups,
      id: `${baseId}--week--${formatWeekStamp(currentDate)}`
    },
    month: {
      collection: firestoreCollections.backups,
      id: `${baseId}--month--${formatMonthStamp(currentDate)}`
    }
  };
}

export function describeFirestoreAccess(target = defaultTarget, currentDate = new Date()) {
  const backups = createBackupTargets(target, currentDate);
  return {
    current: {
      collection: target.collection,
      id: target.id,
      path: `${target.collection}/${target.id}`
    },
    collections: {
      state: firestoreCollections.state,
      history: firestoreCollections.history,
      backups: firestoreCollections.backups
    },
    backups: Object.fromEntries(
      Object.entries(backups).map(([key, value]) => [
        key,
        {
          ...value,
          path: `${value.collection}/${value.id}`
        }
      ])
    )
  };
}

export function createBackendAdapter({
  target = defaultTarget,
  fetchImpl = fetch,
  now = () => new Date(),
  timeouts = { load: 1500, save: 4000 }
} = {}) {
  async function writeDocument(documentTarget, state, metadata) {
    const response = await fetchWithTimeout(
      fetchImpl,
      buildDocumentUrl(documentTarget),
      {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(encodeDocument(state, documentTarget, metadata))
      },
      timeouts.save
    );

    if (!response.ok) {
      throw new Error(`Firestore save failed: ${response.status}`);
    }
  }

  return {
    getAccessInfo(currentDate = now()) {
      return describeFirestoreAccess(target, currentDate);
    },

    async load() {
      try {
        const response = await fetchWithTimeout(fetchImpl, buildDocumentUrl(target), { method: 'GET' }, timeouts.load);
        if (response.status === 404) return null;
        if (!response.ok) throw new Error(`Firestore load failed: ${response.status}`);
        return decodeDocument(await response.json());
      } catch (error) {
        console.warn('Firestore load skipped', error);
        return null;
      }
    },

    async loadVersion(versionTarget) {
      try {
        const response = await fetchWithTimeout(fetchImpl, buildDocumentUrl(versionTarget), { method: 'GET' }, timeouts.load);
        if (response.status === 404) return null;
        if (!response.ok) throw new Error(`Firestore load version failed: ${response.status}`);
        return decodeDocument(await response.json());
      } catch (error) {
        console.warn('Firestore version load skipped', error);
        return null;
      }
    },

    async save(state, { reason = 'mutation', createBackup = false } = {}) {
      const timestamp = now();
      const updatedAtMs = timestamp.getTime();

      await writeDocument(target, state, {
        reason,
        backupKind: 'live',
        updatedAtMs
      });

      if (!createBackup) return state;

      const backupTargets = createBackupTargets(target, timestamp);
      await Promise.all([
        writeDocument(backupTargets.history, state, {
          reason,
          backupKind: 'minute',
          updatedAtMs
        }),
        writeDocument(backupTargets.day, state, {
          reason,
          backupKind: 'day',
          updatedAtMs
        }),
        writeDocument(backupTargets.week, state, {
          reason,
          backupKind: 'week',
          updatedAtMs
        }),
        writeDocument(backupTargets.month, state, {
          reason,
          backupKind: 'month',
          updatedAtMs
        })
      ]);

      return state;
    }
  };
}
