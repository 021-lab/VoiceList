import { buildLogEntry } from '../../src/v2/domain/live-log-entry.js';

/** Flat event log for GPT-Live sessions, in the document object's own SQLite.
 *  Nothing is deleted on a schedule: the table is watched rather than trimmed. */
export class LiveLog {
  constructor(storage) {
    this.storage = storage;
    this.sql = storage.sql;
    this.sql.exec(`CREATE TABLE IF NOT EXISTS vl_live_events (
      seq INTEGER PRIMARY KEY,
      live_session_id TEXT NOT NULL,
      at TEXT NOT NULL,
      offset_ms INTEGER,
      type TEXT NOT NULL,
      delegation_id TEXT,
      response_id TEXT,
      call_id TEXT,
      direction TEXT NOT NULL,
      payload TEXT NOT NULL)`);
    this.sql.exec('CREATE INDEX IF NOT EXISTS vl_live_events_session ON vl_live_events (live_session_id, seq)');
  }

  append(event, options) {
    const row = buildLogEntry(event, options);
    this.sql.exec(
      'INSERT INTO vl_live_events (live_session_id,at,offset_ms,type,delegation_id,response_id,call_id,direction,payload) VALUES (?,?,?,?,?,?,?,?,?)',
      row.liveSessionId, row.at, row.offsetMs, row.type, row.delegationId, row.responseId, row.callId, row.direction, row.payload
    );
    return row;
  }

  read({ sessionId = '', afterSeq = 0, limit = 200 } = {}) {
    const size = Math.min(Math.max(Number(limit) || 200, 1), 1000);
    const after = Math.max(0, Number(afterSeq) || 0);
    const rows = sessionId
      ? this.sql.exec('SELECT * FROM vl_live_events WHERE live_session_id = ? AND seq > ? ORDER BY seq LIMIT ?', sessionId, after, size).toArray()
      : this.sql.exec('SELECT * FROM vl_live_events WHERE seq > ? ORDER BY seq LIMIT ?', after, size).toArray();
    return rows.map(row => ({
      seq: row.seq, liveSessionId: row.live_session_id, at: row.at, offsetMs: row.offset_ms,
      type: row.type, delegationId: row.delegation_id, responseId: row.response_id,
      callId: row.call_id, direction: row.direction, payload: parse(row.payload)
    }));
  }

  sessions(limit = 50) {
    const size = Math.min(Math.max(Number(limit) || 50, 1), 200);
    return this.sql.exec(
      `SELECT live_session_id, COUNT(*) AS events, MIN(at) AS started_at, MAX(at) AS last_at
       FROM vl_live_events GROUP BY live_session_id ORDER BY MAX(seq) DESC LIMIT ?`, size
    ).toArray().map(row => ({ liveSessionId: row.live_session_id, events: row.events, startedAt: row.started_at, lastAt: row.last_at }));
  }

  /** Size is reported rather than enforced: retention was deliberately left manual. */
  stats() {
    const row = this.sql.exec('SELECT COUNT(*) AS events, SUM(LENGTH(payload)) AS bytes FROM vl_live_events').toArray()[0] || {};
    return { events: Number(row.events) || 0, payloadBytes: Number(row.bytes) || 0 };
  }
}

function parse(value) {
  try { return JSON.parse(value); } catch { return { unparsed: String(value).slice(0, 1000) }; }
}
