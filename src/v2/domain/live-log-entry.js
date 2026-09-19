/** Turns one GPT-Live event into one log row.
 *
 *  The unit of the log is the event, not the turn and not the delegation. The requirement is
 *  to record what the framework emits, and what it emits is a stream of events; making a
 *  delegation the unit would leave everything outside one — clarifications, greetings,
 *  session start and close, usage, errors — with no place in the schema. Delegation, turn
 *  and tool call are recovered by query, through delegation_id, response_id and call_id. */

/** A single event should never be this large once audio is excluded. The cap stops one
 *  malformed payload from filling the table, and says so in the row rather than silently. */
export const MAX_PAYLOAD_CHARS = 64_000;

const text = (value) => (typeof value === 'string' && value ? value : null);
const finite = (value) => (Number.isFinite(value) ? Number(value) : null);

/** For a delegation envelope the outer type is always "response.event", which says nothing.
 *  The inner type is what a reader needs, so the index column carries both. Only this column
 *  is derived; the payload is stored exactly as it arrived. */
export function effectiveType(event) {
  const outer = text(event?.type);
  const inner = text(event?.event?.type);
  if (!outer) return 'unknown';
  return inner ? `${outer}/${inner}` : outer;
}

export function buildLogEntry(event, { sessionId, direction = 'in', at = new Date().toISOString() } = {}) {
  const inner = event?.event && typeof event.event === 'object' ? event.event : null;
  const item = inner?.item && typeof inner.item === 'object' ? inner.item : null;
  const serialized = JSON.stringify(event ?? null);
  const truncated = serialized.length > MAX_PAYLOAD_CHARS;
  return {
    liveSessionId: String(sessionId || ''),
    at,
    offsetMs: finite(event?.offset_ms ?? event?.delegation?.offset_ms ?? inner?.offset_ms),
    type: effectiveType(event),
    delegationId: text(event?.delegation_id ?? event?.delegation?.id ?? inner?.delegation_id),
    responseId: text(inner?.response?.id ?? inner?.response_id ?? event?.response_id),
    callId: text(item?.call_id ?? inner?.call_id ?? event?.call_id),
    direction: direction === 'out' ? 'out' : 'in',
    payload: truncated ? JSON.stringify({ truncated: true, chars: serialized.length, head: serialized.slice(0, MAX_PAYLOAD_CHARS) }) : serialized
  };
}
