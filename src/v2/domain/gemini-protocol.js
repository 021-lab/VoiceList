/** The Live API wire protocol: what both sides of the socket need to agree on.
 *
 *  Deliberately free of imports. The browser needs these constants and frame helpers, and
 *  anything this module pulled in would be pulled into the page bundle with it — which is
 *  how a server-side validation library once ended up inlined into the HTML. */

export const GEMINI_MODEL = 'gemini-3.8-live';
export const GEMINI_VOICE = 'Zephyr';
export const GEMINI_LANGUAGE = 'ru-RU';

/** Live API fixes these: raw PCM16 little-endian, 16 kHz up, 24 kHz down. */
export const GEMINI_INPUT_SAMPLE_RATE = 16_000;
export const GEMINI_OUTPUT_SAMPLE_RATE = 24_000;

/** The constrained endpoint is the one an ephemeral token opens: it enforces what the token
 *  was minted with. The plain BidiGenerateContent endpoint takes an API key instead, which is
 *  exactly what must not reach a browser. */
export const GEMINI_WS_URL =
  'wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained';
export const GEMINI_TOKENS_URL = 'https://generativelanguage.googleapis.com/v1beta/auth_tokens';

/** The client's first frame. Everything else about the session already travelled in the
 *  token, so this only names the model. */
export const buildClientSetup = (model = GEMINI_MODEL) => ({ setup: { model: `models/${model}` } });

/** The frame carrying tool calls. Gemini sends them together, and each one answers with its
 *  own id, so the shape is a list both ways. */
export function readToolCalls(frame) {
  const calls = frame?.toolCall?.functionCalls;
  if (!Array.isArray(calls)) return [];
  return calls
    .filter(call => call?.name)
    .map(call => ({ id: call.id || '', name: String(call.name), arguments: call.args && typeof call.args === 'object' ? call.args : {} }));
}

/** The API expects each answer wrapped in a result object, so what the tool returned travels
 *  under `result` rather than as the response itself. */
export function buildToolResponse(results) {
  return {
    toolResponse: {
      functionResponses: results.map(result => ({
        id: result.id,
        name: result.name,
        response: { result: result.response },
        ...(result.scheduling ? { scheduling: result.scheduling } : {})
      }))
    }
  };
}

/** Audio frames are the bulk of the traffic and say nothing a transcript does not; the
 *  log keeps what can be read. */
export function isLoggableFrame(frame) {
  if (!frame || typeof frame !== 'object') return false;
  const parts = frame.serverContent?.modelTurn?.parts;
  if (Array.isArray(parts) && parts.every(part => part?.inlineData)) return false;
  return true;
}
