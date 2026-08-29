const MAX_SETUP_REQUEST_BYTES = 2_048;

function errorResponse(error, status) {
  return Response.json({ error }, {
    status,
    headers: { 'Cache-Control': 'no-store' }
  });
}

function hexToBytes(hex) {
  if (!/^[a-f0-9]{64}$/i.test(hex || '')) return null;
  return Uint8Array.from(hex.match(/.{2}/g), (part) => Number.parseInt(part, 16));
}

async function setupTokenMatches(token, expectedHash) {
  const expected = hexToBytes(expectedHash);
  if (!expected || typeof token !== 'string' || token.length < 32 || token.length > 256) return false;
  const actual = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(token)));
  if (typeof crypto.subtle.timingSafeEqual === 'function') return crypto.subtle.timingSafeEqual(actual, expected);
  let mismatch = 0;
  for (let index = 0; index < expected.length; index += 1) mismatch |= actual[index] ^ expected[index];
  return mismatch === 0;
}

function validOpenAIKey(value) {
  return typeof value === 'string' && value.startsWith('sk-') && value.length >= 20 && value.length <= 512 && !/\s/.test(value);
}

export async function handleOpenAIKeyStatus({ configured, setupAvailable }) {
  return Response.json({ configured: Boolean(configured), setupAvailable: Boolean(setupAvailable) }, {
    headers: { 'Cache-Control': 'no-store' }
  });
}

export async function handleOpenAIKeySetup(request, {
  setupTokenHash,
  configureKey
}) {
  if (request.method !== 'POST') return errorResponse('Method not allowed', 405);
  const origin = request.headers.get('Origin');
  if (origin !== new URL(request.url).origin) return errorResponse('Origin not allowed', 403);
  if (!setupTokenHash) return errorResponse('Key setup is disabled', 404);

  const contentLength = Number(request.headers.get('Content-Length') || 0);
  if (contentLength > MAX_SETUP_REQUEST_BYTES) return errorResponse('Request is too large', 413);

  let body;
  try {
    body = await request.json();
  } catch {
    return errorResponse('Invalid JSON body', 400);
  }

  if (!validOpenAIKey(body?.apiKey)) return errorResponse('Invalid OpenAI API key', 400);
  if (!await setupTokenMatches(body?.setupToken, setupTokenHash)) return errorResponse('Invalid or expired setup link', 403);

  const configured = await configureKey(body.apiKey);
  if (!configured) return errorResponse('OpenAI API key is already configured', 409);
  return Response.json({ configured: true }, {
    status: 201,
    headers: { 'Cache-Control': 'no-store' }
  });
}
