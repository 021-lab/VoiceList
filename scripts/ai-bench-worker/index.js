// Benchmark proxy: takes an OpenAI chat-completions request, runs it on Workers AI through the
// AI binding, answers in chat-completions shape. x-infer-ms is the time spent inside env.AI.run.
export default {
  async fetch(request, env) {
    if (request.headers.get('Authorization') !== 'Bearer ' + env.BENCH_KEY) return new Response('not found', { status: 404 });
    const url = new URL(request.url);
    if (url.pathname === '/models') {
      try { return Response.json(await env.AI.models({ task: 'Text Generation', per_page: 200 })); } catch (error) { return Response.json({ error: String(error) }, { status: 500 }); }
    }
    const body = await request.json();
    const { model, messages, tools, temperature, seed, max_completion_tokens, reasoning_effort, extra = {} } = body;
    const input = { messages, ...(tools ? { tools } : {}), ...(temperature != null ? { temperature } : {}), ...(seed != null ? { seed } : {}),
      max_tokens: max_completion_tokens || 1500, ...(reasoning_effort ? { reasoning_effort } : {}), ...extra };
    const started = Date.now();
    let out;
    try { out = await env.AI.run(model, input); } catch (error) { return Response.json({ error: { message: String(error?.message || error) } }, { status: 502 }); }
    const ms = Date.now() - started;
    const headers = { 'x-infer-ms': String(ms) };
    if (out?.choices) {
      // Qwen3 with thinking off: Workers AI's parser leaves the call as text in reasoning.
      for (const choice of out.choices) {
        const message = choice.message || {};
        if (message.tool_calls?.length) continue;
        const text = [message.content, message.reasoning_content ?? message.reasoning].filter(value => typeof value === 'string').join('\n');
        const found = [...text.matchAll(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g)].map((match, index) => {
          try { const call = JSON.parse(match[1]); return { id: 't' + index, type: 'function', function: { name: call.name, arguments: typeof call.arguments === 'string' ? call.arguments : JSON.stringify(call.arguments || {}) } }; } catch { return null; }
        }).filter(Boolean);
        if (found.length) { message.tool_calls = found; message.content = null; headers['x-tool-call-from-text'] = '1'; }
      }
      return Response.json(out, { headers });
    }
    const calls = (out?.tool_calls || []).map((call, index) => ({ id: 'c' + index, type: 'function', function: {
      name: call.name ?? call.function?.name,
      arguments: (() => { const args = call.arguments ?? call.function?.arguments ?? {}; return typeof args === 'string' ? args : JSON.stringify(args); })() } }));
    const content = typeof out?.response === 'string' ? out.response : out?.response == null ? null : JSON.stringify(out.response);
    return Response.json({ choices: [{ message: { role: 'assistant', content, tool_calls: calls.length ? calls : undefined } }], raw: calls.length ? undefined : out }, { headers });
  }
};
