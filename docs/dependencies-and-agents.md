# Dependencies and Agent Configuration

## Runtime dependencies

- Python `>=3.11`
- `fastapi>=0.116.0`
- `uvicorn>=0.35.0`
- `pydantic>=2.11.0`
- `python-multipart>=0.0.20`
- `httpx>=0.28.0`
- `faster-whisper>=1.2.0`

Notes:
- `faster-whisper` is only required when ASR is enabled and audio is sent to `/api/voice-command`.
- If your client sends `transcript` directly, ASR can be disabled.

## Development dependencies

- `pytest>=8.4.0`
- `pytest-asyncio>=1.1.0`

## Replacing local agents through project settings

The app supports replacing local ASR/LLM agents without changing API routes.

### Environment-based replacement

ASR selection:
- `USE_FASTER_WHISPER=1` -> `FasterWhisperAsrEngine`
- `USE_FASTER_WHISPER=0` -> `PlaceholderAsrEngine` (requires `transcript`)
- `WHISPER_MODEL_SIZE` controls Faster-Whisper model size (default `base`)

LLM selection:
- `LIQUID_ENDPOINT` set -> `LiquidHttpLlmEngine`
- `LIQUID_MODEL` optional model id for Liquid endpoint (default `LFM2-700M`)
- `LIQUID_ENDPOINT` unset -> built-in `RuleBasedLlmEngine`

### Code-level replacement

`create_app()` accepts injected engines:

```python
from app.main import create_app

app = create_app(
    asr_engine=my_custom_asr,
    llm_engine=my_custom_llm,
)
```

This allows swapping local models/providers via project configuration or application wiring.
