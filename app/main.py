from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent import (
    CommandAgent,
    FasterWhisperAsrEngine,
    LiquidHttpLlmEngine,
    PlaceholderAsrEngine,
    RuleBasedLlmEngine,
)
from app.schemas import (
    CommandAction,
    PromptUpdateRequest,
    TaskItem,
    VoiceCommandResponse,
)
from app.storage import DEFAULT_USER_PROMPT, JsonStorage


def create_app(
    data_dir: Path | None = None,
    asr_engine=None,
    llm_engine=None,
) -> FastAPI:
    base_dir = Path(__file__).resolve().parent.parent
    storage = JsonStorage(data_dir or (base_dir / "data"))
    resolved_asr = asr_engine
    if resolved_asr is None:
        if os.getenv("USE_FASTER_WHISPER", "1") == "1":
            resolved_asr = FasterWhisperAsrEngine(model_size=os.getenv("WHISPER_MODEL_SIZE", "base"))
        else:
            resolved_asr = PlaceholderAsrEngine()

    resolved_llm = llm_engine
    if resolved_llm is None:
        resolved_llm = LiquidHttpLlmEngine.from_env() or RuleBasedLlmEngine()

    agent = CommandAgent(resolved_asr, resolved_llm)

    app = FastAPI(title="VoiceList")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = base_dir / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/tasks", response_model=list[TaskItem])
    def get_tasks() -> list[TaskItem]:
        return storage.load_tasks()

    @app.get("/api/prompt")
    def get_prompt() -> dict[str, str]:
        return storage.get_prompt().model_dump()

    @app.put("/api/prompt")
    def put_prompt(payload: PromptUpdateRequest) -> dict[str, str]:
        prompt = storage.set_prompt(payload.user_prompt)
        return prompt.model_dump()

    @app.post("/api/prompt/reset")
    def reset_prompt() -> dict[str, str]:
        prompt = storage.set_prompt(DEFAULT_USER_PROMPT)
        return prompt.model_dump()

    @app.post("/api/voice-command", response_model=VoiceCommandResponse)
    async def voice_command(
        selected_item_id: str | None = Form(default=None),
        transcript: str | None = Form(default=None),
        audio: UploadFile | None = File(default=None),
    ) -> VoiceCommandResponse:
        tasks = storage.load_tasks()
        prompt = storage.get_prompt()

        if transcript is None:
            if audio is None:
                raise HTTPException(status_code=400, detail="either transcript or audio must be provided")
            audio_bytes = await audio.read()
            try:
                transcript = agent.asr_engine.transcribe(audio_bytes).strip()
            except Exception as exc:  # noqa: BLE001
                return VoiceCommandResponse(
                    transcript="",
                    parsed_command=None,
                    applied=False,
                    tasks=tasks,
                    error=f"transcription failed: unsupported or invalid audio ({type(exc).__name__})",
                )
        else:
            transcript = transcript.strip()

        if not transcript:
            return VoiceCommandResponse(
                transcript="",
                parsed_command=None,
                applied=False,
                tasks=tasks,
                error="transcript is empty",
            )

        llm_input = agent.build_input(
            user_prompt=prompt.user_prompt,
            tasks_snapshot=tasks,
            transcript=transcript,
            selected_item_id=selected_item_id,
        )

        try:
            parsed_command = agent.parse(llm_input)
        except Exception as exc:  # noqa: BLE001
            return VoiceCommandResponse(
                transcript=transcript,
                parsed_command=None,
                applied=False,
                tasks=tasks,
                error=f"command parsing failed: {exc}",
            )

        if parsed_command.confidence < 0.4:
            return VoiceCommandResponse(
                transcript=transcript,
                parsed_command=parsed_command,
                applied=False,
                tasks=tasks,
                error="low confidence command",
            )

        try:
            updated = apply_command(storage, parsed_command.action, parsed_command.argument, selected_item_id)
        except ValueError as exc:
            return VoiceCommandResponse(
                transcript=transcript,
                parsed_command=parsed_command,
                applied=False,
                tasks=storage.load_tasks(),
                error=str(exc),
            )

        return VoiceCommandResponse(
            transcript=transcript,
            parsed_command=parsed_command,
            applied=updated,
            tasks=storage.load_tasks(),
            error=None if updated else "no matching task",
        )

    return app


def apply_command(
    storage: JsonStorage,
    action: CommandAction,
    argument: str | None,
    selected_item_id: str | None,
) -> bool:
    if action == CommandAction.ADD:
        if not argument:
            raise ValueError("add requires a non-empty argument")
        storage.add_task(argument)
        return True

    if action == CommandAction.DELETE:
        if not selected_item_id:
            raise ValueError("delete requires selected_item_id")
        return storage.delete_task(selected_item_id)

    if action == CommandAction.RENAME:
        if not selected_item_id:
            raise ValueError("rename requires selected_item_id")
        if not argument:
            raise ValueError("rename requires a non-empty argument")
        return storage.rename_task(selected_item_id, argument)

    raise ValueError("unsupported action")


app = create_app()
