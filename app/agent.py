from __future__ import annotations

import json
import os
import re
import tempfile
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.schemas import LLMInput, LLMOutput, ParsedCommand, TaskItem

SYSTEM_PROMPT = (
    "Interpret the transcript as one of the task commands. "
    "Allowed actions: add, delete, rename. "
    "Return JSON only with keys: action, argument, confidence."
)


class AsrEngine(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> str:
        raise NotImplementedError


class LlmEngine(ABC):
    @abstractmethod
    def parse_command(self, llm_input: LLMInput) -> LLMOutput:
        raise NotImplementedError


class PlaceholderAsrEngine(AsrEngine):
    def transcribe(self, audio_bytes: bytes) -> str:
        raise RuntimeError("ASR engine is not configured")


class PlaceholderLiquidLlmEngine(LlmEngine):
    def parse_command(self, llm_input: LLMInput) -> LLMOutput:
        raise RuntimeError("Liquid LLM engine is not configured")


class RuleBasedLlmEngine(LlmEngine):
    """Fallback parser used when no local LLM endpoint is configured."""

    def parse_command(self, llm_input: LLMInput) -> LLMOutput:
        text = llm_input.transcript.lower().strip()
        normalized = " ".join(text.split()).strip(".,!?")

        add_match = re.match(r"^(?:please\s+)?add\s+(.+)$", normalized)
        if add_match:
            return LLMOutput(action="add", argument=add_match.group(1).strip() or None, confidence=0.95)

        ru_add_match = re.match(
            r"^(?:пожалуйста\s+)?(?:добавить|добавь)(?:\s+задачу)?\s+(.+)$",
            normalized,
        )
        if ru_add_match:
            return LLMOutput(action="add", argument=ru_add_match.group(1).strip() or None, confidence=0.95)

        delete_match = re.match(
            r"^(?:please\s+)?(?:delete|remove)(?:\s+(?:task|item))?$",
            normalized,
        )
        if delete_match:
            return LLMOutput(action="delete", argument=None, confidence=0.95)

        ru_delete_match = re.match(
            r"^(?:пожалуйста\s+)?(?:удалить|удали)(?:\s+(?:задачу|элемент))?$",
            normalized,
        )
        if ru_delete_match:
            return LLMOutput(action="delete", argument=None, confidence=0.95)

        rename_match = re.match(r"^rename\s+(.+)$", normalized)
        if rename_match:
            argument = rename_match.group(1).strip()
            argument = re.sub(r"^(?:to)\s+", "", argument).strip()
            return LLMOutput(action="rename", argument=argument or None, confidence=0.95)

        ru_rename_match = re.match(r"^(?:изменить|измени|переименуй)\s+(.+)$", normalized)
        if ru_rename_match:
            argument = ru_rename_match.group(1).strip()
            argument = re.sub(r"^(?:в)\s+", "", argument).strip()
            return LLMOutput(action="rename", argument=argument or None, confidence=0.95)

        return LLMOutput(action="add", argument=None, confidence=0.1)


class FasterWhisperAsrEngine(AsrEngine):
    def __init__(self, model_size: str = "base") -> None:
        self.model_size = model_size

    def transcribe(self, audio_bytes: bytes) -> str:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("faster-whisper is not installed") from exc

        model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        with tempfile.NamedTemporaryFile(suffix=".webm") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            segments, _ = model.transcribe(tmp.name)
            return " ".join(segment.text for segment in segments).strip()


class LiquidHttpLlmEngine(LlmEngine):
    def __init__(self, endpoint: str, model: str = "LFM2-700M", timeout_seconds: float = 10.0) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "LiquidHttpLlmEngine | None":
        endpoint = os.getenv("LIQUID_ENDPOINT")
        if not endpoint:
            return None
        model = os.getenv("LIQUID_MODEL", "LFM2-700M")
        return cls(endpoint=endpoint, model=model)

    def parse_command(self, llm_input: LLMInput) -> LLMOutput:
        payload = {
            "model": self.model,
            "system_prompt": llm_input.system_prompt,
            "user_prompt": llm_input.user_prompt,
            "transcript": llm_input.transcript,
            "selected_item_id": llm_input.selected_item_id,
            "tasks": [task.model_dump() for task in llm_input.tasks_snapshot],
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(self.endpoint, json=payload)
            response.raise_for_status()
            body = response.json()

        if isinstance(body, dict) and {"action", "argument", "confidence"} <= body.keys():
            return LLMOutput.from_any(body)

        text_output = body["output"] if isinstance(body, dict) and "output" in body else str(body)
        return parse_llm_json(text_output)


class CommandAgent:
    def __init__(self, asr_engine: AsrEngine, llm_engine: LlmEngine) -> None:
        self.asr_engine = asr_engine
        self.llm_engine = llm_engine

    def build_input(
        self,
        user_prompt: str,
        tasks_snapshot: list[TaskItem],
        transcript: str,
        selected_item_id: str | None,
    ) -> LLMInput:
        return LLMInput(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tasks_snapshot=tasks_snapshot,
            transcript=transcript,
            selected_item_id=selected_item_id,
        )

    def parse(self, llm_input: LLMInput) -> ParsedCommand:
        output = self.llm_engine.parse_command(llm_input)
        return ParsedCommand(action=output.action, argument=output.argument, confidence=output.confidence)


def parse_llm_json(payload_text: str) -> LLMOutput:
    loaded: Any = json.loads(payload_text)
    return LLMOutput.from_any(loaded)
