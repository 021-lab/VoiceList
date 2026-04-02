from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CommandAction(str, Enum):
    ADD = "add"
    DELETE = "delete"
    RENAME = "rename"


class TaskItem(BaseModel):
    id: str
    name: str
    created_at: str


class PromptConfig(BaseModel):
    user_prompt: str
    updated_at: str


class ParsedCommand(BaseModel):
    action: CommandAction
    argument: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("argument")
    @classmethod
    def normalize_argument(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized if normalized else None


class VoiceCommandResponse(BaseModel):
    transcript: str
    parsed_command: ParsedCommand | None = None
    applied: bool
    tasks: list[TaskItem]
    error: str | None = None


class PromptUpdateRequest(BaseModel):
    user_prompt: str

    @field_validator("user_prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("user_prompt must not be empty")
        if len(normalized) > 4000:
            raise ValueError("user_prompt is too long (max 4000 chars)")
        return normalized


class LLMInput(BaseModel):
    system_prompt: str
    user_prompt: str
    tasks_snapshot: list[TaskItem]
    transcript: str
    selected_item_id: str | None


class LLMOutput(BaseModel):
    action: CommandAction
    argument: str | None = None
    confidence: float

    @classmethod
    def from_any(cls, payload: Any) -> "LLMOutput":
        return cls.model_validate(payload)
