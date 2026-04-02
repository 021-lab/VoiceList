from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agent import AsrEngine, LlmEngine
from app.main import create_app
from app.schemas import CommandAction, LLMInput, LLMOutput


class DummyAsr(AsrEngine):
    def transcribe(self, audio_bytes: bytes) -> str:
        return "add fallback"


class FailingAsr(AsrEngine):
    def transcribe(self, audio_bytes: bytes) -> str:
        raise RuntimeError("decode failed")


class EmptyAsr(AsrEngine):
    def transcribe(self, audio_bytes: bytes) -> str:
        return "   "


class TranscriptLlm(LlmEngine):
    def __init__(self, invalid: bool = False) -> None:
        self.invalid = invalid
        self.last_input: LLMInput | None = None

    def parse_command(self, llm_input: LLMInput) -> LLMOutput:
        self.last_input = llm_input
        if self.invalid:
            raise ValueError("invalid llm json")

        text = llm_input.transcript.lower().strip()

        if "пожалуйста" in text and "задачу" in text:
            phrase = text.replace("пожалуйста", "").replace("добавь", "").replace("задачу", "").strip()
            return LLMOutput(action=CommandAction.ADD, argument=phrase, confidence=0.82)

        if "добав" in text or text.startswith("add"):
            argument = text.replace("добавить", "").replace("add", "", 1).strip()
            return LLMOutput(action=CommandAction.ADD, argument=argument or None, confidence=0.95)

        if "удал" in text or text.startswith("delete"):
            return LLMOutput(action=CommandAction.DELETE, argument=None, confidence=0.95)

        if "измен" in text or text.startswith("rename"):
            argument = text.replace("изменить", "").replace("rename", "", 1).strip()
            return LLMOutput(action=CommandAction.RENAME, argument=argument or None, confidence=0.95)

        return LLMOutput(action=CommandAction.ADD, argument=None, confidence=0.1)


@pytest.fixture
def test_env(tmp_path: Path):
    llm = TranscriptLlm()
    app = create_app(data_dir=tmp_path, asr_engine=DummyAsr(), llm_engine=llm)
    client = TestClient(app)
    return client, llm, tmp_path


def _voice(client: TestClient, transcript: str, selected_item_id: str | None = None):
    data = {"transcript": transcript}
    if selected_item_id is not None:
        data["selected_item_id"] = selected_item_id
    return client.post("/api/voice-command", data=data)


def _task_ids(client: TestClient) -> list[str]:
    payload = client.get("/api/tasks").json()
    return [item["id"] for item in payload]


def test_add_ru(test_env):
    client, _, _ = test_env
    res = _voice(client, "добавить купить молоко")
    body = res.json()
    assert res.status_code == 200
    assert body["applied"] is True
    assert len(body["tasks"]) == 1


def test_add_en(test_env):
    client, _, _ = test_env
    res = _voice(client, "add buy milk")
    body = res.json()
    assert body["applied"] is True
    assert body["tasks"][0]["name"] == "buy milk"


def test_delete_ru_selected_item(test_env):
    client, _, _ = test_env
    _voice(client, "add first")
    _voice(client, "add second")
    ids = _task_ids(client)
    res = _voice(client, "удалить", selected_item_id=ids[0])
    body = res.json()
    assert body["applied"] is True
    assert len(body["tasks"]) == 1
    assert body["tasks"][0]["id"] == ids[1]


def test_rename_ru_selected_item(test_env):
    client, _, _ = test_env
    _voice(client, "add old name")
    task_id = _task_ids(client)[0]
    res = _voice(client, "изменить новое имя", selected_item_id=task_id)
    body = res.json()
    assert body["applied"] is True
    assert body["tasks"][0]["name"] == "новое имя"


def test_duplicate_names_delete_by_id(test_env):
    client, _, _ = test_env
    _voice(client, "add same")
    _voice(client, "add same")
    ids = _task_ids(client)
    res = _voice(client, "delete", selected_item_id=ids[0])
    body = res.json()
    assert body["applied"] is True
    assert len(body["tasks"]) == 1
    assert body["tasks"][0]["id"] == ids[1]


def test_missing_selected_item_errors(test_env):
    client, _, _ = test_env
    delete_res = _voice(client, "delete")
    rename_res = _voice(client, "rename fresh")
    assert delete_res.json()["applied"] is False
    assert "selected_item_id" in delete_res.json()["error"]
    assert rename_res.json()["applied"] is False
    assert "selected_item_id" in rename_res.json()["error"]


def test_empty_argument_errors(test_env):
    client, _, _ = test_env
    add_res = _voice(client, "add")
    _voice(client, "add source")
    task_id = _task_ids(client)[0]
    rename_res = _voice(client, "rename", selected_item_id=task_id)
    assert add_res.json()["applied"] is False
    assert "non-empty argument" in add_res.json()["error"]
    assert rename_res.json()["applied"] is False
    assert "non-empty argument" in rename_res.json()["error"]


def test_unknown_command_low_confidence(test_env):
    client, _, _ = test_env
    res = _voice(client, "сделай красиво")
    body = res.json()
    assert body["applied"] is False
    assert body["error"] == "low confidence command"


def test_noise_phrase_adds(test_env):
    client, _, _ = test_env
    res = _voice(client, "пожалуйста добавь задачу позвонить маме")
    body = res.json()
    assert body["applied"] is True
    assert body["tasks"][0]["name"] == "позвонить маме"


def test_invalid_llm_output_returns_graceful_error(tmp_path: Path):
    app = create_app(data_dir=tmp_path, asr_engine=DummyAsr(), llm_engine=TranscriptLlm(invalid=True))
    client = TestClient(app)
    res = _voice(client, "add test")
    body = res.json()
    assert res.status_code == 200
    assert body["applied"] is False
    assert "command parsing failed" in body["error"]


def test_concurrent_add_keeps_integrity(test_env):
    client, _, _ = test_env

    def run_once(i: int):
        response = _voice(client, f"add item-{i}")
        assert response.status_code == 200

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in range(30):
            pool.submit(run_once, i)

    tasks_res = client.get("/api/tasks")
    tasks = tasks_res.json()
    assert len(tasks) == 30


def test_get_prompt_creates_default(test_env):
    client, _, _ = test_env
    res = client.get("/api/prompt")
    body = res.json()
    assert res.status_code == 200
    assert body["user_prompt"]


def test_put_prompt_persists_across_restart(tmp_path: Path):
    llm = TranscriptLlm()
    app_a = create_app(data_dir=tmp_path, asr_engine=DummyAsr(), llm_engine=llm)
    client_a = TestClient(app_a)

    put_res = client_a.put("/api/prompt", json={"user_prompt": "custom rule"})
    assert put_res.status_code == 200

    app_b = create_app(data_dir=tmp_path, asr_engine=DummyAsr(), llm_engine=llm)
    client_b = TestClient(app_b)
    get_res = client_b.get("/api/prompt")
    assert get_res.json()["user_prompt"] == "custom rule"


def test_put_prompt_is_used_in_next_command(test_env):
    client, llm, _ = test_env
    put_res = client.put("/api/prompt", json={"user_prompt": "strictly parse tasks"})
    assert put_res.status_code == 200

    _voice(client, "add apples")

    assert llm.last_input is not None
    assert llm.last_input.user_prompt == "strictly parse tasks"


def test_reset_default_prompt_affects_next_command(test_env):
    client, llm, _ = test_env
    client.put("/api/prompt", json={"user_prompt": "custom"})
    client.post("/api/prompt/reset")

    _voice(client, "add reset check")

    assert llm.last_input is not None
    assert "интерпретатор голосовых команд" in llm.last_input.user_prompt


def test_put_prompt_validation_does_not_override_existing(test_env):
    client, _, _ = test_env
    client.put("/api/prompt", json={"user_prompt": "stable prompt"})
    bad = client.put("/api/prompt", json={"user_prompt": "    "})
    current = client.get("/api/prompt").json()
    assert bad.status_code == 422
    assert current["user_prompt"] == "stable prompt"


def test_default_app_works_without_liquid_endpoint(tmp_path: Path):
    app = create_app(data_dir=tmp_path, asr_engine=DummyAsr(), llm_engine=None)
    client = TestClient(app)
    res = _voice(client, "add working fallback")
    body = res.json()
    assert res.status_code == 200
    assert body["applied"] is True
    assert body["tasks"][0]["name"] == "working fallback"


def test_default_fallback_parses_colloquial_russian_add(tmp_path: Path):
    app = create_app(data_dir=tmp_path, asr_engine=DummyAsr(), llm_engine=None)
    client = TestClient(app)
    res = _voice(client, "добавь задачу купить молоко")
    body = res.json()
    assert res.status_code == 200
    assert body["applied"] is True
    assert body["tasks"][0]["name"] == "купить молоко"


def test_default_fallback_parses_colloquial_russian_delete(tmp_path: Path):
    app = create_app(data_dir=tmp_path, asr_engine=DummyAsr(), llm_engine=None)
    client = TestClient(app)
    _voice(client, "add temp item")
    task_id = _task_ids(client)[0]
    res = _voice(client, "удали задачу", selected_item_id=task_id)
    body = res.json()
    assert res.status_code == 200
    assert body["applied"] is True
    assert body["tasks"] == []


def test_default_fallback_parses_colloquial_russian_rename(tmp_path: Path):
    app = create_app(data_dir=tmp_path, asr_engine=DummyAsr(), llm_engine=None)
    client = TestClient(app)
    _voice(client, "add old title")
    task_id = _task_ids(client)[0]
    res = _voice(client, "переименуй в новое имя", selected_item_id=task_id)
    body = res.json()
    assert res.status_code == 200
    assert body["applied"] is True
    assert body["tasks"][0]["name"] == "новое имя"


def test_audio_transcription_failure_returns_structured_error(tmp_path: Path):
    app = create_app(data_dir=tmp_path, asr_engine=FailingAsr(), llm_engine=TranscriptLlm())
    client = TestClient(app)
    res = client.post(
        "/api/voice-command",
        files={"audio": ("voice.webm", b"not-real-audio", "audio/webm")},
    )
    body = res.json()
    assert res.status_code == 200
    assert body["applied"] is False
    assert body["error"].startswith("transcription failed:")


def test_audio_empty_transcript_returns_structured_error(tmp_path: Path):
    app = create_app(data_dir=tmp_path, asr_engine=EmptyAsr(), llm_engine=TranscriptLlm())
    client = TestClient(app)
    res = client.post(
        "/api/voice-command",
        files={"audio": ("voice.webm", b"not-real-audio", "audio/webm")},
    )
    body = res.json()
    assert res.status_code == 200
    assert body["applied"] is False
    assert body["error"] == "transcript is empty"
