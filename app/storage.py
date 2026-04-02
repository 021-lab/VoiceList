from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import PromptConfig, TaskItem

DEFAULT_USER_PROMPT = (
    "Ты интерпретатор голосовых команд для списка задач. "
    "Верни ТОЛЬКО JSON: {\"action\":\"add|delete|rename\",\"argument\":string|null,\"confidence\":0..1}. "
    "АЛГОРИТМ: "
    "1) Определи команду по ПЕРВОМУ слову команды. "
    "Слово может содержать несколько опечаток; выбери ближайшую команду по смыслу и написанию. "
    "Всегда выводи confidence и наиболее вероятную команду в поле action. "
    "Если confidence < 0.5, это считается 'команда не распознана', но action все равно должен содержать наиболее вероятную команду. "
    "2) Только если confidence >= 0.5, примени команду к списку: "
    "add/добавление -> argument это текст после первого слова; "
    "delete/удаление -> argument всегда null; "
    "rename/изменение -> argument это новое имя после первого слова. "
    "Если для add/rename аргумент пуст, понижай confidence ниже 0.5."
)


class JsonStorage:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.tasks_path = data_dir / "tasks.json"
        self.prompt_path = data_dir / "prompt.json"
        self._lock = threading.Lock()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _atomic_write(self, path: Path, payload: object) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    def _read_tasks_unlocked(self) -> list[TaskItem]:
        if not self.tasks_path.exists():
            self._atomic_write(self.tasks_path, [])
            return []
        with self.tasks_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return [TaskItem.model_validate(item) for item in raw]

    def load_tasks(self) -> list[TaskItem]:
        with self._lock:
            return self._read_tasks_unlocked()

    def save_tasks(self, tasks: list[TaskItem]) -> None:
        with self._lock:
            payload = [task.model_dump() for task in tasks]
            self._atomic_write(self.tasks_path, payload)

    def add_task(self, name: str) -> TaskItem:
        task = TaskItem(id=str(uuid.uuid4()), name=name, created_at=self._now_iso())
        with self._lock:
            tasks = self._read_tasks_unlocked()
            tasks.append(task)
            payload = [item.model_dump() for item in tasks]
            self._atomic_write(self.tasks_path, payload)
        return task

    def delete_task(self, task_id: str) -> bool:
        with self._lock:
            tasks = self._read_tasks_unlocked()
            kept = [task for task in tasks if task.id != task_id]
            if len(kept) == len(tasks):
                return False
            payload = [item.model_dump() for item in kept]
            self._atomic_write(self.tasks_path, payload)
            return True

    def rename_task(self, task_id: str, new_name: str) -> bool:
        with self._lock:
            tasks = self._read_tasks_unlocked()
            changed = False
            for task in tasks:
                if task.id == task_id:
                    task.name = new_name
                    changed = True
                    break
            if not changed:
                return False
            payload = [item.model_dump() for item in tasks]
            self._atomic_write(self.tasks_path, payload)
            return True

    def get_prompt(self) -> PromptConfig:
        with self._lock:
            if not self.prompt_path.exists():
                prompt = PromptConfig(user_prompt=DEFAULT_USER_PROMPT, updated_at=self._now_iso())
                self._atomic_write(self.prompt_path, prompt.model_dump())
                return prompt
            with self.prompt_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        return PromptConfig.model_validate(raw)

    def set_prompt(self, user_prompt: str) -> PromptConfig:
        prompt = PromptConfig(user_prompt=user_prompt, updated_at=self._now_iso())
        with self._lock:
            self._atomic_write(self.prompt_path, prompt.model_dump())
        return prompt
