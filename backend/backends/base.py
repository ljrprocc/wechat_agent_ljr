from __future__ import annotations

from typing import Any, Protocol


class ChatBackend(Protocol):
    backend_name: str

    def chat(self, model_id: str, messages: list[dict[str, str]]) -> str:
        ...

    def loaded_model_ids(self) -> set[str]:
        ...

    def status(self) -> dict[str, Any]:
        ...
