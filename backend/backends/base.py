from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class BackendDebugMetrics:
    backend: str
    model_id: str
    latency_ms: float
    generation_ms: float | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    tokens_per_second: float | None = None
    model_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "backend": self.backend,
            "model_id": self.model_id,
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.generation_ms is not None:
            payload["generation_ms"] = round(self.generation_ms, 2)
        if self.prompt_tokens is not None:
            payload["prompt_tokens"] = self.prompt_tokens
        if self.generated_tokens is not None:
            payload["generated_tokens"] = self.generated_tokens
        if self.tokens_per_second is not None:
            payload["tokens_per_second"] = round(self.tokens_per_second, 2)
        if self.model_source:
            payload["model_source"] = self.model_source
        return payload


@dataclass(frozen=True)
class BackendChatResult:
    text: str
    debug: BackendDebugMetrics | None = None


class ChatBackend(Protocol):
    backend_name: str

    def chat(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        *,
        capture_debug: bool = False,
    ) -> BackendChatResult:
        ...

    def loaded_model_ids(self) -> set[str]:
        ...

    def status(self) -> dict[str, Any]:
        ...
