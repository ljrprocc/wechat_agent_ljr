from __future__ import annotations

from typing import Any

from backend.backends import create_backend
from backend.backends.base import ChatBackend
from backend.model_registry import ModelRegistry
from backend.runtime import RuntimeSettings
from backend.session_store import SessionStore


class LocalChatAgent:
    def __init__(self, settings: RuntimeSettings, registry: ModelRegistry | None = None) -> None:
        self.settings = settings
        self.registry = registry or ModelRegistry.from_settings(settings)
        self.store = SessionStore(settings.db_path)
        self.backend: ChatBackend = create_backend(settings=self.settings, registry=self.registry)

    def list_models(self) -> list[dict[str, Any]]:
        return self.registry.list_models(loaded_model_ids=self.backend.loaded_model_ids())

    def reply(
        self,
        session_id: str,
        user_text: str,
        model_id: str | None = None,
        stream: bool = False,
    ) -> str:
        if stream:
            raise NotImplementedError("Streaming responses are reserved for v1.x and are not implemented in v1.0.")

        resolved_model = self.registry.get(model_id)
        resolved_session_id = session_id.strip()
        text = user_text.strip()

        if not resolved_session_id:
            raise ValueError("session_id cannot be empty")
        if not text:
            return "我收到的是空消息，可以再发一次。"

        if text in {"/reset", "/clear", "重置对话", "清空对话"}:
            self.store.clear(resolved_session_id, resolved_model.model_id)
            return "当前会话记忆已清空。"

        history = self.store.get_recent_messages(
            resolved_session_id,
            resolved_model.model_id,
            limit=self.settings.history_turns * 2,
        )
        messages = [{"role": "system", "content": self.settings.system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": text})

        answer = self.backend.chat(resolved_model.model_id, messages)
        self.store.append(resolved_session_id, resolved_model.model_id, "user", text)
        self.store.append(resolved_session_id, resolved_model.model_id, "assistant", answer)
        return answer

    def status(self) -> dict[str, Any]:
        return {
            "model_backend": self.settings.model_backend,
            "default_model_id": self.registry.default_model_id,
            "supported_models": self.list_models(),
            "db_path": self.settings.db_path,
            "history_turns": self.settings.history_turns,
            "max_new_tokens": self.settings.max_new_tokens,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            **self.backend.status(),
        }


SimpleWechatAgent = LocalChatAgent


def build_agent_from_env() -> LocalChatAgent:
    return LocalChatAgent(RuntimeSettings.from_env())
