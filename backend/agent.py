from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "你是一个本地部署的 Qwen 助手。"
    "请优先用中文回答，尽量简洁、准确；信息不足时直接说明不确定。"
)

MODEL_PATH_ENV_BY_ID = {
    "qwen2.5-0.5b": "QWEN25_05B_MODEL_PATH",
    "qwen2.5-3b": "QWEN25_3B_MODEL_PATH",
}
LEGACY_MODEL_PATH_ENV = "QWEN_MODEL_PATH"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _clip_text(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 12].rstrip() + "\n\n[内容已截断]"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_path: str
    label: str
    family: str
    description: str = ""

    def to_dict(self, loaded: bool = False) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "label": self.label,
            "family": self.family,
            "description": self.description,
            "loaded": loaded,
        }


@dataclass(frozen=True)
class RuntimeSettings:
    model_config_path: str
    default_model_id: str
    device: str
    device_map: str
    torch_dtype: str
    attn_implementation: str
    trust_remote_code: bool
    max_new_tokens: int
    temperature: float
    top_p: float
    history_turns: int
    max_reply_chars: int
    system_prompt: str
    db_path: str

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            model_config_path=os.getenv("QWEN_MODEL_CONFIG_PATH", "./config/models/qwen2.5.json"),
            default_model_id=os.getenv("QWEN_DEFAULT_MODEL_ID", "qwen2.5-0.5b"),
            device=os.getenv("QWEN_DEVICE", "auto"),
            device_map=os.getenv("QWEN_DEVICE_MAP", "auto"),
            torch_dtype=os.getenv("QWEN_TORCH_DTYPE", "auto"),
            attn_implementation=os.getenv("QWEN_ATTENTION_IMPL", ""),
            trust_remote_code=os.getenv("QWEN_TRUST_REMOTE_CODE", "0") == "1",
            max_new_tokens=_env_int("QWEN_MAX_NEW_TOKENS", 256),
            temperature=_env_float("QWEN_TEMPERATURE", 0.7),
            top_p=_env_float("QWEN_TOP_P", 0.9),
            history_turns=_env_int("AGENT_HISTORY_TURNS", 6),
            max_reply_chars=_env_int("AGENT_MAX_REPLY_CHARS", 900),
            system_prompt=os.getenv("AGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            db_path=os.getenv("AGENT_DB_PATH", "./agent_memory.sqlite3"),
        )


class ModelRegistry:
    def __init__(self, default_model_id: str, models: dict[str, ModelSpec]) -> None:
        if not models:
            raise ValueError("Model registry cannot be empty")

        self.models = models
        self.default_model_id = default_model_id if default_model_id in models else next(iter(models))

    @classmethod
    def from_settings(cls, settings: RuntimeSettings) -> "ModelRegistry":
        config_path = Path(settings.model_config_path)
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            default_model_id = data.get("default_model_id", settings.default_model_id)
            models: dict[str, ModelSpec] = {}
            for item in data.get("models", []):
                model_id = item["model_id"]
                model_path = item["model_path"]

                env_name = MODEL_PATH_ENV_BY_ID.get(model_id)
                if env_name and os.getenv(env_name):
                    model_path = os.getenv(env_name, model_path)
                elif model_id == default_model_id and os.getenv(LEGACY_MODEL_PATH_ENV):
                    model_path = os.getenv(LEGACY_MODEL_PATH_ENV, model_path)

                models[model_id] = ModelSpec(
                    model_id=model_id,
                    model_path=model_path,
                    label=item.get("label", model_id),
                    family=item.get("family", "qwen2.5"),
                    description=item.get("description", ""),
                )
            return cls(default_model_id=default_model_id, models=models)

        fallback_model_path = os.getenv(LEGACY_MODEL_PATH_ENV, "Qwen/Qwen2.5-0.5B-Instruct")
        fallback_model_id = settings.default_model_id
        fallback_models = {
            fallback_model_id: ModelSpec(
                model_id=fallback_model_id,
                model_path=fallback_model_path,
                label="Qwen2.5 Fallback",
                family="qwen2.5",
                description="Fallback single-model registry generated from environment.",
            )
        }
        return cls(default_model_id=fallback_model_id, models=fallback_models)

    def get(self, model_id: str | None) -> ModelSpec:
        resolved_model_id = model_id or self.default_model_id
        if resolved_model_id not in self.models:
            raise ValueError(f"Unsupported model_id: {resolved_model_id}")
        return self.models[resolved_model_id]

    def list_models(self, loaded_model_ids: set[str] | None = None) -> list[dict[str, Any]]:
        loaded_model_ids = loaded_model_ids or set()
        return [
            self.models[model_id].to_dict(loaded=model_id in loaded_model_ids)
            for model_id in sorted(self.models.keys())
        ]


class SessionStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path))
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _columns(self, conn: sqlite3.Connection) -> set[str]:
        rows = conn.execute("PRAGMA table_info(messages)").fetchall()
        return {row[1] for row in rows}

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    model_id TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            columns = self._columns(conn)
            if "model_id" not in columns:
                conn.execute("ALTER TABLE messages ADD COLUMN model_id TEXT NOT NULL DEFAULT ''")

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, model_id, id)
                """
            )

    def get_recent_messages(self, session_id: str, model_id: str, limit: int) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ? AND model_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, model_id, limit),
            ).fetchall()
        rows.reverse()
        return [{"role": role, "content": content} for role, content in rows]

    def append(self, session_id: str, model_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages(session_id, model_id, role, content)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, model_id, role, content),
            )

    def clear(self, session_id: str, model_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND model_id = ?",
                (session_id, model_id),
            )


class LocalQwenChat:
    def __init__(self, model_spec: ModelSpec, settings: RuntimeSettings) -> None:
        self.model_spec = model_spec
        self.settings = settings
        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _load(self) -> None:
        if self.loaded:
            return

        with self._load_lock:
            if self.loaded:
                return

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_spec.model_path,
                trust_remote_code=self.settings.trust_remote_code,
            )

            attn_impl = self.settings.attn_implementation
            if not attn_impl and torch.backends.mps.is_available() and self.settings.device != "cpu":
                attn_impl = "eager"

            load_kwargs: dict[str, Any] = {
                "torch_dtype": self.settings.torch_dtype,
                "trust_remote_code": self.settings.trust_remote_code,
            }
            if attn_impl:
                load_kwargs["attn_implementation"] = attn_impl

            if self.settings.device == "auto":
                load_kwargs["device_map"] = self.settings.device_map
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_spec.model_path,
                    **load_kwargs,
                ).eval()
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_spec.model_path,
                    **load_kwargs,
                ).to(torch.device(self.settings.device)).eval()

            if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def _model_device(self) -> torch.device:
        if hasattr(self._model, "device"):
            return self._model.device
        return next(self._model.parameters()).device

    def chat(self, messages: list[dict[str, str]]) -> str:
        self._load()

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer([prompt], return_tensors="pt")
        device = self._model_device()
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        generate_kwargs = {
            "max_new_tokens": self.settings.max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if self.settings.temperature > 0:
            generate_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.settings.temperature,
                    "top_p": self.settings.top_p,
                }
            )
        else:
            generate_kwargs["do_sample"] = False

        with self._infer_lock, torch.inference_mode():
            generated = self._model.generate(**inputs, **generate_kwargs)

        output_ids = generated[0][inputs["input_ids"].shape[1] :]
        reply = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        return _clip_text(reply, self.settings.max_reply_chars)


class LocalChatAgent:
    def __init__(self, settings: RuntimeSettings, registry: ModelRegistry | None = None) -> None:
        self.settings = settings
        self.registry = registry or ModelRegistry.from_settings(settings)
        self.store = SessionStore(settings.db_path)
        self._clients: dict[str, LocalQwenChat] = {}
        self._clients_lock = threading.Lock()

    def _get_client(self, model_id: str) -> LocalQwenChat:
        if model_id in self._clients:
            return self._clients[model_id]

        with self._clients_lock:
            if model_id in self._clients:
                return self._clients[model_id]

            model_spec = self.registry.get(model_id)
            self._clients[model_id] = LocalQwenChat(model_spec=model_spec, settings=self.settings)
            return self._clients[model_id]

    def list_models(self) -> list[dict[str, Any]]:
        return self.registry.list_models(loaded_model_ids=set(self._clients.keys()))

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

        answer = self._get_client(resolved_model.model_id).chat(messages)
        self.store.append(resolved_session_id, resolved_model.model_id, "user", text)
        self.store.append(resolved_session_id, resolved_model.model_id, "assistant", answer)
        return answer

    def status(self) -> dict[str, Any]:
        return {
            "default_model_id": self.registry.default_model_id,
            "supported_models": self.list_models(),
            "loaded_models": sorted(self._clients.keys()),
            "device": self.settings.device,
            "device_map": self.settings.device_map,
            "attn_implementation": self.settings.attn_implementation or (
                "eager" if torch.backends.mps.is_available() and self.settings.device != "cpu" else ""
            ),
            "db_path": self.settings.db_path,
            "history_turns": self.settings.history_turns,
        }


SimpleWechatAgent = LocalChatAgent


def build_agent_from_env() -> LocalChatAgent:
    return LocalChatAgent(RuntimeSettings.from_env())
