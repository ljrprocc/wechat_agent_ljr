from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "你是一个部署在微信里的本地 Qwen 助手。"
    "请优先用中文回答，尽量简洁、准确；信息不足时直接说明不确定。"
)


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
class AgentSettings:
    model_path: str
    device_map: str
    torch_dtype: str
    trust_remote_code: bool
    max_new_tokens: int
    temperature: float
    top_p: float
    history_turns: int
    max_reply_chars: int
    system_prompt: str
    db_path: str

    @classmethod
    def from_env(cls) -> "AgentSettings":
        return cls(
            model_path=os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct"),
            device_map=os.getenv("QWEN_DEVICE_MAP", "auto"),
            torch_dtype=os.getenv("QWEN_TORCH_DTYPE", "auto"),
            trust_remote_code=os.getenv("QWEN_TRUST_REMOTE_CODE", "0") == "1",
            max_new_tokens=_env_int("QWEN_MAX_NEW_TOKENS", 256),
            temperature=_env_float("QWEN_TEMPERATURE", 0.7),
            top_p=_env_float("QWEN_TOP_P", 0.9),
            history_turns=_env_int("AGENT_HISTORY_TURNS", 6),
            max_reply_chars=_env_int("AGENT_MAX_REPLY_CHARS", 900),
            system_prompt=os.getenv("AGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            db_path=os.getenv("AGENT_DB_PATH", "./agent_memory.sqlite3"),
        )


class SessionStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path))
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id)
                """
            )

    def get_recent_messages(self, session_id: str, limit: int) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        rows.reverse()
        return [{"role": role, "content": content} for role, content in rows]

    def append(self, session_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages(session_id, role, content)
                VALUES (?, ?, ?)
                """,
                (session_id, role, content),
            )

    def clear(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


class LocalQwenChat:
    def __init__(self, settings: AgentSettings) -> None:
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
                self.settings.model_path,
                trust_remote_code=self.settings.trust_remote_code,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.settings.model_path,
                torch_dtype=self.settings.torch_dtype,
                device_map=self.settings.device_map,
                trust_remote_code=self.settings.trust_remote_code,
            ).eval()

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


class SimpleWechatAgent:
    def __init__(self, settings: AgentSettings) -> None:
        self.settings = settings
        self.store = SessionStore(settings.db_path)
        self.llm = LocalQwenChat(settings)

    def reply(self, session_id: str, user_text: str) -> str:
        text = user_text.strip()
        if not text:
            return "我收到的是空消息，可以再发一次。"

        if text in {"/reset", "/clear", "重置对话", "清空对话"}:
            self.store.clear(session_id)
            return "当前会话记忆已清空。"

        history = self.store.get_recent_messages(
            session_id,
            limit=self.settings.history_turns * 2,
        )
        messages = [{"role": "system", "content": self.settings.system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": text})

        answer = self.llm.chat(messages)
        self.store.append(session_id, "user", text)
        self.store.append(session_id, "assistant", answer)
        return answer

    def status(self) -> dict[str, object]:
        return {
            "model_path": self.settings.model_path,
            "model_loaded": self.llm.loaded,
            "db_path": self.settings.db_path,
            "history_turns": self.settings.history_turns,
        }


def build_agent_from_env() -> SimpleWechatAgent:
    return SimpleWechatAgent(AgentSettings.from_env())
