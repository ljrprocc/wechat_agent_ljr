from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_SYSTEM_PROMPT = (
    "你是一个本地部署的 Qwen 助手。"
    "请优先用中文回答，尽量简洁、准确；信息不足时直接说明不确定。"
)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeSettings:
    model_backend: str
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
    mlx_local_root: str
    mlx_prefer_local: bool
    mlx_eos_token: str

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            model_backend=os.getenv("MODEL_BACKEND", "mlx"),
            model_config_path=os.getenv("QWEN_MODEL_CONFIG_PATH", "./config/models/qwen2.5.json"),
            default_model_id=os.getenv("QWEN_DEFAULT_MODEL_ID", "qwen2.5-0.5b"),
            device=os.getenv("QWEN_DEVICE", "auto"),
            device_map=os.getenv("QWEN_DEVICE_MAP", "auto"),
            torch_dtype=os.getenv("QWEN_TORCH_DTYPE", "auto"),
            attn_implementation=os.getenv("QWEN_ATTENTION_IMPL", ""),
            trust_remote_code=_env_bool("QWEN_TRUST_REMOTE_CODE", False),
            max_new_tokens=_env_int("QWEN_MAX_NEW_TOKENS", 256),
            temperature=_env_float("QWEN_TEMPERATURE", 0.7),
            top_p=_env_float("QWEN_TOP_P", 0.9),
            history_turns=_env_int("AGENT_HISTORY_TURNS", 6),
            max_reply_chars=_env_int("AGENT_MAX_REPLY_CHARS", 900),
            system_prompt=os.getenv("AGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            db_path=os.getenv("AGENT_DB_PATH", "./agent_memory.sqlite3"),
            mlx_local_root=os.getenv("MLX_QWEN_LOCAL_ROOT", "./mlx"),
            mlx_prefer_local=_env_bool("MLX_QWEN_PREFER_LOCAL", True),
            mlx_eos_token=os.getenv("MLX_QWEN_EOS_TOKEN", "<|im_end|>").strip(),
        )
