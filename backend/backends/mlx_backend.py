from __future__ import annotations

import threading
from time import perf_counter
from pathlib import Path
from typing import Any

from backend.backends.base import BackendChatResult, BackendDebugMetrics
from backend.model_registry import ModelRegistry, ModelSpec
from backend.runtime import RuntimeSettings


def _clip_text(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 12].rstrip() + "\n\n[内容已截断]"


def _count_tokens(tokenizer: Any, text: str) -> int | None:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return len(encode(text))
        except Exception:
            pass

    try:
        encoded = tokenizer(text)
        if isinstance(encoded, dict) and "input_ids" in encoded:
            return len(encoded["input_ids"])
    except Exception:
        return None

    return None


def _load_mlx_modules() -> tuple[Any, Any, Any]:
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError(
            "mlx-lm is not installed. Install it with `pip install -r requirements-mlx.txt` in your active environment."
        ) from exc

    return generate, load, make_sampler


class MlxModelClient:
    def __init__(self, model_spec: ModelSpec, settings: RuntimeSettings) -> None:
        self.model_spec = model_spec
        self.settings = settings
        self._model = None
        self._tokenizer = None
        self._source = self._resolve_model_source()
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def source(self) -> str:
        return self._source

    def _resolve_model_source(self) -> str:
        configured_path = Path(self.model_spec.model_path)
        if configured_path.exists():
            return str(configured_path)

        base_name = self.model_spec.model_path.split("/")[-1]
        local_dir_name = base_name if base_name.endswith("-4bit") else f"{base_name}-4bit"
        local_candidate = Path(self.settings.mlx_local_root) / local_dir_name
        if self.settings.mlx_prefer_local and local_candidate.exists():
            return str(local_candidate)

        if self.model_spec.model_path.startswith("mlx-community/") or base_name.endswith("-4bit") or base_name.endswith("-MLX"):
            return self.model_spec.model_path

        return f"mlx-community/{local_dir_name}"

    def _load(self) -> None:
        if self.loaded:
            return

        with self._load_lock:
            if self.loaded:
                return

            _, load, _ = _load_mlx_modules()
            tokenizer_config: dict[str, object] | None = None
            if self.settings.mlx_eos_token:
                tokenizer_config = {"eos_token": self.settings.mlx_eos_token}

            self._model, self._tokenizer = load(self.source, tokenizer_config=tokenizer_config)

    def _build_prompt(self, messages: list[dict[str, str]]) -> str:
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        chat_template = getattr(self._tokenizer, "chat_template", None)

        if callable(apply_chat_template) and chat_template is not None:
            return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return messages[-1]["content"]

    def chat(self, messages: list[dict[str, str]], *, capture_debug: bool = False) -> BackendChatResult:
        total_started = perf_counter()
        self._load()

        generate, _, make_sampler = _load_mlx_modules()
        prompt = self._build_prompt(messages)
        prompt_tokens = _count_tokens(self._tokenizer, prompt) if capture_debug else None
        sampler = None
        if self.settings.temperature > 0 or self.settings.top_p > 0:
            sampler = make_sampler(temp=self.settings.temperature, top_p=self.settings.top_p)

        generation_started = perf_counter()
        with self._infer_lock:
            response = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self.settings.max_new_tokens,
                sampler=sampler,
                verbose=False,
            )
        generation_elapsed = perf_counter() - generation_started
        total_elapsed = perf_counter() - total_started
        clipped = _clip_text(response, self.settings.max_reply_chars)

        if not capture_debug:
            return BackendChatResult(text=clipped)

        generated_tokens = _count_tokens(self._tokenizer, response)
        tokens_per_second = None
        if generated_tokens and generation_elapsed > 0:
            tokens_per_second = generated_tokens / generation_elapsed

        return BackendChatResult(
            text=clipped,
            debug=BackendDebugMetrics(
                backend="mlx",
                model_id=self.model_spec.model_id,
                latency_ms=total_elapsed * 1000,
                generation_ms=generation_elapsed * 1000,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                tokens_per_second=tokens_per_second,
                model_source=self.source,
            ),
        )


class MlxChatBackend:
    backend_name = "mlx"

    def __init__(self, registry: ModelRegistry, settings: RuntimeSettings) -> None:
        self.registry = registry
        self.settings = settings
        self._clients: dict[str, MlxModelClient] = {}
        self._clients_lock = threading.Lock()

    def _get_client(self, model_id: str) -> MlxModelClient:
        if model_id in self._clients:
            return self._clients[model_id]

        with self._clients_lock:
            if model_id in self._clients:
                return self._clients[model_id]

            model_spec = self.registry.get(model_id)
            self._clients[model_id] = MlxModelClient(model_spec=model_spec, settings=self.settings)
            return self._clients[model_id]

    def chat(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        *,
        capture_debug: bool = False,
    ) -> BackendChatResult:
        return self._get_client(model_id).chat(messages, capture_debug=capture_debug)

    def loaded_model_ids(self) -> set[str]:
        return {model_id for model_id, client in self._clients.items() if client.loaded}

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "loaded_models": sorted(self.loaded_model_ids()),
            "mlx_local_root": self.settings.mlx_local_root,
            "mlx_prefer_local": self.settings.mlx_prefer_local,
            "mlx_eos_token": self.settings.mlx_eos_token,
            "model_sources": {
                model_id: client.source for model_id, client in sorted(self._clients.items())
            },
        }
