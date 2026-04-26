from __future__ import annotations

import threading
from time import perf_counter
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.backends.base import BackendChatResult, BackendDebugMetrics
from backend.model_registry import ModelRegistry, ModelSpec
from backend.runtime import RuntimeSettings


def _clip_text(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 12].rstrip() + "\n\n[内容已截断]"


class TransformersModelClient:
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

    def chat(self, messages: list[dict[str, str]], *, capture_debug: bool = False) -> BackendChatResult:
        total_started = perf_counter()
        self._load()

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer([prompt], return_tensors="pt")
        device = self._model_device()
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1]) if capture_debug else None

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

        generation_started = perf_counter()
        with self._infer_lock, torch.inference_mode():
            generated = self._model.generate(**inputs, **generate_kwargs)
        generation_elapsed = perf_counter() - generation_started
        total_elapsed = perf_counter() - total_started

        output_ids = generated[0][inputs["input_ids"].shape[1] :]
        reply = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        clipped = _clip_text(reply, self.settings.max_reply_chars)

        if not capture_debug:
            return BackendChatResult(text=clipped)

        generated_tokens = int(output_ids.shape[0])
        tokens_per_second = None
        if generated_tokens > 0 and generation_elapsed > 0:
            tokens_per_second = generated_tokens / generation_elapsed

        return BackendChatResult(
            text=clipped,
            debug=BackendDebugMetrics(
                backend="transformers",
                model_id=self.model_spec.model_id,
                latency_ms=total_elapsed * 1000,
                generation_ms=generation_elapsed * 1000,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                tokens_per_second=tokens_per_second,
                model_source=self.model_spec.model_path,
            ),
        )


class TransformersChatBackend:
    backend_name = "transformers"

    def __init__(self, registry: ModelRegistry, settings: RuntimeSettings) -> None:
        self.registry = registry
        self.settings = settings
        self._clients: dict[str, TransformersModelClient] = {}
        self._clients_lock = threading.Lock()

    def _get_client(self, model_id: str) -> TransformersModelClient:
        if model_id in self._clients:
            return self._clients[model_id]

        with self._clients_lock:
            if model_id in self._clients:
                return self._clients[model_id]

            model_spec = self.registry.get(model_id)
            self._clients[model_id] = TransformersModelClient(model_spec=model_spec, settings=self.settings)
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
        attn_impl = self.settings.attn_implementation or (
            "eager" if torch.backends.mps.is_available() and self.settings.device != "cpu" else ""
        )
        return {
            "backend": self.backend_name,
            "loaded_models": sorted(self.loaded_model_ids()),
            "device": self.settings.device,
            "device_map": self.settings.device_map,
            "torch_dtype": self.settings.torch_dtype,
            "attn_implementation": attn_impl,
        }
