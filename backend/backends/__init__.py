from __future__ import annotations

from backend.backends.base import ChatBackend
from backend.backends.mlx_backend import MlxChatBackend
from backend.backends.transformers_backend import TransformersChatBackend
from backend.model_registry import ModelRegistry
from backend.runtime import RuntimeSettings


def create_backend(settings: RuntimeSettings, registry: ModelRegistry) -> ChatBackend:
    if settings.model_backend == "mlx":
        return MlxChatBackend(registry=registry, settings=settings)
    if settings.model_backend == "transformers":
        return TransformersChatBackend(registry=registry, settings=settings)
    raise ValueError(f"Unsupported MODEL_BACKEND: {settings.model_backend}")
