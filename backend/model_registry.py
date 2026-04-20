from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.runtime import RuntimeSettings


MODEL_PATH_ENV_BY_ID = {
    "qwen2.5-0.5b": "QWEN25_05B_MODEL_PATH",
    "qwen2.5-3b": "QWEN25_3B_MODEL_PATH",
}
LEGACY_MODEL_PATH_ENV = "QWEN_MODEL_PATH"


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
