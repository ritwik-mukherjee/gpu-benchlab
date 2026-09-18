"""Model registry and weights management."""

from __future__ import annotations

from gpu_benchlab.models.registry import (
    RANDOM_WEIGHTS,
    ModelSpec,
    WeightsSpec,
    get_model,
    list_models,
    register_model,
    unregister_model,
)

__all__ = [
    "RANDOM_WEIGHTS",
    "ModelSpec",
    "WeightsSpec",
    "get_model",
    "list_models",
    "register_model",
    "unregister_model",
]
