"""Model registry: the one place that knows what a model name means.

Each entry pins an architecture, its expected parameter count, its input and
output shapes, and the exact weights files it may load. See ``docs/models.md`` for
why each model was selected.

Nothing here imports torch or torchvision at module import time. Builders import
lazily, so the registry (and `gpu-bench models`) works without torch installed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gpu_benchlab.core.errors import ConfigurationError

__all__ = [
    "RANDOM_WEIGHTS",
    "ModelSpec",
    "WeightsSpec",
    "get_model",
    "list_models",
    "register_model",
    "unregister_model",
]

RANDOM_WEIGHTS = "random"


@dataclass(frozen=True)
class WeightsSpec:
    """A pinned weights file.

    Attributes:
        url: Exact download URL. Pinned here rather than read from the library
            at runtime, so a library upgrade cannot silently change the weights.
        sha256_prefix: The hash prefix the publisher embeds in the filename.
            The full SHA-256 of the downloaded file must start with it.
    """

    url: str
    sha256_prefix: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to build, verify and describe a model."""

    name: str
    family: str
    architecture: str
    """Expected class name of the built module, checked after construction."""

    build: Callable[[], Any]
    """Constructs the architecture with no weights and no downloads."""

    input_shape: tuple[int, ...]
    """Per-sample input shape, excluding batch."""

    output_shape: tuple[int, ...]
    """Per-sample output shape, excluding batch."""

    expected_parameters: int
    default_weights: str
    weights: dict[str, WeightsSpec] = field(default_factory=dict)
    source: str = ""
    license_note: str = ""
    test_only: bool = False

    def resolve_weights(self, requested: str | None) -> str:
        """Turn a requested weights id into a pinned one.

        ``None`` resolves to this registry's explicit default -- never to a
        library's moving ``DEFAULT`` alias.

        Raises:
            ConfigurationError: for an unknown weights id.
        """
        weights = requested or self.default_weights
        if weights == RANDOM_WEIGHTS or weights in self.weights:
            return weights
        valid = ", ".join([*self.weights, RANDOM_WEIGHTS])
        raise ConfigurationError(
            f"Unknown weights {weights!r} for model {self.name!r}. Valid: {valid}."
        )


def _build_resnet50() -> Any:
    from torchvision.models import resnet50

    return resnet50(weights=None)


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> None:
    """Add a model. Re-registering an existing name is an error."""
    if spec.name in _REGISTRY:
        raise ValueError(f"Model {spec.name!r} is already registered.")
    _REGISTRY[spec.name] = spec


def unregister_model(name: str) -> None:
    """Remove a model (used by tests that register temporary fixtures)."""
    _REGISTRY.pop(name, None)


def get_model(name: str) -> ModelSpec:
    """Look up a model.

    Raises:
        ConfigurationError: if the model is not registered.
    """
    spec = _REGISTRY.get(name)
    if spec is None:
        available = ", ".join(sorted(n for n, s in _REGISTRY.items() if not s.test_only))
        raise ConfigurationError(f"Unknown model {name!r}. Registered models: {available}.")
    return spec


def list_models(*, include_test_only: bool = False) -> list[ModelSpec]:
    return [s for s in _REGISTRY.values() if include_test_only or not s.test_only]


register_model(
    ModelSpec(
        name="resnet50",
        family="vision",
        architecture="ResNet",
        build=_build_resnet50,
        input_shape=(3, 224, 224),
        output_shape=(1000,),
        # Published in the torchvision docs; asserted after every load so the wrong
        # architecture can never be benchmarked under this name.
        expected_parameters=25_557_032,
        default_weights="IMAGENET1K_V2",
        weights={
            # URLs verified against torchvision 0.29's ResNet50_Weights enum.
            "IMAGENET1K_V1": WeightsSpec(
                url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
                sha256_prefix="0676ba61",
            ),
            "IMAGENET1K_V2": WeightsSpec(
                url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
                sha256_prefix="11ad3fa6",
            ),
        },
        source="torchvision.models.resnet50 (ResNet v1.5)",
        license_note=(
            "torchvision code: BSD-3-Clause. Pretrained weights may carry ImageNet-derived "
            "terms (non-commercial research); they are downloaded by the user and never "
            "redistributed by this project. See docs/models.md."
        ),
    )
)
