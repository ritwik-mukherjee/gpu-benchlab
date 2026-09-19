"""Load a registry model as a verified PyTorch module, outside any benchmark run.

Used by ONNX export and by the PyTorch-vs-ONNX correctness check, which both need
the exact reference model but are not benchmarks. It applies the same guarantees
as the Phase 3 PyTorch backend -- architecture class, parameter count, pinned
weights with SHA-256, ``weights_only=True`` -- without changing that backend.
(The small overlap is deliberate: Phase 3 is left untouched.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gpu_benchlab.core.backend import ModelInfo
from gpu_benchlab.core.errors import BackendError, UnavailableError
from gpu_benchlab.models.registry import RANDOM_WEIGHTS, get_model
from gpu_benchlab.models.weights import ensure_weights

__all__ = ["LoadedModel", "load_reference_model"]


@dataclass(frozen=True)
class LoadedModel:
    module: Any
    """``torch.nn.Module`` in eval mode, FP32, on CPU."""

    info: ModelInfo


def load_reference_model(
    name: str, weights: str | None = None, *, seed: int = 0, allow_download: bool = True
) -> LoadedModel:
    """Build a registry model in FP32 eval mode on CPU, verified against the registry.

    Raises:
        UnavailableError: torch/torchvision missing, or weights cannot be obtained.
        BackendError: wrong architecture, wrong parameter count, or bad weights.
    """
    try:
        import torch
    except ImportError as exc:
        raise UnavailableError("PyTorch is required to load the reference model.") from exc

    spec = get_model(name)
    weights_id = spec.resolve_weights(weights)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        module = spec.build()

    architecture = type(module).__name__
    if architecture != spec.architecture:
        raise BackendError(
            f"Registry model {name!r} built a {architecture}, expected {spec.architecture}."
        )

    weights_url = weights_sha = None
    if weights_id != RANDOM_WEIGHTS:
        resolved = ensure_weights(spec.weights[weights_id], allow_download=allow_download)
        state = torch.load(resolved.path, map_location="cpu", weights_only=True)
        module.load_state_dict(state, strict=True)
        weights_url = spec.weights[weights_id].url
        weights_sha = resolved.sha256

    count = sum(int(p.numel()) for p in module.parameters())
    if count != spec.expected_parameters:
        raise BackendError(
            f"Model {name!r} has {count:,} parameters; expected {spec.expected_parameters:,}."
        )

    module = module.eval().to(dtype=torch.float32)
    info = ModelInfo(
        name=spec.name,
        architecture=architecture,
        source=spec.source,
        weights=weights_id,
        weights_url=weights_url,
        weights_sha256=weights_sha,
        random_init_seed=seed if weights_id == RANDOM_WEIGHTS else None,
        parameter_count=count,
        expected_parameter_count=spec.expected_parameters,
    )
    return LoadedModel(module=module, info=info)
