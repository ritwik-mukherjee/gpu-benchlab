"""Phase 5B: prove both backends feed the GPU the same input bytes.

`tests/unit/test_input_identity.py` proves this on the CPU device, where the array a
backend hands its runtime is directly readable. On CUDA the inputs are device-resident
(a `cuda` tensor for PyTorch, an `OrtValue` bound through IOBinding for ORT), so this
script drives the real backends on `cuda:0`, copies each device input back to the host,
and compares the bytes with the canonical array.

The round trip is lossless for float32, so an exact comparison is the right one: any
difference means the runtimes were not benchmarked on the same input.

Checks:
  I1  PyTorch's device tensor equals core.inputs.synthetic_input exactly
  I2  ORT's device-resident OrtValue equals it exactly
  I3  the two backends' device inputs are byte-identical to each other
  I4  both record input_generator / input_seed provenance in their settings
  I5  both ran with device_kind cuda and the CUDA EP (no CPU fallback)

Usage: python analysis/gpu_validation/input_identity_gpu.py <out_dir> [--batch N]...
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from _common import INFO, Checks, require_cuda_torch

CUDA_EP = "CUDAExecutionProvider"


def device_input(backend_name: str, batch: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Run a backend's real lifecycle on cuda:0 and copy its device input back."""

    from gpu_benchlab.cli.run_cmd import build_backend
    from gpu_benchlab.core.config import parse_config
    from gpu_benchlab.hardware.detect import detect_environment

    options: dict[str, Any] = {"allow_export": False} if backend_name == "onnxruntime" else {}
    config = parse_config(
        {
            "name": f"phase5b-input-identity-{backend_name}-b{batch}",
            "backend": backend_name,
            "device": "cuda:0",
            "precision": "fp32",
            "batch_size": batch,
            "model": {"name": "resnet50", "weights": "IMAGENET1K_V2", "seed": seed},
            "benchmark": {"warmup_iterations": 1, "measurement_iterations": 1},
            "backend_options": options,
        }
    )
    backend = build_backend(config, seed=seed)
    try:
        backend.validate(config, detect_environment(include_frameworks=False))
        backend.load()
        backend.build()
        prepared = backend.prepare(config)
        settings = dict(backend.descriptor.settings)
        kind = backend.descriptor.device_kind.value
        if hasattr(prepared, "detach"):  # PyTorch: a cuda tensor
            host = prepared.detach().cpu().numpy()
        elif isinstance(prepared, dict):  # ORT without IOBinding: a host feed
            host = next(iter(prepared.values()))
        else:  # ORT with IOBinding: read the bound device input back
            bound = prepared.get_inputs()[0] if hasattr(prepared, "get_inputs") else None
            if bound is None:
                raise AssertionError(
                    f"{backend_name} prepared a {type(prepared).__name__} this script "
                    "cannot read; extend it rather than skipping the check."
                )
            host = bound.numpy()
        return np.ascontiguousarray(host), {**settings, "device_kind": kind}
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--batch", action="append", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    batches = args.batch or [1, 8]

    require_cuda_torch()
    from gpu_benchlab.core.inputs import INPUT_GENERATOR, synthetic_input

    checks = Checks("input_identity_gpu")
    details: dict[str, Any] = {"generator": INPUT_GENERATOR, "seed": args.seed, "cases": {}}

    for batch in batches:
        canonical = synthetic_input((batch, 3, 224, 224), args.seed)
        digest = hashlib.sha256(canonical.tobytes()).hexdigest()
        arrays: dict[str, np.ndarray] = {}
        for name in ("pytorch", "onnxruntime"):
            array, settings = device_input(name, batch, args.seed)
            arrays[name] = array
            checks.expect(
                f"I{1 if name == 'pytorch' else 2}.b{batch}",
                array.dtype == np.float32
                and array.shape == canonical.shape
                and np.array_equal(array, canonical),
                f"{name}: device input equals the canonical array exactly (batch {batch})",
                sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                canonical_sha256=digest,
            )
            checks.expect(
                f"I4.{name}.b{batch}",
                settings.get("input_generator") == INPUT_GENERATOR
                and settings.get("input_seed") == args.seed,
                f"{name}: records input_generator and input_seed",
                input_generator=settings.get("input_generator"),
                input_seed=settings.get("input_seed"),
            )
            on_cuda = settings.get("device_kind") == "cuda" and (
                name == "pytorch" or CUDA_EP in str(settings.get("active_providers", ""))
            )
            checks.expect(
                f"I5.{name}.b{batch}",
                on_cuda,
                f"{name}: executed on CUDA, no CPU fallback (batch {batch})",
                device_kind=settings.get("device_kind"),
                active_providers=settings.get("active_providers"),
                node_placement=settings.get(f"node_placement.{CUDA_EP}"),
                io_binding=settings.get("io_binding"),
            )
        checks.expect(
            f"I3.b{batch}",
            np.array_equal(arrays["pytorch"], arrays["onnxruntime"]),
            f"both backends' device inputs are byte-identical (batch {batch})",
        )
        details["cases"][f"b{batch}"] = {
            "canonical_sha256": digest,
            "shape": list(canonical.shape),
            "dtype": str(canonical.dtype),
        }

    checks.add("provenance", INFO, "canonical input digests", **details)
    checks.finish(args.out_dir, details)


if __name__ == "__main__":
    main()
