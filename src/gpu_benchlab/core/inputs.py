"""Backend-independent synthetic inputs.

One definition of "the input for seed S and shape X", in numpy, so every runtime
that uses it receives bit-identical float32 values. The ONNX Runtime backend and
the PyTorch-vs-ONNX correctness check use it.

The PyTorch backend (Phase 3) still draws its benchmark inputs with
``torch.randn``; unifying it is a prerequisite for the first cross-backend latency
comparison (docs/plans/phase-4-onnxruntime.md §4).

Values are standard normal: close to the distribution of ImageNet images after
the usual mean/std normalisation, which is the input range these models expect.
Preprocessing is deliberately outside the model: every consumer of these arrays
receives the already-normalised tensor, so preprocessing is identical by
construction.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["INPUT_GENERATOR", "synthetic_input"]

INPUT_GENERATOR = "numpy.random.default_rng(seed).standard_normal(shape, dtype=float32)"
"""Recorded verbatim on results so the exact input can be regenerated."""


def synthetic_input(shape: tuple[int, ...], seed: int) -> npt.NDArray[np.float32]:
    """Deterministic standard-normal float32 array (numpy PCG64)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape, dtype=np.float32)
