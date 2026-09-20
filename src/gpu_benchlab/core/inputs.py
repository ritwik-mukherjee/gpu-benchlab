"""Backend-independent synthetic inputs.

One definition of "the input for seed S and shape X", in numpy, so every runtime
that uses it receives bit-identical float32 values. **Every executing backend takes
its benchmark input from here** -- PyTorch, ONNX Runtime, and the PyTorch-vs-ONNX
correctness check -- which is what makes a cross-backend comparison differ only in
the runtime under test. `tests/unit/test_input_identity.py` fails if a backend
deviates.

Until 2026-09-20 the PyTorch backend drew from ``torch.randn`` instead: same shape,
dtype and distribution, different values for the same seed. Results recorded before
that carry ``input_generator`` saying which stream produced them, so they are not
silently mixed with later ones.

A backend may still convert these values: casting to a requested precision
(fp16/bf16), moving them to a device, or changing memory format. Those are
properties of the run being measured, not different inputs.

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
