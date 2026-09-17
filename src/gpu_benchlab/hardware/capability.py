"""Mapping from NVIDIA compute capability to architecture and precision support.

This module is deliberately pure (no NVML, no CUDA, no I/O) so that the whole
capability matrix is unit-testable on a machine with no NVIDIA GPU.

Sources for the thresholds below are the CUDA C++ Programming Guide's
"Compute Capabilities" tables and the per-architecture tuning guides. Where a
precision is supported in two different ways (emulated / native ALU / tensor
core), that distinction is preserved rather than flattened into a boolean,
because it materially changes benchmark interpretation: FP16 on a Pascal card
is not the same thing as FP16 on a card with tensor cores.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Precision",
    "PrecisionSupport",
    "architecture_name",
    "precision_support",
    "supported_precisions",
]


class Precision(str, Enum):
    """A numeric format a benchmark can request."""

    FP32 = "fp32"
    TF32 = "tf32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    FP8 = "fp8"
    INT4 = "int4"
    FP4 = "fp4"


@dataclass(frozen=True)
class PrecisionSupport:
    """Whether a precision is usable on a given compute capability, and how.

    Attributes:
        precision: The format in question.
        supported: Whether the hardware can execute this format at all.
        tensor_core: Whether dedicated tensor-core acceleration exists for it.
            A precision can be ``supported`` without being ``tensor_core``
            accelerated, in which case expect little or no speedup over FP32.
        note: Human-readable justification, surfaced in `gpu-bench hardware`.
    """

    precision: Precision
    supported: bool
    tensor_core: bool
    note: str


# (major, minor) -> marketing architecture name.
# Ordered most-specific first; falls back to the major-version family.
_ARCH_BY_CC: dict[tuple[int, int], str] = {
    (3, 0): "Kepler",
    (3, 5): "Kepler",
    (3, 7): "Kepler",
    (5, 0): "Maxwell",
    (5, 2): "Maxwell",
    (5, 3): "Maxwell",
    (6, 0): "Pascal",
    (6, 1): "Pascal",
    (6, 2): "Pascal",
    (7, 0): "Volta",
    (7, 2): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 7): "Ampere",
    (8, 9): "Ada Lovelace",
    (9, 0): "Hopper",
    (10, 0): "Blackwell",
    (10, 1): "Blackwell",
    (10, 3): "Blackwell",
    (11, 0): "Blackwell",
    (12, 0): "Blackwell",
    (12, 1): "Blackwell",
}

_ARCH_BY_MAJOR: dict[int, str] = {
    3: "Kepler",
    5: "Maxwell",
    6: "Pascal",
    7: "Volta/Turing",
    8: "Ampere/Ada Lovelace",
    9: "Hopper",
    10: "Blackwell",
    11: "Blackwell",
    12: "Blackwell",
}


def architecture_name(major: int, minor: int) -> str:
    """Return the marketing architecture name for a compute capability.

    Unknown newer capabilities degrade to a best-effort family name rather than
    raising, so that a future GPU is still benchmarkable.
    """
    exact = _ARCH_BY_CC.get((major, minor))
    if exact is not None:
        return exact
    family = _ARCH_BY_MAJOR.get(major)
    if family is not None:
        return f"{family} (unrecognised SM {major}.{minor})"
    return f"Unknown (SM {major}.{minor})"


def _cc(major: int, minor: int) -> float:
    """Collapse a compute capability to a comparable number (8, 9 -> 8.9)."""
    return major + minor / 10.0


def precision_support(major: int, minor: int) -> list[PrecisionSupport]:
    """Describe support for every known precision on this compute capability.

    Returns one entry per :class:`Precision`, including unsupported ones, so
    that callers can explain *why* a configuration is unsupported instead of
    silently dropping it.
    """
    cc = _cc(major, minor)
    out: list[PrecisionSupport] = []

    out.append(
        PrecisionSupport(
            Precision.FP32,
            supported=True,
            tensor_core=False,
            note="Baseline format; supported on every CUDA GPU.",
        )
    )

    out.append(
        PrecisionSupport(
            Precision.TF32,
            supported=cc >= 8.0,
            tensor_core=cc >= 8.0,
            note=(
                "TF32 tensor-core path for FP32 math (SM 8.0+)."
                if cc >= 8.0
                else "Requires Ampere or newer (SM 8.0+)."
            ),
        )
    )

    # Native half-precision arithmetic appeared with SM 5.3; tensor cores with SM 7.0.
    out.append(
        PrecisionSupport(
            Precision.FP16,
            supported=cc >= 5.3,
            tensor_core=cc >= 7.0,
            note=(
                "FP16 with tensor-core acceleration (SM 7.0+)."
                if cc >= 7.0
                else "Native FP16 arithmetic but no tensor cores; expect little speedup."
                if cc >= 5.3
                else "Requires SM 5.3+."
            ),
        )
    )

    out.append(
        PrecisionSupport(
            Precision.BF16,
            supported=cc >= 8.0,
            tensor_core=cc >= 8.0,
            note=(
                "BF16 tensor cores (SM 8.0+)."
                if cc >= 8.0
                else "Requires Ampere or newer (SM 8.0+)."
            ),
        )
    )

    # DP4A integer dot-product from SM 6.1; INT8 tensor cores from SM 7.5.
    out.append(
        PrecisionSupport(
            Precision.INT8,
            supported=cc >= 6.1,
            tensor_core=cc >= 7.5,
            note=(
                "INT8 tensor cores (SM 7.5+)."
                if cc >= 7.5
                else "DP4A integer path only, no INT8 tensor cores."
                if cc >= 6.1
                else "Requires SM 6.1+."
            ),
        )
    )

    # FP8 (E4M3/E5M2): Hopper SM 9.0 and Ada SM 8.9.
    fp8 = cc >= 8.9
    out.append(
        PrecisionSupport(
            Precision.FP8,
            supported=fp8,
            tensor_core=fp8,
            note=(
                "FP8 (E4M3/E5M2) tensor cores."
                if fp8
                else "Requires Ada Lovelace (SM 8.9) or Hopper (SM 9.0)+."
            ),
        )
    )

    out.append(
        PrecisionSupport(
            Precision.INT4,
            supported=7.5 <= cc < 10.0,
            tensor_core=7.5 <= cc < 10.0,
            note=(
                "INT4 tensor-core path (SM 7.5-9.x); deprecated on newer parts."
                if 7.5 <= cc < 10.0
                else "Not exposed on this architecture."
            ),
        )
    )

    # FP4 (NVFP4/MXFP4) is a Blackwell-generation feature.
    fp4 = cc >= 10.0
    out.append(
        PrecisionSupport(
            Precision.FP4,
            supported=fp4,
            tensor_core=fp4,
            note="FP4 tensor cores (Blackwell)." if fp4 else "Requires Blackwell (SM 10.0+).",
        )
    )

    return out


def supported_precisions(major: int, minor: int) -> list[Precision]:
    """Return only the precisions this compute capability can execute."""
    return [p.precision for p in precision_support(major, minor) if p.supported]
