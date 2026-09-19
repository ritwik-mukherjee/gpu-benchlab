"""Numerical comparison of a candidate runtime's output against a reference.

Pure numpy, so the comparison itself can be unit-tested with known answers and
re-derived by anyone from stored outputs.

The pass criterion (docs/plans/phase-4-onnxruntime.md §3, fixed before the
pinned-weight run was made):

* identical shape and dtype, all values finite;
* elementwise ``|cand - ref| <= atol + rtol * |ref|`` with ``rtol = 1e-4`` and a
  scale-aware ``atol = 1e-4 * max|ref|``;
* identical top-1 class for every sample (for classifier outputs).

The tolerance is tight enough to *reject* reduced-precision execution: TF32 and
FP16 both have a 10-bit mantissa (unit roundoff 2^-11 ~= 4.9e-4). A negative
control in the test suite checks that an FP16 run of the same model fails it.
It must not be loosened to make a comparison pass.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DEFAULT_ATOL_SCALE",
    "DEFAULT_RTOL",
    "MEANINGFUL_REL_FRACTION",
    "OutputComparison",
    "compare_outputs",
]

DEFAULT_RTOL = 1e-4
DEFAULT_ATOL_SCALE = 1e-4
MEANINGFUL_REL_FRACTION = 1e-3
"""Relative error is only reported for elements with |ref| > this x max|ref|.

Near-zero reference values make relative error explode without meaning anything;
reporting it there would be noise dressed up as a finding.
"""


class OutputComparison(BaseModel):
    """Result of comparing one candidate output tensor to its reference."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    reasons: list[str] = Field(default_factory=list, description="Why it failed, if it did.")

    reference_shape: list[int]
    candidate_shape: list[int]
    reference_dtype: str
    candidate_dtype: str
    all_finite: bool

    rtol: float
    atol: float = Field(description="Absolute tolerance actually applied (scale-aware).")
    reference_max_abs: float | None = None
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    max_rel_error: float | None = Field(
        default=None,
        description="Over elements where |ref| > MEANINGFUL_REL_FRACTION * max|ref| only.",
    )
    violations: int | None = Field(default=None, description="Elements outside atol + rtol*|ref|.")
    worst_violation_ratio: float | None = Field(
        default=None,
        description="max(|cand-ref| / (atol + rtol*|ref|)); <= 1 means within tolerance. "
        "Shows how much headroom the pass had.",
    )
    top1_agreement: float | None = Field(
        default=None, description="Fraction of samples with identical argmax."
    )
    top5_overlap: float | None = Field(
        default=None, description="Mean overlap of the top-5 index sets, in [0, 1]."
    )


def _topk(x: npt.NDArray[np.floating], k: int) -> npt.NDArray[np.intp]:
    return np.argsort(-x, axis=-1, kind="stable")[..., :k]


def compare_outputs(
    reference: npt.NDArray[np.floating],
    candidate: npt.NDArray[np.floating],
    *,
    rtol: float = DEFAULT_RTOL,
    atol_scale: float = DEFAULT_ATOL_SCALE,
    classifier: bool = True,
) -> OutputComparison:
    """Compare ``candidate`` to ``reference``.

    Args:
        classifier: When true, outputs are ``[batch, classes]`` logits and top-1
            agreement is part of the pass criterion.
    """
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    reasons: list[str] = []
    if ref.shape != cand.shape:
        reasons.append(f"shape mismatch: reference {ref.shape} vs candidate {cand.shape}")
    if ref.dtype != cand.dtype:
        reasons.append(f"dtype mismatch: reference {ref.dtype} vs candidate {cand.dtype}")
    finite = bool(np.isfinite(ref).all() and np.isfinite(cand).all())
    if not finite:
        reasons.append("non-finite values present")

    if ref.shape != cand.shape or not finite or ref.size == 0:
        return OutputComparison(
            passed=False,
            reasons=reasons or ["empty output"],
            reference_shape=list(ref.shape),
            candidate_shape=list(cand.shape),
            reference_dtype=str(ref.dtype),
            candidate_dtype=str(cand.dtype),
            all_finite=finite,
            rtol=rtol,
            atol=0.0,
        )

    ref64 = ref.astype(np.float64)
    cand64 = cand.astype(np.float64)
    diff = np.abs(cand64 - ref64)
    ref_abs = np.abs(ref64)
    ref_max = float(ref_abs.max())
    atol = atol_scale * ref_max
    bound = atol + rtol * ref_abs

    meaningful = ref_abs > MEANINGFUL_REL_FRACTION * ref_max
    max_rel = float((diff[meaningful] / ref_abs[meaningful]).max()) if meaningful.any() else None

    over = diff > bound
    violations = int(over.sum())
    ratio = float((diff / np.where(bound > 0, bound, np.inf)).max()) if bound.size else 0.0
    if violations:
        reasons.append(
            f"{violations} element(s) outside atol + rtol*|ref| "
            f"(worst at {ratio:.3g}x the allowed error)"
        )

    top1 = top5 = None
    if classifier and ref.ndim == 2:
        top1 = float((ref.argmax(axis=1) == cand.argmax(axis=1)).mean())
        k = min(5, ref.shape[1])
        r5, c5 = _topk(ref64, k), _topk(cand64, k)
        top5 = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(r5, c5, strict=True)]))
        if top1 < 1.0:
            reasons.append(f"top-1 disagreement in {round((1 - top1) * ref.shape[0])} sample(s)")

    return OutputComparison(
        passed=not reasons,
        reasons=reasons,
        all_finite=finite,
        atol=atol,
        reference_max_abs=ref_max,
        max_abs_error=float(diff.max()),
        mean_abs_error=float(diff.mean()),
        max_rel_error=max_rel,
        violations=violations,
        worst_violation_ratio=ratio,
        top1_agreement=top1,
        top5_overlap=top5,
        reference_shape=list(ref.shape),
        candidate_shape=list(cand.shape),
        reference_dtype=str(ref.dtype),
        candidate_dtype=str(cand.dtype),
        rtol=rtol,
    )
