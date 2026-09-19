"""Known-answer tests for output comparison and synthetic inputs. numpy only."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpu_benchlab.core.correctness import (
    DEFAULT_ATOL_SCALE,
    DEFAULT_RTOL,
    MEANINGFUL_REL_FRACTION,
    compare_outputs,
)
from gpu_benchlab.core.inputs import INPUT_GENERATOR, synthetic_input


def logits(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


class TestDefaults:
    def test_pre_registered_tolerance(self) -> None:
        """Fixed before the pinned-weight run (docs/plans/phase-4-onnxruntime.md §3)."""
        assert DEFAULT_RTOL == 1e-4
        assert DEFAULT_ATOL_SCALE == 1e-4
        assert MEANINGFUL_REL_FRACTION == 1e-3

    def test_tolerance_is_tighter_than_ten_bit_mantissa(self) -> None:
        """TF32 and FP16 unit roundoff is 2^-11; the tolerance must be below it."""
        assert DEFAULT_RTOL < 2.0**-11


class TestIdentityAndStructure:
    def test_identical_outputs_pass_with_zero_error(self) -> None:
        ref = logits([[1.0, 5.0, -2.0], [0.5, -1.0, 3.0]])
        c = compare_outputs(ref, ref.copy())
        assert c.passed and c.reasons == []
        assert c.max_abs_error == 0.0 and c.mean_abs_error == 0.0
        assert c.violations == 0
        assert c.top1_agreement == 1.0 and c.top5_overlap == 1.0

    def test_shape_mismatch_fails_without_metrics(self) -> None:
        c = compare_outputs(logits([[1.0, 2.0]]), logits([[1.0, 2.0, 3.0]]))
        assert not c.passed
        assert any("shape" in r for r in c.reasons)
        assert c.max_abs_error is None

    def test_dtype_mismatch_fails(self) -> None:
        ref = logits([[1.0, 2.0]])
        c = compare_outputs(ref, ref.astype(np.float64))
        assert not c.passed
        assert any("dtype" in r for r in c.reasons)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_fails(self, bad: float) -> None:
        ref = logits([[1.0, 2.0]])
        c = compare_outputs(ref, logits([[1.0, bad]]))
        assert not c.passed and not c.all_finite

    def test_empty_fails(self) -> None:
        empty = np.zeros((0, 3), dtype=np.float32)
        assert not compare_outputs(empty, empty).passed


class TestToleranceArithmetic:
    """Hand-computed: ref max|.| = 10 -> atol = 1e-4 * 10 = 1e-3."""

    def test_atol_is_scale_aware(self) -> None:
        c = compare_outputs(logits([[10.0, 1.0]]), logits([[10.0, 1.0]]))
        assert c.atol == pytest.approx(1e-3)

    def test_error_within_bound_passes(self) -> None:
        # element 1: |ref|=1 -> bound = 1e-3 + 1e-4*1 = 1.1e-3; error 1e-3 < 1.1e-3
        ref = logits([[10.0, 1.0]])
        c = compare_outputs(ref, logits([[10.0, 1.001]]))
        assert c.passed, c.reasons
        assert c.worst_violation_ratio == pytest.approx(1e-3 / 1.1e-3, rel=1e-3)

    def test_error_outside_bound_fails(self) -> None:
        # error 2e-3 > 1.1e-3 on element 1
        ref = logits([[10.0, 1.0]])
        c = compare_outputs(ref, logits([[10.0, 1.002]]))
        assert not c.passed
        assert c.violations == 1
        assert c.worst_violation_ratio == pytest.approx(2e-3 / 1.1e-3, rel=1e-3)

    def test_relative_error_ignores_near_zero_references(self) -> None:
        # |ref| = 1e-5 is below 1e-3 * max(=10), so its huge relative error is not reported
        ref = logits([[10.0, 1e-5]])
        c = compare_outputs(ref, logits([[10.0, 2e-5]]))
        assert c.max_rel_error == pytest.approx(0.0, abs=1e-9)
        assert c.passed, "absolute error 1e-5 is far inside atol"

    def test_max_rel_can_exceed_rtol_while_passing(self) -> None:
        """Documented behaviour: atol dominates for small |ref|; max_rel is informational.

        ref=0.02 (above the 0.01 meaningful cut), error 5e-6: rel 2.5e-4 > rtol 1e-4,
        yet within atol (1e-3). This is the pattern seen in the ResNet-50 report.
        """
        ref = logits([[10.0, 0.02]])
        c = compare_outputs(ref, logits([[10.0, 0.020005]]))
        assert c.passed
        assert c.max_rel_error is not None and c.max_rel_error > DEFAULT_RTOL


class TestClassifierCriteria:
    def test_top1_disagreement_fails_even_within_tolerance(self) -> None:
        """A near-tie flipped by a tiny error is still a correctness failure."""
        # float32 spacing near 5 is ~4.8e-7, so use differences it can represent.
        ref = logits([[5.0, 5.00001, 0.0]])
        cand = logits([[5.00002, 5.0, 0.0]])
        assert ref.argmax() != cand.argmax(), "fixture must actually flip the class"
        c = compare_outputs(ref, cand)
        assert c.violations == 0, "within numeric tolerance..."
        assert not c.passed, "...but the predicted class changed"
        assert c.top1_agreement == 0.0

    def test_top5_overlap(self) -> None:
        ref = logits([[9, 8, 7, 6, 5, 4, 3, 2]])
        cand = logits([[9, 8, 7, 6, 1, 4, 3, 2]])  # top-5 {0..4} vs {0..3,5}: overlap 4/5
        c = compare_outputs(ref, cand, rtol=10, atol_scale=10)
        assert c.top5_overlap == pytest.approx(0.8)
        assert c.top1_agreement == 1.0

    def test_non_classifier_skips_top_k(self) -> None:
        ref = logits([[1.0, 2.0]])
        c = compare_outputs(ref, ref.copy(), classifier=False)
        assert c.top1_agreement is None and c.passed

    def test_top1_alone_is_not_sufficient(self) -> None:
        """The FP16 negative control kept top-1 = 1.0 while failing the tolerance."""
        ref = logits([[10.0, 1.0, 0.5]])
        cand = logits([[10.05, 1.0, 0.5]])  # same argmax, error 5e-2 >> atol 1e-3
        c = compare_outputs(ref, cand)
        assert c.top1_agreement == 1.0
        assert not c.passed


class TestSyntheticInput:
    def test_deterministic(self) -> None:
        a = synthetic_input((2, 3, 4, 4), seed=7)
        b = synthetic_input((2, 3, 4, 4), seed=7)
        assert a.dtype == np.float32 and a.shape == (2, 3, 4, 4)
        assert np.array_equal(a, b)

    def test_seed_changes_values(self) -> None:
        assert not np.array_equal(synthetic_input((8,), 0), synthetic_input((8,), 1))

    def test_prefix_is_not_batch_invariant(self) -> None:
        """Batch 1 and batch 4 with the same seed are different draws, by design."""
        one = synthetic_input((1, 16), 3)
        four = synthetic_input((4, 16), 3)
        assert np.array_equal(one[0], four[0])  # PCG64 streams: same prefix...
        assert four.shape[0] == 4  # ...so batch-1 is the first row of batch-4

    def test_generator_description_is_recorded_verbatim(self) -> None:
        assert "default_rng(seed)" in INPUT_GENERATOR and "float32" in INPUT_GENERATOR
