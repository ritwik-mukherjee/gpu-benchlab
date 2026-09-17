"""Tests for latency statistics.

Where possible the expected values are computed **by hand** from the definition
rather than by calling numpy, so these tests would catch a wrong-but-consistent
implementation. A test that asserts `np.percentile(x) == np.percentile(x)` proves
nothing.
"""

from __future__ import annotations

import math

import pytest

from gpu_benchlab.core.errors import InvalidSampleError
from gpu_benchlab.core.statistics import (
    PERCENTILE_MIN_SAMPLES,
    REPORTED_PERCENTILES,
    LatencyStatistics,
    validate_samples,
)

# 1..100 inclusive. Chosen because every statistic is analytically known:
#   mean   = 50.5
#   median = 50.5
#   linear-interpolated percentile p = 1 + (p/100)*(n-1) = 1 + 0.99p
#     p50 -> 1 + 49.5  = 50.5
#     p90 -> 1 + 89.1  = 90.1
#     p95 -> 1 + 94.05 = 95.05
#     p99 -> 1 + 98.01 = 99.01
#   sample stddev (ddof=1) = sqrt(n(n+1)/12) = sqrt(100*101/12) = 29.011491...
LINEAR_1_TO_100 = [float(i) for i in range(1, 101)]


class TestValidation:
    def test_empty_is_rejected(self) -> None:
        with pytest.raises(InvalidSampleError, match="No timing samples"):
            validate_samples([])

    def test_nan_is_rejected_not_dropped(self) -> None:
        """Dropping a NaN would silently change every other statistic."""
        with pytest.raises(InvalidSampleError, match="NaN"):
            validate_samples([1.0, float("nan"), 3.0])

    def test_positive_infinity_is_rejected(self) -> None:
        with pytest.raises(InvalidSampleError, match="infinite"):
            validate_samples([1.0, float("inf")])

    def test_negative_infinity_is_rejected(self) -> None:
        with pytest.raises(InvalidSampleError):
            validate_samples([1.0, float("-inf")])

    def test_negative_duration_is_rejected(self) -> None:
        """A negative duration means a non-monotonic clock or a timer bug."""
        with pytest.raises(InvalidSampleError, match="negative"):
            validate_samples([1.0, -0.5, 3.0])

    def test_zero_is_allowed(self) -> None:
        """A zero-duration sample is implausible but not invalid."""
        validate_samples([0.0, 1.0])

    def test_non_numeric_is_rejected(self) -> None:
        with pytest.raises(InvalidSampleError, match="not a number"):
            validate_samples([1.0, "2.0"])  # type: ignore[list-item]

    def test_bool_is_rejected(self) -> None:
        """bool is a subclass of int; accepting it would silently coerce True->1."""
        with pytest.raises(InvalidSampleError, match="not a number"):
            validate_samples([1.0, True])  # type: ignore[list-item]

    def test_error_names_the_offending_index(self) -> None:
        with pytest.raises(InvalidSampleError, match="Sample 2"):
            validate_samples([1.0, 2.0, float("nan")])

    def test_from_samples_rejects_invalid(self) -> None:
        with pytest.raises(InvalidSampleError):
            LatencyStatistics.from_samples([1.0, float("nan")])


class TestKnownAnswers:
    """Hand-computed expectations for an analytically tractable dataset."""

    @pytest.fixture
    def stats(self) -> LatencyStatistics:
        return LatencyStatistics.from_samples(LINEAR_1_TO_100)

    def test_count(self, stats: LatencyStatistics) -> None:
        assert stats.count == 100

    def test_min_max(self, stats: LatencyStatistics) -> None:
        assert stats.min_ms == 1.0
        assert stats.max_ms == 100.0

    def test_mean(self, stats: LatencyStatistics) -> None:
        assert stats.mean_ms == pytest.approx(50.5)

    def test_median(self, stats: LatencyStatistics) -> None:
        assert stats.median_ms == pytest.approx(50.5)

    def test_p50_equals_median(self, stats: LatencyStatistics) -> None:
        assert stats.p50_ms == pytest.approx(stats.median_ms)

    def test_p90_hand_computed(self, stats: LatencyStatistics) -> None:
        # index = 1 + (90/100) * (100-1) = 90.1
        assert stats.p90_ms == pytest.approx(90.1)

    def test_p95_hand_computed(self, stats: LatencyStatistics) -> None:
        assert stats.p95_ms == pytest.approx(95.05)

    def test_p99_hand_computed(self, stats: LatencyStatistics) -> None:
        assert stats.p99_ms == pytest.approx(99.01)

    def test_sample_stddev_uses_ddof_1(self, stats: LatencyStatistics) -> None:
        # Population stddev of 1..100 would be 28.8661; sample stddev is 29.0115.
        # Asserting the difference proves ddof=1 is actually applied.
        expected = math.sqrt(100 * 101 / 12)
        assert stats.stddev_ms == pytest.approx(expected, rel=1e-9)
        assert stats.stddev_ms != pytest.approx(28.8661, rel=1e-4)

    def test_method_is_recorded(self, stats: LatencyStatistics) -> None:
        assert stats.percentile_method == "linear"
        assert stats.stddev_ddof == 1


class TestOrdering:
    def test_percentiles_are_monotonic(self) -> None:
        stats = LatencyStatistics.from_samples(LINEAR_1_TO_100)
        assert stats.min_ms <= stats.p50_ms <= stats.p90_ms
        assert stats.p90_ms <= stats.p95_ms <= stats.p99_ms <= stats.max_ms

    def test_input_order_does_not_matter(self) -> None:
        """Statistics are order statistics; shuffling must not change them."""
        forward = LatencyStatistics.from_samples(LINEAR_1_TO_100)
        backward = LatencyStatistics.from_samples(list(reversed(LINEAR_1_TO_100)))
        assert forward.model_dump() == backward.model_dump()


class TestSmallSamples:
    def test_single_sample(self) -> None:
        stats = LatencyStatistics.from_samples([7.5])
        assert stats.count == 1
        assert stats.min_ms == stats.max_ms == stats.mean_ms == 7.5
        assert stats.p50_ms == stats.p99_ms == 7.5

    def test_single_sample_stddev_is_zero_not_nan(self) -> None:
        """ddof=1 on one sample divides by zero; that must not leak a NaN out."""
        stats = LatencyStatistics.from_samples([7.5])
        assert stats.stddev_ms == 0.0
        assert not math.isnan(stats.stddev_ms)

    def test_two_samples(self) -> None:
        stats = LatencyStatistics.from_samples([10.0, 20.0])
        assert stats.mean_ms == pytest.approx(15.0)
        assert stats.p50_ms == pytest.approx(15.0)
        # sample stddev of {10,20} = sqrt(((10-15)^2 + (20-15)^2)/1) = sqrt(50)
        assert stats.stddev_ms == pytest.approx(math.sqrt(50.0))

    def test_identical_samples_have_zero_spread(self) -> None:
        stats = LatencyStatistics.from_samples([5.0] * 10)
        assert stats.stddev_ms == pytest.approx(0.0)
        assert stats.min_ms == stats.max_ms == stats.p99_ms == 5.0


class TestOutliers:
    def test_outlier_moves_mean_more_than_median(self) -> None:
        """The classic reason p50 is reported alongside the mean."""
        clean = [10.0] * 99 + [10.0]
        spiked = [10.0] * 99 + [1000.0]
        clean_stats = LatencyStatistics.from_samples(clean)
        spiked_stats = LatencyStatistics.from_samples(spiked)

        assert spiked_stats.median_ms == pytest.approx(clean_stats.median_ms)
        assert spiked_stats.mean_ms > clean_stats.mean_ms

    def test_single_outlier_dominates_p99_at_100_samples(self) -> None:
        """Demonstrates exactly why p99 is flagged low-confidence below 1000 samples.

        With 99 samples at 10ms and one at 1000ms, the linear-interpolated p99
        sits at index 1 + 0.99*99 = 99.01, i.e. 1% of the way from the 99th
        order statistic (10.0) to the 100th (1000.0): 10 + 0.01*990 = 19.9.

        So a single outlier doubles the reported p99 while leaving p50 untouched.
        Note it does NOT approach the outlier's own value -- a reader who assumes
        "p99 ~ the slow case" would be wrong in the other direction. Both errors
        are why the statistic is flagged rather than presented as stable.
        """
        samples = [10.0] * 99 + [1000.0]
        stats = LatencyStatistics.from_samples(samples)

        assert stats.p50_ms == pytest.approx(10.0), "p50 is unmoved by one outlier"
        assert stats.p99_ms == pytest.approx(19.9)
        assert stats.p99_ms > 1.9 * stats.p50_ms, "one outlier nearly doubled p99"
        assert stats.p99_ms < stats.max_ms / 10, "yet p99 is nowhere near the outlier"
        assert stats.is_low_confidence(99)

    def test_outlier_is_preserved_in_max(self) -> None:
        stats = LatencyStatistics.from_samples([10.0] * 99 + [1000.0])
        assert stats.max_ms == 1000.0


class TestConfidenceFlags:
    def test_thresholds_match_documented_values(self) -> None:
        """These are referenced by docs/methodology.md §6."""
        assert PERCENTILE_MIN_SAMPLES == {50: 30, 90: 100, 95: 200, 99: 1000}

    def test_every_reported_percentile_has_a_threshold(self) -> None:
        assert set(REPORTED_PERCENTILES) == set(PERCENTILE_MIN_SAMPLES)

    def test_tiny_sample_flags_everything(self) -> None:
        stats = LatencyStatistics.from_samples([1.0, 2.0, 3.0])
        assert stats.low_confidence_percentiles == ["p50", "p90", "p95", "p99"]

    def test_100_samples_flags_p95_and_p99_only(self) -> None:
        stats = LatencyStatistics.from_samples(LINEAR_1_TO_100)
        assert stats.low_confidence_percentiles == ["p95", "p99"]

    def test_1000_samples_flags_nothing(self) -> None:
        stats = LatencyStatistics.from_samples([float(i) for i in range(1000)])
        assert stats.low_confidence_percentiles == []

    def test_boundary_is_inclusive(self) -> None:
        """Exactly 30 samples must satisfy the p50 threshold, not fall short of it."""
        assert not LatencyStatistics.from_samples([1.0] * 30).is_low_confidence(50)
        assert LatencyStatistics.from_samples([1.0] * 29).is_low_confidence(50)


class TestSerialisation:
    def test_round_trips(self) -> None:
        stats = LatencyStatistics.from_samples(LINEAR_1_TO_100)
        restored = LatencyStatistics.model_validate_json(stats.model_dump_json())
        assert restored == stats

    def test_is_frozen(self) -> None:
        stats = LatencyStatistics.from_samples([1.0])
        with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError on frozen
            stats.count = 99  # type: ignore[misc]
