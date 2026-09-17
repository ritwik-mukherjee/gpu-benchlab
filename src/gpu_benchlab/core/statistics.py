"""Latency statistics derived from raw samples.

Every statistic here is computed from the stored raw samples and nothing else.
Aggregates are never stored in place of the samples that produced them, so any
statistic can be recomputed, audited, or corrected retroactively if the
methodology changes.

Two decisions are stated explicitly because tools differ and the differences are
visible at small sample counts:

* **Percentiles** use linear interpolation between the two nearest order
  statistics (``numpy.percentile`` default, equivalent to NIST/Excel
  ``PERCENTILE.INC``). Some tools use nearest-rank instead, which yields
  different p95 values for the same data.
* **Standard deviation** uses ``ddof=1`` (sample standard deviation, Bessel's
  correction). Benchmark iterations are a sample drawn from the distribution of
  possible runs, not the entire population.

Both choices are recorded in the emitted schema, so a reader never has to guess.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gpu_benchlab.core.errors import InvalidSampleError

__all__ = [
    "PERCENTILE_MIN_SAMPLES",
    "REPORTED_PERCENTILES",
    "LatencyStatistics",
    "validate_samples",
]

REPORTED_PERCENTILES: tuple[int, ...] = (50, 90, 95, 99)

# Minimum sample count for a percentile to be treated as a stable estimate.
#
# Rule of thumb: ~10 / (1 - p) samples, i.e. roughly ten observations are needed
# in the tail beyond the percentile before its position stops being dominated by
# individual outliers. p50 is pinned at 30 rather than the rule's 20, matching
# docs/methodology.md §6.
#
# Below these counts the percentile is still reported -- it is a real order
# statistic of real data -- but it is flagged, because a p99 computed from 100
# samples is essentially the maximum and a single outlier defines it.
PERCENTILE_MIN_SAMPLES: dict[int, int] = {
    50: 30,
    90: 100,
    95: 200,
    99: 1000,
}


def validate_samples(samples: Sequence[float]) -> None:
    """Reject sample sets that cannot support a statistic.

    Raises:
        InvalidSampleError: if the set is empty, or contains NaN, infinity or a
            negative duration.

    Bad samples are never silently dropped or coerced. Discarding a NaN changes
    every statistic computed from the remainder, and a NaN in a timing sample
    means the measurement mechanism misbehaved -- which the user needs to know
    about, not have quietly smoothed over.
    """
    if len(samples) == 0:
        raise InvalidSampleError(
            "No timing samples were collected, so no statistics can be computed."
        )

    for index, value in enumerate(samples):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidSampleError(
                f"Sample {index} is {value!r} ({type(value).__name__}), not a number."
            )
        if math.isnan(value):
            raise InvalidSampleError(
                f"Sample {index} is NaN. The timing mechanism produced an invalid "
                "measurement; statistics would be meaningless."
            )
        if math.isinf(value):
            raise InvalidSampleError(
                f"Sample {index} is infinite. The timing mechanism produced an invalid measurement."
            )
        if value < 0:
            raise InvalidSampleError(
                f"Sample {index} is negative ({value}). A duration cannot be "
                "negative; this indicates a non-monotonic clock or a timer bug."
            )


class LatencyStatistics(BaseModel):
    """Aggregate statistics over a set of per-iteration latency samples.

    All values are in milliseconds. Construct with :meth:`from_samples`; the
    fields are not intended to be assembled by hand.
    """

    model_config = ConfigDict(frozen=True)

    count: int = Field(description="Number of samples the statistics were computed from.")
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    stddev_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float

    percentile_method: str = Field(
        default="linear",
        description=(
            "Interpolation method between order statistics. 'linear' matches "
            "numpy.percentile's default and Excel PERCENTILE.INC."
        ),
    )
    stddev_ddof: int = Field(
        default=1,
        description="Delta degrees of freedom: 1 = sample standard deviation.",
    )
    low_confidence_percentiles: list[str] = Field(
        default_factory=list,
        description=(
            "Percentiles reported from too few samples to be stable estimates. "
            "The values are real order statistics, but should not be treated as "
            "representative. See docs/methodology.md §6."
        ),
    )

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> LatencyStatistics:
        """Compute statistics from raw per-iteration samples.

        Raises:
            InvalidSampleError: if the samples cannot support a statistic.
        """
        validate_samples(samples)

        array = np.asarray(samples, dtype=np.float64)
        count = int(array.size)

        percentiles = np.percentile(array, REPORTED_PERCENTILES, method="linear")
        by_percentile = dict(zip(REPORTED_PERCENTILES, percentiles, strict=True))

        # A single sample has no spread to measure; ddof=1 would divide by zero.
        stddev = float(np.std(array, ddof=1)) if count > 1 else 0.0

        return cls(
            count=count,
            min_ms=float(array.min()),
            max_ms=float(array.max()),
            mean_ms=float(array.mean()),
            median_ms=float(np.median(array)),
            stddev_ms=stddev,
            p50_ms=float(by_percentile[50]),
            p90_ms=float(by_percentile[90]),
            p95_ms=float(by_percentile[95]),
            p99_ms=float(by_percentile[99]),
            low_confidence_percentiles=[
                f"p{p}" for p in REPORTED_PERCENTILES if count < PERCENTILE_MIN_SAMPLES[p]
            ],
        )

    def is_low_confidence(self, percentile: int) -> bool:
        """Whether the given percentile was computed from too few samples."""
        return f"p{percentile}" in self.low_confidence_percentiles
