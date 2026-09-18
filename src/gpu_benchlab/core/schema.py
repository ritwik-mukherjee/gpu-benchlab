"""Versioned result schema.

One :class:`BenchmarkResult` is the complete, self-describing record of one
experiment: what was run, on what machine, how it was timed, what happened, every
raw sample, and any error. It is designed so that a reader who has only this
object can decide whether to believe it.

Schema evolution is additive. ``schema_version`` is stamped on every result, and
new fields are optional so that older results stay readable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gpu_benchlab.core.backend import BackendDescriptor, DeviceKind, ModelInfo
from gpu_benchlab.core.config import ExperimentConfig
from gpu_benchlab.core.statistics import LatencyStatistics
from gpu_benchlab.core.timing import TimingMechanism
from gpu_benchlab.hardware.types import EnvironmentReport

# 1.1 (Phase 3) adds: device_kind, model_info, backend settings, secondary timing
# and backend_options. All additive with defaults, so 1.0 results still load.
# Bumped rather than left at 1.0 because a 1.1 config is NOT readable by 1.0 code
# (ExperimentConfig forbids unknown keys) and a 1.0 reader would silently drop the
# secondary timing data. Consumers need to be able to tell.
RESULT_SCHEMA_VERSION = "1.1"

__all__ = [
    "RESULT_SCHEMA_VERSION",
    "BenchmarkResult",
    "BenchmarkStatus",
    "ErrorRecord",
    "PhaseTimings",
    "Provenance",
    "RawSamples",
    "ThroughputMetric",
    "ThroughputUnit",
]


class BenchmarkStatus(str, Enum):
    """Terminal outcome of an experiment.

    Every experiment ends in exactly one of these. There is no silent skip: a
    configuration that did not run is recorded with the reason it did not.
    """

    OK = "ok"
    """Ran to completion and produced trustworthy measurements."""

    UNSUPPORTED = "unsupported"
    """Well-formed but invalid for this hardware/backend/model combination."""

    FAILED = "failed"
    """Valid configuration, but execution errored (including OOM)."""

    UNAVAILABLE = "unavailable"
    """A required device or dependency is not present on this machine."""

    SKIPPED = "skipped"
    """Deliberately not run."""


class ThroughputUnit(str, Enum):
    """Throughput is never reported without naming what it counts (PRD §11)."""

    SAMPLES_PER_SECOND = "samples/sec"
    IMAGES_PER_SECOND = "images/sec"
    TOKENS_PER_SECOND = "tokens/sec"
    REQUESTS_PER_SECOND = "requests/sec"


class ThroughputMetric(BaseModel):
    """A throughput figure, its unit, and how it was derived.

    Two throughput numbers are meaningful and they are not the same:

    * derived from mean latency (``batch_size / mean_latency_s``) — the rate a
      single stream sustains, ignoring inter-iteration overhead;
    * observed over the whole measured window
      (``iterations * batch_size / total_wall_s``) — includes that overhead.

    Both are reported, each labelled, rather than picking one and calling it
    "throughput".
    """

    model_config = ConfigDict(frozen=True)

    value: float
    unit: ThroughputUnit
    basis: str = Field(description="The formula this value was computed from.")


class PhaseTimings(BaseModel):
    """Durations of the phases surrounding steady-state measurement.

    These are kept structurally separate from inference latency so that a
    TensorRT engine build cannot leak into it by accident (PRD §29).
    """

    model_config = ConfigDict(frozen=True)

    model_load_ms: float | None = None
    engine_build_ms: float | None = None
    prepare_inputs_ms: float | None = None
    warmup_total_ms: float | None = None
    measurement_total_ms: float | None = Field(
        default=None,
        description=(
            "Wall time for the entire measured loop, including per-iteration "
            "harness overhead. Not the sum of the samples."
        ),
    )


class ErrorRecord(BaseModel):
    """A failure, preserved rather than hidden (PRD §33)."""

    model_config = ConfigDict(frozen=True)

    type: str
    message: str
    traceback: str | None = None
    phase: str | None = Field(default=None, description="Which phase raised it.")
    timestamp_utc: str


class RawSamples(BaseModel):
    """Every individual measurement, preserved for independent analysis.

    Aggregates are always derived from this, never stored in its place.
    """

    model_config = ConfigDict(frozen=True)

    latency_ms: list[float] = Field(default_factory=list)
    warmup_latency_ms: list[float] = Field(
        default_factory=list,
        description=(
            "Warmup iteration durations. Excluded from all reported statistics, "
            "but retained so warmup convergence can be analysed empirically."
        ),
    )
    secondary_latency_ms: list[float] = Field(
        default_factory=list,
        description=(
            "Per-iteration secondary measurement (e.g. synchronized host wall time "
            "alongside CUDA-event device time). Same iterations as latency_ms."
        ),
    )
    secondary_warmup_latency_ms: list[float] = Field(default_factory=list)


class Provenance(BaseModel):
    """Enough version state to tie a result to the code that produced it."""

    model_config = ConfigDict(frozen=True)

    benchlab_version: str
    git_commit: str | None = None
    git_dirty: bool | None = Field(
        default=None,
        description="True if the working tree had uncommitted changes. Such a "
        "result is not reproducible from the commit alone.",
    )


class BenchmarkResult(BaseModel):
    """The complete record of one experiment."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = RESULT_SCHEMA_VERSION
    experiment_id: str
    group_id: str | None = Field(
        default=None, description="Shared by repeats of the same configuration."
    )
    repeat_index: int = 0

    timestamp_utc: str
    status: BenchmarkStatus

    is_simulated: bool = Field(
        description=(
            "True when produced by a simulated backend. Such a result exercises "
            "the framework and is NEVER valid hardware performance data. Stamped "
            "at the top level so it cannot be missed by any consumer."
        )
    )

    configuration: ExperimentConfig
    backend: BackendDescriptor
    environment: EnvironmentReport

    timing_mechanism: TimingMechanism | None = Field(
        description="How each sample was measured. Results using different "
        "mechanisms must not be compared without saying so. None when the run ended "
        "before any timer was created (e.g. unsupported or unavailable), because "
        "nothing was timed."
    )
    measures_steady_state: bool = Field(
        description="False when warmup_iterations is 0, meaning cold-start cost is included."
    )
    device_kind: DeviceKind | None = Field(
        default=None,
        description=(
            "cuda / cpu / simulated. A CPU result is a real measurement of the host "
            "CPU and is never GPU performance. None only in schema-1.0 results."
        ),
    )
    model_info: ModelInfo | None = Field(
        default=None, description="The model actually loaded, resolved to exact bytes."
    )
    secondary_timing_mechanism: TimingMechanism | None = None
    secondary_latency: LatencyStatistics | None = Field(
        default=None,
        description="Statistics over raw_samples.secondary_latency_ms. Never the headline.",
    )

    phases: PhaseTimings = Field(default_factory=PhaseTimings)
    latency: LatencyStatistics | None = Field(default=None, description="None unless status is ok.")
    throughput: list[ThroughputMetric] = Field(default_factory=list)
    raw_samples: RawSamples = Field(default_factory=RawSamples)

    provenance: Provenance
    errors: list[ErrorRecord] = Field(default_factory=list)
    notes: list[str] = Field(
        default_factory=list,
        description="Caveats that must travel with the result, e.g. a simulation warning.",
    )

    @property
    def succeeded(self) -> bool:
        return self.status is BenchmarkStatus.OK

    @property
    def is_real_measurement(self) -> bool:
        """True only for a successful run on a real backend.

        The single check any consumer should use before treating a number as
        performance data.
        """
        return self.status is BenchmarkStatus.OK and not self.is_simulated

    @property
    def is_gpu_measurement(self) -> bool:
        """True only for a successful, non-simulated run on a CUDA device.

        The check to use before presenting a number as GPU performance. A CPU
        run is a real measurement, but not of a GPU.
        """
        return self.is_real_measurement and self.device_kind is DeviceKind.CUDA

    def summary_dict(self) -> dict[str, Any]:
        """Compact form for tables and comparisons, without raw samples."""
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "is_simulated": self.is_simulated,
            "backend": self.backend.name,
            "device_kind": self.device_kind.value if self.device_kind else None,
            "device": self.backend.device,
            "precision": self.configuration.precision.value,
            "batch_size": self.configuration.batch_size,
            "p50_ms": self.latency.p50_ms if self.latency else None,
            "p95_ms": self.latency.p95_ms if self.latency else None,
            "p99_ms": self.latency.p99_ms if self.latency else None,
            "samples": self.latency.count if self.latency else 0,
        }


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
