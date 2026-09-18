"""Benchmark core: timing, statistics, configuration, result schema and the engine."""

from __future__ import annotations

from gpu_benchlab.core.backend import Backend, BackendDescriptor, DeviceKind, ModelInfo
from gpu_benchlab.core.config import (
    DEFAULT_MEASUREMENT_ITERATIONS,
    DEFAULT_WARMUP_ITERATIONS,
    BenchmarkConfig,
    ExperimentConfig,
    ModelConfig,
    load_config,
    parse_config,
)
from gpu_benchlab.core.engine import BenchmarkEngine
from gpu_benchlab.core.errors import (
    BackendError,
    BenchLabError,
    ConfigurationError,
    InvalidSampleError,
    OutOfMemoryError,
    UnavailableError,
    UnsupportedConfigurationError,
)
from gpu_benchlab.core.provenance import capture_provenance
from gpu_benchlab.core.schema import (
    RESULT_SCHEMA_VERSION,
    BenchmarkResult,
    BenchmarkStatus,
    ErrorRecord,
    PhaseTimings,
    Provenance,
    RawSamples,
    ThroughputMetric,
    ThroughputUnit,
)
from gpu_benchlab.core.statistics import (
    PERCENTILE_MIN_SAMPLES,
    REPORTED_PERCENTILES,
    LatencyStatistics,
    validate_samples,
)
from gpu_benchlab.core.storage import DEFAULT_RESULTS_DIR, ResultStore
from gpu_benchlab.core.timing import (
    ScriptedTimer,
    Timer,
    TimingMechanism,
    WallClockTimer,
)

__all__ = [
    "DEFAULT_MEASUREMENT_ITERATIONS",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_WARMUP_ITERATIONS",
    "PERCENTILE_MIN_SAMPLES",
    "REPORTED_PERCENTILES",
    "RESULT_SCHEMA_VERSION",
    "Backend",
    "BackendDescriptor",
    "BackendError",
    "BenchLabError",
    "BenchmarkConfig",
    "BenchmarkEngine",
    "BenchmarkResult",
    "BenchmarkStatus",
    "ConfigurationError",
    "DeviceKind",
    "ErrorRecord",
    "ExperimentConfig",
    "InvalidSampleError",
    "LatencyStatistics",
    "ModelConfig",
    "ModelInfo",
    "OutOfMemoryError",
    "PhaseTimings",
    "Provenance",
    "RawSamples",
    "ResultStore",
    "ScriptedTimer",
    "ThroughputMetric",
    "ThroughputUnit",
    "Timer",
    "TimingMechanism",
    "UnavailableError",
    "UnsupportedConfigurationError",
    "WallClockTimer",
    "capture_provenance",
    "load_config",
    "parse_config",
    "validate_samples",
]
