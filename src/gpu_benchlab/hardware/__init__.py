"""Hardware, driver and framework detection."""

from __future__ import annotations

from gpu_benchlab.hardware.capability import (
    Precision,
    PrecisionSupport,
    architecture_name,
    precision_support,
    supported_precisions,
)
from gpu_benchlab.hardware.detect import detect_environment
from gpu_benchlab.hardware.frameworks import probe_frameworks
from gpu_benchlab.hardware.host import probe_host
from gpu_benchlab.hardware.nvml import NVMLProbe, NVMLResult
from gpu_benchlab.hardware.types import (
    ENVIRONMENT_SCHEMA_VERSION,
    DetectionStatus,
    EnvironmentReport,
    FrameworkInfo,
    GPUInfo,
    HostInfo,
    NVMLInfo,
    PrecisionSupportInfo,
)

__all__ = [
    "ENVIRONMENT_SCHEMA_VERSION",
    "DetectionStatus",
    "EnvironmentReport",
    "FrameworkInfo",
    "GPUInfo",
    "HostInfo",
    "NVMLInfo",
    "NVMLProbe",
    "NVMLResult",
    "Precision",
    "PrecisionSupport",
    "PrecisionSupportInfo",
    "architecture_name",
    "detect_environment",
    "precision_support",
    "probe_frameworks",
    "probe_host",
    "supported_precisions",
]
