"""Environment detection orchestrator.

Produces the :class:`EnvironmentReport` that is embedded into every benchmark
result. This is the single entry point the rest of the codebase should use.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_benchlab import __version__
from gpu_benchlab.hardware.frameworks import probe_frameworks
from gpu_benchlab.hardware.host import probe_host
from gpu_benchlab.hardware.nvml import NVMLProbe
from gpu_benchlab.hardware.types import EnvironmentReport

__all__ = ["detect_environment"]


def detect_environment(*, include_frameworks: bool = True) -> EnvironmentReport:
    """Detect the full execution environment.

    Args:
        include_frameworks: Whether to import and probe PyTorch / ONNX Runtime /
            TensorRT. This is slow (importing torch takes seconds) so callers on
            a hot path can skip it.

    Returns:
        A populated report. This function never raises: an environment that
        cannot be detected is itself a finding, and is reported through
        ``detection_status`` and ``detection_error``.
    """
    nvml_result = NVMLProbe().probe()

    return EnvironmentReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        benchlab_version=__version__,
        host=probe_host(),
        nvml=nvml_result.info,
        gpus=nvml_result.gpus,
        frameworks=probe_frameworks() if include_frameworks else [],
        detection_status=nvml_result.status,
        detection_error=nvml_result.error,
    )
