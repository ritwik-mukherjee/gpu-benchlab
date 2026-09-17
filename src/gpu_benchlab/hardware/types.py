"""Schema for hardware, software and environment detection.

These models are versioned and serialised into every benchmark result so that a
measurement can be traced back to the exact machine that produced it (PRD §16/§17).

Every optional field means "this machine or driver did not report it", never
"we did not bother to look". Callers must be able to distinguish a real zero
from a missing reading, which is why nullable fields are used instead of
sentinel numbers.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from gpu_benchlab.hardware.capability import Precision, PrecisionSupport

ENVIRONMENT_SCHEMA_VERSION = "1.0"

__all__ = [
    "ENVIRONMENT_SCHEMA_VERSION",
    "DetectionStatus",
    "EnvironmentReport",
    "FrameworkInfo",
    "GPUInfo",
    "HostInfo",
    "NVMLInfo",
    "PrecisionSupportInfo",
]


class DetectionStatus(str, Enum):
    """Outcome of attempting to enumerate NVIDIA GPUs.

    These are deliberately distinct: "you have no GPU" and "you have a GPU but
    no driver" require completely different user action, and collapsing them
    into a single failure mode is the most common bug in tools like this.
    """

    OK = "ok"
    """NVML initialised and reported at least one device."""

    NO_NVIDIA_DEVICE = "no_nvidia_device"
    """NVML initialised successfully but reported zero devices."""

    DRIVER_UNAVAILABLE = "driver_unavailable"
    """The NVML shared library could not be loaded — typically no NVIDIA driver."""

    LIBRARY_UNAVAILABLE = "library_unavailable"
    """The `nvidia-ml-py` Python package is not installed."""

    ERROR = "error"
    """NVML raised an unexpected error; see `detection_error`."""


class PrecisionSupportInfo(BaseModel):
    """Serialisable form of :class:`~gpu_benchlab.hardware.capability.PrecisionSupport`."""

    model_config = ConfigDict(frozen=True)

    precision: Precision
    supported: bool
    tensor_core: bool
    note: str

    @classmethod
    def from_dataclass(cls, ps: PrecisionSupport) -> PrecisionSupportInfo:
        return cls(
            precision=ps.precision,
            supported=ps.supported,
            tensor_core=ps.tensor_core,
            note=ps.note,
        )


class GPUInfo(BaseModel):
    """A single physical NVIDIA GPU as reported by NVML.

    Telemetry fields (utilisation, power, temperature, clocks) are *instantaneous*
    samples taken at detection time. They are recorded for environment context
    only and must never be confused with the sampled telemetry collected during a
    benchmark run.
    """

    model_config = ConfigDict(frozen=True)

    index: int
    name: str
    uuid: str | None = None
    pci_bus_id: str | None = None
    serial: str | None = None

    architecture: str | None = None
    compute_capability: str | None = Field(
        default=None, description='Compute capability as "major.minor", e.g. "8.9".'
    )
    compute_capability_major: int | None = None
    compute_capability_minor: int | None = None
    multiprocessor_count: int | None = None

    memory_total_bytes: int | None = None
    memory_free_bytes: int | None = None
    memory_used_bytes: int | None = None

    # --- instantaneous telemetry at detection time -----------------------------------
    utilization_gpu_percent: int | None = None
    utilization_memory_percent: int | None = None
    temperature_celsius: int | None = None
    power_usage_watts: float | None = None
    power_limit_watts: float | None = None
    sm_clock_mhz: int | None = None
    max_sm_clock_mhz: int | None = None
    memory_clock_mhz: int | None = None
    max_memory_clock_mhz: int | None = None

    persistence_mode: bool | None = None
    compute_mode: str | None = None

    precision_support: list[PrecisionSupportInfo] = Field(default_factory=list)

    @property
    def memory_total_gib(self) -> float | None:
        if self.memory_total_bytes is None:
            return None
        return self.memory_total_bytes / (1024**3)

    @property
    def memory_free_gib(self) -> float | None:
        if self.memory_free_bytes is None:
            return None
        return self.memory_free_bytes / (1024**3)

    def supported_precisions(self) -> list[Precision]:
        return [p.precision for p in self.precision_support if p.supported]


class NVMLInfo(BaseModel):
    """Driver-level information reported by NVML, independent of any single GPU."""

    model_config = ConfigDict(frozen=True)

    driver_version: str | None = None
    nvml_version: str | None = None
    cuda_driver_version: str | None = Field(
        default=None,
        description=(
            "Maximum CUDA runtime version supported by the installed driver, "
            'formatted "major.minor". This is NOT the CUDA toolkit version that '
            "any given framework was compiled against."
        ),
    )


class FrameworkInfo(BaseModel):
    """Presence and CUDA-readiness of one inference framework.

    `available` means importable. `cuda_available` means the framework itself
    reports a usable CUDA device — which can be False even when a GPU exists,
    for example when a CPU-only wheel is installed.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    available: bool
    version: str | None = None
    cuda_available: bool | None = None
    cuda_version: str | None = None
    device_count: int | None = None
    providers: list[str] | None = Field(
        default=None, description="ONNX Runtime execution providers, when applicable."
    )
    detail: str | None = None


class HostInfo(BaseModel):
    """Operating system, CPU, memory and Python environment."""

    model_config = ConfigDict(frozen=True)

    hostname: str
    os_name: str
    os_version: str
    os_release: str | None = None
    architecture: str
    cpu_model: str | None = None
    cpu_physical_cores: int | None = None
    cpu_logical_cores: int | None = None
    total_memory_bytes: int | None = None
    python_version: str
    python_implementation: str
    python_executable: str


class EnvironmentReport(BaseModel):
    """Complete snapshot of the machine, embedded in every benchmark result."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = ENVIRONMENT_SCHEMA_VERSION
    timestamp_utc: str
    benchlab_version: str

    host: HostInfo
    nvml: NVMLInfo = Field(default_factory=NVMLInfo)
    gpus: list[GPUInfo] = Field(default_factory=list)
    frameworks: list[FrameworkInfo] = Field(default_factory=list)

    detection_status: DetectionStatus
    detection_error: str | None = None

    @property
    def has_nvidia_gpu(self) -> bool:
        """True only when at least one NVIDIA GPU was actually enumerated."""
        return self.detection_status is DetectionStatus.OK and len(self.gpus) > 0

    def framework(self, name: str) -> FrameworkInfo | None:
        for fw in self.frameworks:
            if fw.name == name:
                return fw
        return None
