"""NVML access layer.

Uses ``nvidia-ml-py``, the binding authored and published by NVIDIA. Note that
the *package* is ``nvidia-ml-py`` while the *module* it installs is ``pynvml``;
the separately-published ``pynvml`` distribution is deprecated and must not be
used (see CLAUDE.md §12).

Design rules for this module:

* Never raise on a missing GPU, missing driver or missing library. Absence is a
  reportable state, not an exception (PRD §33).
* Never let an unsupported NVML query kill the whole report. Many fields are
  genuinely unavailable on consumer cards (power limits, some clocks), and on
  virtualised GPUs. Each query is individually guarded and degrades to ``None``.
* Return ``None`` rather than ``0`` for anything that could not be read, so that
  a real zero is distinguishable from a missing reading.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from gpu_benchlab.hardware.capability import architecture_name, precision_support
from gpu_benchlab.hardware.types import (
    DetectionStatus,
    GPUInfo,
    NVMLInfo,
    PrecisionSupportInfo,
)

__all__ = ["NVMLProbe", "NVMLResult"]

T = TypeVar("T")


class NVMLResult:
    """Outcome of an NVML probe."""

    def __init__(
        self,
        status: DetectionStatus,
        info: NVMLInfo,
        gpus: list[GPUInfo],
        error: str | None = None,
    ) -> None:
        self.status = status
        self.info = info
        self.gpus = gpus
        self.error = error


def _decode(value: Any) -> str | None:
    """NVML returned bytes in older bindings and str in newer ones. Accept both."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _safe(fn: Callable[[], T]) -> T | None:
    """Run an NVML query, returning None if the driver/GPU does not support it.

    NVML raises a family of exceptions (NotSupported, NoPermission, Unknown) for
    fields that are simply unavailable on a given part. Those are expected, not
    exceptional, so they collapse to None.
    """
    try:
        return fn()
    except Exception:  # noqa: BLE001 - NVML error taxonomy is broad and version-dependent
        return None


class NVMLProbe:
    """Enumerates NVIDIA GPUs via NVML, degrading honestly when none are present."""

    def __init__(self) -> None:
        self._nvml: Any | None = None

    # -- lifecycle ---------------------------------------------------------------------

    @contextlib.contextmanager
    def _session(self) -> Iterator[Any]:
        """Initialise NVML for the duration of the block, always shutting down."""
        import pynvml  # provided by the `nvidia-ml-py` distribution

        pynvml.nvmlInit()
        try:
            yield pynvml
        finally:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()

    # -- public API --------------------------------------------------------------------

    def probe(self) -> NVMLResult:
        """Enumerate GPUs. Never raises."""
        try:
            import pynvml  # noqa: F401
        except ImportError as exc:
            return NVMLResult(
                DetectionStatus.LIBRARY_UNAVAILABLE,
                NVMLInfo(),
                [],
                error=(
                    "The 'nvidia-ml-py' package is not installed, so GPU telemetry "
                    f"cannot be queried ({exc})."
                ),
            )

        try:
            with self._session() as nvml:
                return self._collect(nvml)
        except Exception as exc:  # noqa: BLE001
            return self._classify_init_failure(exc)

    # -- internals ---------------------------------------------------------------------

    def _classify_init_failure(self, exc: Exception) -> NVMLResult:
        """Turn an nvmlInit() failure into a specific, actionable status."""
        text = str(exc)
        lowered = text.lower()
        if "not found" in lowered or "libnvidia-ml" in lowered or "driver" in lowered:
            return NVMLResult(
                DetectionStatus.DRIVER_UNAVAILABLE,
                NVMLInfo(),
                [],
                error=(
                    "NVML could not be loaded. This normally means no NVIDIA driver is "
                    f"installed on this machine. Underlying error: {text}"
                ),
            )
        return NVMLResult(DetectionStatus.ERROR, NVMLInfo(), [], error=text)

    def _collect(self, nvml: Any) -> NVMLResult:
        info = NVMLInfo(
            driver_version=_decode(_safe(nvml.nvmlSystemGetDriverVersion)),
            nvml_version=_decode(_safe(nvml.nvmlSystemGetNVMLVersion)),
            cuda_driver_version=self._cuda_driver_version(nvml),
        )

        count = _safe(nvml.nvmlDeviceGetCount)
        if count is None:
            return NVMLResult(
                DetectionStatus.ERROR,
                info,
                [],
                error="NVML initialised but nvmlDeviceGetCount() failed.",
            )
        if count == 0:
            return NVMLResult(
                DetectionStatus.NO_NVIDIA_DEVICE,
                info,
                [],
                error="NVML initialised successfully but reported zero NVIDIA devices.",
            )

        gpus = [self._device(nvml, i) for i in range(count)]
        return NVMLResult(DetectionStatus.OK, info, gpus)

    @staticmethod
    def _cuda_driver_version(nvml: Any) -> str | None:
        """Max CUDA version the driver supports, as "major.minor".

        NVML encodes this as an integer, e.g. 12040 -> "12.4".
        """
        raw = _safe(nvml.nvmlSystemGetCudaDriverVersion_v2)
        if raw is None:
            raw = _safe(nvml.nvmlSystemGetCudaDriverVersion)
        if raw is None:
            return None
        return f"{raw // 1000}.{(raw % 1000) // 10}"

    def _device(self, nvml: Any, index: int) -> GPUInfo:
        handle = nvml.nvmlDeviceGetHandleByIndex(index)

        name = _decode(_safe(lambda: nvml.nvmlDeviceGetName(handle))) or f"NVIDIA GPU {index}"

        cc = _safe(lambda: nvml.nvmlDeviceGetCudaComputeCapability(handle))
        cc_major, cc_minor = cc if cc is not None else (None, None)

        arch: str | None = None
        precisions: list[PrecisionSupportInfo] = []
        cc_str: str | None = None
        if cc_major is not None and cc_minor is not None:
            cc_str = f"{cc_major}.{cc_minor}"
            arch = architecture_name(cc_major, cc_minor)
            precisions = [
                PrecisionSupportInfo.from_dataclass(ps)
                for ps in precision_support(cc_major, cc_minor)
            ]

        mem = _safe(lambda: nvml.nvmlDeviceGetMemoryInfo(handle))
        util = _safe(lambda: nvml.nvmlDeviceGetUtilizationRates(handle))
        pci = _safe(lambda: nvml.nvmlDeviceGetPciInfo(handle))

        power_usage_mw = _safe(lambda: nvml.nvmlDeviceGetPowerUsage(handle))
        power_limit_mw = _safe(lambda: nvml.nvmlDeviceGetEnforcedPowerLimit(handle))

        return GPUInfo(
            index=index,
            name=name,
            uuid=_decode(_safe(lambda: nvml.nvmlDeviceGetUUID(handle))),
            pci_bus_id=_decode(getattr(pci, "busId", None)) if pci is not None else None,
            serial=_decode(_safe(lambda: nvml.nvmlDeviceGetSerial(handle))),
            architecture=arch,
            compute_capability=cc_str,
            compute_capability_major=cc_major,
            compute_capability_minor=cc_minor,
            cuda_core_count=_safe(
                # CUDA cores, not SMs: measured 7424 on an L4 whose SM count is 58.
                lambda: nvml.nvmlDeviceGetNumGpuCores(handle)
            ),
            memory_total_bytes=getattr(mem, "total", None) if mem is not None else None,
            memory_free_bytes=getattr(mem, "free", None) if mem is not None else None,
            memory_used_bytes=getattr(mem, "used", None) if mem is not None else None,
            utilization_gpu_percent=getattr(util, "gpu", None) if util is not None else None,
            utilization_memory_percent=getattr(util, "memory", None) if util is not None else None,
            temperature_celsius=_safe(
                lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            ),
            power_usage_watts=(power_usage_mw / 1000.0) if power_usage_mw is not None else None,
            power_limit_watts=(power_limit_mw / 1000.0) if power_limit_mw is not None else None,
            sm_clock_mhz=_safe(lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)),
            max_sm_clock_mhz=_safe(
                lambda: nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_SM)
            ),
            memory_clock_mhz=_safe(
                lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
            ),
            max_memory_clock_mhz=_safe(
                lambda: nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_MEM)
            ),
            persistence_mode=self._persistence_mode(nvml, handle),
            compute_mode=self._compute_mode(nvml, handle),
            precision_support=precisions,
        )

    @staticmethod
    def _persistence_mode(nvml: Any, handle: Any) -> bool | None:
        raw = _safe(lambda: nvml.nvmlDeviceGetPersistenceMode(handle))
        if raw is None:
            return None
        return bool(raw)

    @staticmethod
    def _compute_mode(nvml: Any, handle: Any) -> str | None:
        raw = _safe(lambda: nvml.nvmlDeviceGetComputeMode(handle))
        if raw is None:
            return None
        names = {
            0: "default",
            1: "exclusive_thread",
            2: "prohibited",
            3: "exclusive_process",
        }
        return names.get(int(raw), f"unknown({raw})")
