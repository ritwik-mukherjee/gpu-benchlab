"""Tests for the NVML probe using an injected fake NVML module.

These run on any machine, with or without a GPU. They exist because the failure
modes are the whole point of this layer: telling "no driver" apart from "no GPU"
apart from "driver present but this field is unsupported" is what keeps the tool
from misreporting a machine.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from gpu_benchlab.hardware.capability import Precision
from gpu_benchlab.hardware.nvml import NVMLProbe
from gpu_benchlab.hardware.types import DetectionStatus


class FakeMemory:
    def __init__(self, total: int, free: int, used: int) -> None:
        self.total = total
        self.free = free
        self.used = used


class FakeUtil:
    def __init__(self, gpu: int, memory: int) -> None:
        self.gpu = gpu
        self.memory = memory


class FakePci:
    def __init__(self, bus_id: str) -> None:
        self.busId = bus_id  # mirrors the real NVML attribute name


class NotSupportedError(Exception):
    """Stands in for NVMLError_NotSupported."""


def make_fake_nvml(
    *,
    device_count: int = 1,
    init_error: Exception | None = None,
    unsupported: set[str] | None = None,
    name_as_bytes: bool = False,
) -> types.ModuleType:
    """Build a fake `pynvml` module.

    Args:
        device_count: How many GPUs to report.
        init_error: If set, nvmlInit raises this.
        unsupported: Names of NVML functions that should raise NotSupportedError,
            emulating fields that real consumer GPUs genuinely do not expose.
        name_as_bytes: Emulate the older bindings that returned bytes.
    """
    unsupported = unsupported or set()
    mod = types.ModuleType("pynvml")

    mod.NVML_TEMPERATURE_GPU = 0
    mod.NVML_CLOCK_SM = 1
    mod.NVML_CLOCK_MEM = 2

    def guard(fn_name: str, value: Any) -> Any:
        if fn_name in unsupported:
            raise NotSupportedError(f"{fn_name} is not supported on this device")
        return value

    def nvmlInit() -> None:  # noqa: N802 - mirrors NVML naming
        if init_error is not None:
            raise init_error

    mod.nvmlInit = nvmlInit
    mod.nvmlShutdown = lambda: None
    mod.nvmlSystemGetDriverVersion = lambda: guard("nvmlSystemGetDriverVersion", "560.94")
    mod.nvmlSystemGetNVMLVersion = lambda: guard("nvmlSystemGetNVMLVersion", "12.560.94")
    # 12040 -> "12.4"
    mod.nvmlSystemGetCudaDriverVersion_v2 = lambda: guard(
        "nvmlSystemGetCudaDriverVersion_v2", 12040
    )
    mod.nvmlSystemGetCudaDriverVersion = lambda: 12040
    mod.nvmlDeviceGetCount = lambda: device_count
    mod.nvmlDeviceGetHandleByIndex = lambda i: f"handle-{i}"

    raw_name = b"NVIDIA GeForce RTX 4090" if name_as_bytes else "NVIDIA GeForce RTX 4090"
    mod.nvmlDeviceGetName = lambda h: guard("nvmlDeviceGetName", raw_name)
    mod.nvmlDeviceGetUUID = lambda h: guard("nvmlDeviceGetUUID", "GPU-abc123")
    mod.nvmlDeviceGetSerial = lambda h: guard("nvmlDeviceGetSerial", "0323")
    mod.nvmlDeviceGetPciInfo = lambda h: guard("nvmlDeviceGetPciInfo", FakePci("0000:01:00.0"))
    mod.nvmlDeviceGetCudaComputeCapability = lambda h: guard(
        "nvmlDeviceGetCudaComputeCapability", (8, 9)
    )
    mod.nvmlDeviceGetNumGpuCores = lambda h: guard("nvmlDeviceGetNumGpuCores", 128)
    mod.nvmlDeviceGetMemoryInfo = lambda h: guard(
        "nvmlDeviceGetMemoryInfo",
        FakeMemory(total=25_757_220_864, free=24_000_000_000, used=1_757_220_864),
    )
    mod.nvmlDeviceGetUtilizationRates = lambda h: guard(
        "nvmlDeviceGetUtilizationRates", FakeUtil(gpu=13, memory=4)
    )
    mod.nvmlDeviceGetTemperature = lambda h, s: guard("nvmlDeviceGetTemperature", 41)
    mod.nvmlDeviceGetPowerUsage = lambda h: guard("nvmlDeviceGetPowerUsage", 32_500)
    mod.nvmlDeviceGetEnforcedPowerLimit = lambda h: guard(
        "nvmlDeviceGetEnforcedPowerLimit", 450_000
    )
    mod.nvmlDeviceGetClockInfo = lambda h, c: guard("nvmlDeviceGetClockInfo", 210)
    mod.nvmlDeviceGetMaxClockInfo = lambda h, c: guard("nvmlDeviceGetMaxClockInfo", 2520)
    mod.nvmlDeviceGetPersistenceMode = lambda h: guard("nvmlDeviceGetPersistenceMode", 0)
    mod.nvmlDeviceGetComputeMode = lambda h: guard("nvmlDeviceGetComputeMode", 0)
    return mod


@pytest.fixture
def install_fake(monkeypatch: pytest.MonkeyPatch):
    def _install(mod: types.ModuleType | None) -> None:
        if mod is None:
            monkeypatch.setitem(sys.modules, "pynvml", None)
        else:
            monkeypatch.setitem(sys.modules, "pynvml", mod)

    return _install


class TestFailureModes:
    def test_missing_package_reports_library_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = __import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "pynvml":
                raise ImportError("No module named 'pynvml'")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "pynvml", raising=False)
        monkeypatch.setattr("builtins.__import__", fake_import)

        result = NVMLProbe().probe()
        assert result.status is DetectionStatus.LIBRARY_UNAVAILABLE
        assert result.gpus == []
        assert result.error is not None and "nvidia-ml-py" in result.error

    def test_missing_driver_reports_driver_unavailable(self, install_fake) -> None:
        install_fake(make_fake_nvml(init_error=RuntimeError("NVML Shared Library Not Found")))
        result = NVMLProbe().probe()
        assert result.status is DetectionStatus.DRIVER_UNAVAILABLE
        assert result.error is not None and "driver" in result.error.lower()

    def test_zero_devices_is_distinct_from_missing_driver(self, install_fake) -> None:
        """The whole point of this layer: these two are not the same condition."""
        install_fake(make_fake_nvml(device_count=0))
        result = NVMLProbe().probe()
        assert result.status is DetectionStatus.NO_NVIDIA_DEVICE
        assert result.gpus == []
        # Driver info is still available even with no devices.
        assert result.info.driver_version == "560.94"

    def test_unexpected_error_reports_error_status(self, install_fake) -> None:
        install_fake(make_fake_nvml(init_error=RuntimeError("something bizarre happened")))
        result = NVMLProbe().probe()
        assert result.status is DetectionStatus.ERROR

    def test_probe_never_raises(self, install_fake) -> None:
        install_fake(make_fake_nvml(init_error=ValueError("boom")))
        NVMLProbe().probe()  # must not raise


class TestSuccessfulProbe:
    def test_reports_ok_and_one_device(self, install_fake) -> None:
        install_fake(make_fake_nvml(device_count=1))
        result = NVMLProbe().probe()
        assert result.status is DetectionStatus.OK
        assert len(result.gpus) == 1

    def test_driver_and_cuda_version_parsing(self, install_fake) -> None:
        install_fake(make_fake_nvml())
        info = NVMLProbe().probe().info
        assert info.driver_version == "560.94"
        assert info.cuda_driver_version == "12.4", "12040 must decode to 12.4"

    def test_device_fields(self, install_fake) -> None:
        install_fake(make_fake_nvml())
        gpu = NVMLProbe().probe().gpus[0]
        assert gpu.index == 0
        assert gpu.name == "NVIDIA GeForce RTX 4090"
        assert gpu.uuid == "GPU-abc123"
        assert gpu.pci_bus_id == "0000:01:00.0"
        assert gpu.compute_capability == "8.9"
        assert gpu.architecture == "Ada Lovelace"
        assert gpu.memory_total_bytes == 25_757_220_864
        assert gpu.utilization_gpu_percent == 13
        assert gpu.temperature_celsius == 41
        assert gpu.compute_mode == "default"
        assert gpu.persistence_mode is False

    def test_milliwatts_are_converted_to_watts(self, install_fake) -> None:
        install_fake(make_fake_nvml())
        gpu = NVMLProbe().probe().gpus[0]
        assert gpu.power_usage_watts == pytest.approx(32.5)
        assert gpu.power_limit_watts == pytest.approx(450.0)

    def test_precision_matrix_is_attached(self, install_fake) -> None:
        install_fake(make_fake_nvml())
        gpu = NVMLProbe().probe().gpus[0]
        supported = gpu.supported_precisions()
        assert Precision.FP8 in supported, "SM 8.9 supports FP8"
        assert Precision.FP4 not in supported, "SM 8.9 does not support FP4"

    def test_byte_strings_are_decoded(self, install_fake) -> None:
        """Older NVML bindings return bytes; both must work."""
        install_fake(make_fake_nvml(name_as_bytes=True))
        gpu = NVMLProbe().probe().gpus[0]
        assert gpu.name == "NVIDIA GeForce RTX 4090"

    def test_multiple_devices_are_indexed(self, install_fake) -> None:
        install_fake(make_fake_nvml(device_count=3))
        gpus = NVMLProbe().probe().gpus
        assert [g.index for g in gpus] == [0, 1, 2]


class TestGracefulFieldDegradation:
    """Real consumer GPUs do not expose every NVML field. Missing != zero."""

    def test_unsupported_power_becomes_none_not_zero(self, install_fake) -> None:
        install_fake(
            make_fake_nvml(
                unsupported={"nvmlDeviceGetPowerUsage", "nvmlDeviceGetEnforcedPowerLimit"}
            )
        )
        result = NVMLProbe().probe()
        assert result.status is DetectionStatus.OK
        gpu = result.gpus[0]
        assert gpu.power_usage_watts is None
        assert gpu.power_limit_watts is None
        # Everything else still came through.
        assert gpu.memory_total_bytes == 25_757_220_864

    def test_unsupported_compute_capability_does_not_crash(self, install_fake) -> None:
        install_fake(make_fake_nvml(unsupported={"nvmlDeviceGetCudaComputeCapability"}))
        gpu = NVMLProbe().probe().gpus[0]
        assert gpu.compute_capability is None
        assert gpu.architecture is None
        assert gpu.precision_support == []

    def test_unsupported_name_falls_back(self, install_fake) -> None:
        install_fake(make_fake_nvml(unsupported={"nvmlDeviceGetName"}))
        gpu = NVMLProbe().probe().gpus[0]
        assert gpu.name == "NVIDIA GPU 0"

    def test_many_unsupported_fields_still_yields_a_result(self, install_fake) -> None:
        install_fake(
            make_fake_nvml(
                unsupported={
                    "nvmlDeviceGetUUID",
                    "nvmlDeviceGetSerial",
                    "nvmlDeviceGetPciInfo",
                    "nvmlDeviceGetUtilizationRates",
                    "nvmlDeviceGetTemperature",
                    "nvmlDeviceGetClockInfo",
                    "nvmlDeviceGetMaxClockInfo",
                    "nvmlDeviceGetPersistenceMode",
                    "nvmlDeviceGetComputeMode",
                    "nvmlDeviceGetNumGpuCores",
                }
            )
        )
        result = NVMLProbe().probe()
        assert result.status is DetectionStatus.OK
        gpu = result.gpus[0]
        assert gpu.uuid is None
        assert gpu.temperature_celsius is None
        assert gpu.memory_total_bytes is not None, "supported fields must survive"
