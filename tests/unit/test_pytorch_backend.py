"""Tests for the PyTorch backend.

CPU tests execute real PyTorch inference. CUDA tests are STRUCTURAL: they patch
torch's CUDA queries or use a fake event API to verify validation logic and the
CUDA-event timing contract. None of them execute on an NVIDIA GPU, and none
should be read as evidence that the CUDA path works on real hardware.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from gpu_benchlab.backends.pytorch import CudaEventTimer, PyTorchBackend  # noqa: E402
from gpu_benchlab.core.backend import DeviceKind  # noqa: E402
from gpu_benchlab.core.config import ExperimentConfig, parse_config  # noqa: E402
from gpu_benchlab.core.engine import CPU_RESULT_NOTE, BenchmarkEngine  # noqa: E402
from gpu_benchlab.core.errors import (  # noqa: E402
    ConfigurationError,
    UnavailableError,
    UnsupportedConfigurationError,
)
from gpu_benchlab.core.schema import BenchmarkResult, BenchmarkStatus  # noqa: E402
from gpu_benchlab.core.timing import TimingMechanism  # noqa: E402
from gpu_benchlab.hardware.detect import detect_environment  # noqa: E402
from gpu_benchlab.models.registry import (  # noqa: E402
    RANDOM_WEIGHTS,
    ModelSpec,
    WeightsSpec,
    register_model,
    unregister_model,
)

pytestmark = pytest.mark.torch

TINY_PARAMS = (3 * 4 * 3 * 3 + 4) + (4 * 10 + 10)  # conv + linear = 162


class TinyNet(nn.Module):  # type: ignore[misc]
    """Small CNN standing in for ResNet-50 so mechanics tests run in milliseconds."""

    def __init__(
        self,
        *,
        raise_on_call: int | None = None,
        exc: Exception | None = None,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 10)
        self.raise_on_call = raise_on_call
        self.exc = exc
        self.transform = transform
        self.calls = 0
        self.inference_mode_seen: list[bool] = []

    def forward(self, x: Any) -> Any:
        self.calls += 1
        self.inference_mode_seen.append(torch.is_inference_mode_enabled())
        if self.raise_on_call is not None and self.calls >= self.raise_on_call:
            raise self.exc or RuntimeError("boom")
        out = self.fc(torch.flatten(self.pool(torch.relu(self.conv(x))), 1))
        return self.transform(out) if self.transform else out


INSTANCES: list[TinyNet] = []


def tiny_spec(
    name: str = "tiny-cnn",
    *,
    expected: int = TINY_PARAMS,
    architecture: str = "TinyNet",
    weights: dict[str, WeightsSpec] | None = None,
    **model_kwargs: Any,
) -> ModelSpec:
    def build() -> TinyNet:
        model = TinyNet(**model_kwargs)
        INSTANCES.append(model)
        return model

    return ModelSpec(
        name=name,
        family="vision",
        architecture=architecture,
        build=build,
        input_shape=(3, 16, 16),
        output_shape=(10,),
        expected_parameters=expected,
        default_weights=RANDOM_WEIGHTS,
        weights=weights or {},
        test_only=True,
    )


@pytest.fixture
def register() -> Iterator[Callable[[ModelSpec], None]]:
    names: list[str] = []

    def _register(spec: ModelSpec) -> None:
        register_model(spec)
        names.append(spec.name)

    yield _register
    for n in names:
        unregister_model(n)
    INSTANCES.clear()


@pytest.fixture(scope="module")
def environment():
    return detect_environment(include_frameworks=False)


@pytest.fixture
def engine(environment) -> BenchmarkEngine:
    return BenchmarkEngine(environment=environment)


def make_config(
    model: str = "tiny-cnn",
    *,
    device: str | None = "cpu",
    precision: str = "fp32",
    batch_size: int = 2,
    warmup: int = 2,
    iterations: int = 5,
    weights: str | None = None,
    options: dict[str, Any] | None = None,
) -> ExperimentConfig:
    data: dict[str, Any] = {
        "name": "pytorch-test",
        "backend": "pytorch",
        "precision": precision,
        "batch_size": batch_size,
        "model": {"name": model, **({"weights": weights} if weights else {})},
        "benchmark": {"warmup_iterations": warmup, "measurement_iterations": iterations},
        "backend_options": options or {},
    }
    if device is not None:
        data["device"] = device
    return parse_config(data)


def fp32_state() -> dict[str, Any]:
    b = torch.backends
    return {
        "conv": b.cudnn.conv.fp32_precision,
        "rnn": b.cudnn.rnn.fp32_precision,
        "matmul": b.cuda.matmul.fp32_precision,
        "cudnn": b.cudnn.fp32_precision,
        "global": b.fp32_precision,
        "benchmark": b.cudnn.benchmark,
        "threads": torch.get_num_threads(),
    }


# ---------------------------------------------------------------------------------
# construction (no torch needed for these checks, but they live with the backend)
# ---------------------------------------------------------------------------------


class TestConstruction:
    def test_device_is_required(self, register) -> None:
        register(tiny_spec())
        with pytest.raises(ConfigurationError, match="explicit device"):
            PyTorchBackend(make_config(device=None))

    @pytest.mark.parametrize("device", ["gpu", "cuda:x", "tpu", "cuda:-1"])
    def test_bad_device_rejected(self, register, device: str) -> None:
        register(tiny_spec())
        with pytest.raises(ConfigurationError, match="Unrecognised device"):
            PyTorchBackend(make_config(device=device))

    def test_bare_cuda_means_cuda_0(self, register) -> None:
        register(tiny_spec())
        assert PyTorchBackend(make_config(device="cuda")).descriptor.device == "cuda:0"

    def test_unknown_option_rejected(self, register) -> None:
        register(tiny_spec())
        with pytest.raises(ConfigurationError, match="backend_options"):
            PyTorchBackend(make_config(options={"cudnn_benchmrk": True}))

    def test_unknown_model_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown model"):
            PyTorchBackend(make_config(model="nope"))

    def test_unknown_weights_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown weights"):
            PyTorchBackend(make_config(model="resnet50", weights="DEFAULT"))


# ---------------------------------------------------------------------------------
# real CPU execution
# ---------------------------------------------------------------------------------


class TestCpuExecution:
    def test_real_run_succeeds(self, engine, register) -> None:
        register(tiny_spec())
        result = engine.run(PyTorchBackend(make_config()), make_config())
        assert result.status is BenchmarkStatus.OK, result.errors
        assert len(result.raw_samples.latency_ms) == 5
        assert len(result.raw_samples.warmup_latency_ms) == 2

    def test_labelled_as_cpu_and_not_simulated(self, engine, register) -> None:
        register(tiny_spec())
        result = engine.run(PyTorchBackend(make_config()), make_config())
        assert result.device_kind is DeviceKind.CPU
        assert result.is_simulated is False
        assert result.is_real_measurement is True
        assert result.is_gpu_measurement is False
        assert CPU_RESULT_NOTE in result.notes

    def test_cpu_uses_unsynchronized_wall_clock_and_no_secondary(self, engine, register) -> None:
        register(tiny_spec())
        result = engine.run(PyTorchBackend(make_config()), make_config())
        assert result.timing_mechanism is TimingMechanism.WALL_CLOCK
        assert result.secondary_timing_mechanism is None

    def test_samples_are_real_positive_durations(self, engine, register) -> None:
        register(tiny_spec())
        result = engine.run(PyTorchBackend(make_config()), make_config())
        assert all(s > 0 for s in result.raw_samples.latency_ms)

    def test_every_iteration_executes_the_model(self, engine, register) -> None:
        """1 sanity pass + 2 warmup + 5 measured. Nothing skipped, nothing faked."""
        register(tiny_spec())
        engine.run(PyTorchBackend(make_config()), make_config())
        assert INSTANCES[-1].calls == 1 + 2 + 5

    def test_inference_mode_active_for_sanity_warmup_and_measurement(
        self, engine, register
    ) -> None:
        register(tiny_spec())
        engine.run(PyTorchBackend(make_config()), make_config())
        seen = INSTANCES[-1].inference_mode_seen
        assert len(seen) == 8
        assert all(seen)

    def test_model_info_and_parameter_check(self, engine, register) -> None:
        register(tiny_spec())
        result = engine.run(PyTorchBackend(make_config()), make_config())
        info = result.model_info
        assert info is not None
        assert info.architecture == "TinyNet"
        assert info.parameter_count == info.expected_parameter_count == TINY_PARAMS
        assert info.weights == RANDOM_WEIGHTS and info.random_init_seed == 0

    def test_settings_record_effective_fp32_policy(self, engine, register) -> None:
        register(tiny_spec())
        s = engine.run(PyTorchBackend(make_config()), make_config()).backend.settings
        assert s["fp32_precision.cudnn.conv"] == "ieee"
        assert s["fp32_precision.cudnn.rnn"] == "ieee"
        assert s["fp32_precision.cuda.matmul"] == "ieee"
        assert s["dtype"] == "float32"
        assert s["cuda_flags_applicable"] is False
        assert s["sanity_check"] == "passed"
        assert s["torch_version"] == torch.__version__

    def test_effective_input_shape_is_recorded(self, engine, register) -> None:
        """Regression: with input_shape omitted, the executed shape was recorded nowhere."""
        register(tiny_spec())
        cfg = make_config(batch_size=3)
        assert cfg.model.input_shape is None
        result = engine.run(PyTorchBackend(cfg), cfg)
        assert result.backend.settings["input_shape"] == "3x3x16x16"
        assert result.backend.settings["input_seed"] == 0

    def test_options_are_applied_and_recorded(self, engine, register) -> None:
        register(tiny_spec())
        opts = {"num_threads": 1, "cudnn_benchmark": False, "channels_last": True}
        s = engine.run(
            PyTorchBackend(make_config(options=opts)), make_config(options=opts)
        ).backend.settings
        assert s["num_threads"] == 1
        assert s["cudnn.benchmark"] is False
        assert s["memory_format"] == "channels_last"

    @pytest.mark.parametrize(("precision", "dtype"), [("fp16", "float16"), ("bf16", "bfloat16")])
    def test_reduced_precision_is_real_or_honestly_unsupported(
        self, engine, register, precision: str, dtype: str
    ) -> None:
        """Probed, not assumed. On this machine's torch build both probes pass."""
        register(tiny_spec())
        cfg = make_config(precision=precision)
        result = engine.run(PyTorchBackend(cfg), cfg)
        if result.status is BenchmarkStatus.OK:
            assert result.backend.settings["dtype"] == dtype
        else:
            assert result.status is BenchmarkStatus.UNSUPPORTED


class TestGlobalStateHygiene:
    def test_restored_after_success(self, engine, register) -> None:
        register(tiny_spec())
        before = fp32_state()
        opts = {"num_threads": 1, "cudnn_benchmark": not before["benchmark"]}
        engine.run(PyTorchBackend(make_config(options=opts)), make_config(options=opts))
        assert fp32_state() == before

    def test_restored_after_failure(self, engine, register) -> None:
        register(tiny_spec(raise_on_call=3))
        before = fp32_state()
        result = engine.run(
            PyTorchBackend(make_config(options={"num_threads": 1})),
            make_config(options={"num_threads": 1}),
        )
        assert result.status is BenchmarkStatus.FAILED
        assert fp32_state() == before

    def test_legacy_flags_still_readable_afterwards(self, engine, register) -> None:
        """Reading legacy allow_tf32 raises while the new-API flags are mixed."""
        register(tiny_spec())
        engine.run(PyTorchBackend(make_config()), make_config())
        _ = torch.backends.cudnn.allow_tf32  # must not raise

    def test_global_rng_untouched_by_random_init(self, engine, register) -> None:
        register(tiny_spec())
        torch.manual_seed(1234)
        expected = torch.rand(3)
        torch.manual_seed(1234)
        engine.run(PyTorchBackend(make_config()), make_config())
        assert torch.equal(torch.rand(3), expected)


class TestFailureMapping:
    def test_parameter_count_mismatch_fails_at_load(self, engine, register) -> None:
        register(tiny_spec(expected=999))
        result = engine.run(PyTorchBackend(make_config()), make_config())
        assert result.status is BenchmarkStatus.FAILED
        assert result.errors[0].phase == "load"
        assert "parameters" in result.errors[0].message

    def test_architecture_mismatch_fails_at_load(self, engine, register) -> None:
        register(tiny_spec(architecture="ResNet"))
        result = engine.run(PyTorchBackend(make_config()), make_config())
        assert result.errors[0].phase == "load"
        assert "TinyNet" in result.errors[0].message

    def test_wrong_output_shape_fails_sanity_check(self, engine, register) -> None:
        register(tiny_spec(transform=lambda out: out[:, :3]))
        result = engine.run(PyTorchBackend(make_config()), make_config())
        assert result.errors[0].phase == "prepare"
        assert "shape" in result.errors[0].message
        assert INSTANCES[-1].calls == 1, "nothing is timed after a failed sanity check"

    def test_non_finite_output_fails_sanity_check(self, engine, register) -> None:
        register(tiny_spec(transform=lambda out: out * float("nan")))
        result = engine.run(PyTorchBackend(make_config()), make_config())
        assert result.errors[0].phase == "prepare"
        assert "non-finite" in result.errors[0].message

    def test_precision_that_did_not_take_effect_fails(self, engine, register) -> None:
        register(tiny_spec(transform=lambda out: out.double()))
        result = engine.run(PyTorchBackend(make_config()), make_config())
        assert result.errors[0].phase == "prepare"
        assert "did not take effect" in result.errors[0].message

    def test_torch_oom_during_measurement(self, engine, register) -> None:
        register(tiny_spec(raise_on_call=5, exc=torch.OutOfMemoryError("CUDA out of memory.")))
        result = engine.run(PyTorchBackend(make_config()), make_config(warmup=2, iterations=5))
        assert result.status is BenchmarkStatus.FAILED
        assert result.errors[0].type == "OutOfMemoryError"
        assert result.errors[0].phase == "measure"

    @pytest.mark.parametrize("precision", ["int8", "fp8", "int4", "fp4"])
    def test_quantized_precisions_unsupported(self, engine, register, precision: str) -> None:
        register(tiny_spec())
        cfg = make_config(precision=precision)
        result = engine.run(PyTorchBackend(cfg), cfg)
        assert result.status is BenchmarkStatus.UNSUPPORTED
        assert result.errors[0].phase == "validate"
        assert "TensorRT" in result.errors[0].message

    def test_tf32_on_cpu_unsupported(self, engine, register) -> None:
        register(tiny_spec())
        cfg = make_config(precision="tf32")
        result = engine.run(PyTorchBackend(cfg), cfg)
        assert result.status is BenchmarkStatus.UNSUPPORTED

    def test_failed_cpu_probe_is_unsupported(self, register, environment, monkeypatch) -> None:
        register(tiny_spec())

        def refuse(*a: Any, **k: Any) -> Any:
            raise RuntimeError('"slow_conv2d_cpu" not implemented for Half')

        monkeypatch.setattr(torch.nn.functional, "conv2d", refuse)
        backend = PyTorchBackend(make_config(precision="fp16"))
        with pytest.raises(UnsupportedConfigurationError, match="not implemented for Half"):
            backend.validate(make_config(precision="fp16"), environment)


class TestPinnedWeights:
    def test_sha256_of_loaded_file_is_recorded(
        self, engine, register, tmp_path: Path, monkeypatch
    ) -> None:
        source = TinyNet()
        path = tmp_path / "state.pth"
        torch.save(source.state_dict(), path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        cache = tmp_path / "cache"
        (cache / "weights").mkdir(parents=True)
        filename = f"tiny-{digest[:8]}.pth"
        (cache / "weights" / filename).write_bytes(path.read_bytes())
        monkeypatch.setenv("GPU_BENCHLAB_CACHE", str(cache))

        url = f"https://example.invalid/{filename}"
        register(tiny_spec(weights={"LOCAL": WeightsSpec(url=url, sha256_prefix=digest[:8])}))
        cfg = make_config(weights="LOCAL", options={"allow_download": False})
        result = engine.run(PyTorchBackend(cfg), cfg)

        assert result.status is BenchmarkStatus.OK, result.errors
        assert result.model_info is not None
        assert result.model_info.weights_sha256 == digest
        assert result.model_info.weights_url == url
        assert result.model_info.random_init_seed is None
        assert result.backend.settings["weights_downloaded_this_run"] is False
        loaded = INSTANCES[-1]
        assert torch.equal(loaded.fc.weight.float(), source.fc.weight), "pinned weights loaded"

    def test_missing_weights_without_download_is_unavailable(
        self, engine, register, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("GPU_BENCHLAB_CACHE", str(tmp_path))
        spec = WeightsSpec(url="https://example.invalid/tiny-abc.pth", sha256_prefix="abc")
        register(tiny_spec(weights={"LOCAL": spec}))
        cfg = make_config(weights="LOCAL", options={"allow_download": False})
        result = engine.run(PyTorchBackend(cfg), cfg)
        assert result.status is BenchmarkStatus.UNAVAILABLE
        assert result.errors[0].phase == "validate"


# ---------------------------------------------------------------------------------
# CUDA: genuine behaviour on THIS machine, then structural checks
# ---------------------------------------------------------------------------------


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA path")
class TestCudaUnavailableOnThisMachine:
    def test_cuda_request_is_unavailable_not_run_on_cpu(self, engine, register) -> None:
        register(tiny_spec())
        cfg = make_config(device="cuda:0")
        result = engine.run(PyTorchBackend(cfg), cfg)
        assert result.status is BenchmarkStatus.UNAVAILABLE
        assert result.errors[0].phase == "validate"
        assert "Not falling back to CPU" in result.errors[0].message
        assert result.device_kind is DeviceKind.CUDA
        assert result.backend.device == "cuda:0"
        assert result.raw_samples.latency_ms == []
        assert result.timing_mechanism is None
        assert result.model_info is None
        assert not INSTANCES, "the model was never even built"

    def test_reason_includes_nvidia_detection_status(self, engine, register, environment) -> None:
        register(tiny_spec())
        cfg = make_config(device="cuda:0")
        message = engine.run(PyTorchBackend(cfg), cfg).errors[0].message
        assert environment.detection_status.value in message


class FakeCuda:
    """Patches torch's CUDA queries to describe a device that does not exist here."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        capability: tuple[int, int],
        available: bool = True,
        count: int = 1,
    ) -> None:
        monkeypatch.setattr(torch.version, "cuda", "13.0")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: capability)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: f"Fake SM{capability}")


class TestCudaValidationStructural:
    """Validation logic only. No kernel ever runs on a GPU in these tests."""

    def test_cuda_build_without_device_is_unavailable(
        self, register, environment, monkeypatch
    ) -> None:
        register(tiny_spec())
        FakeCuda(monkeypatch, capability=(8, 9), available=False)
        with pytest.raises(UnavailableError, match="no usable CUDA device"):
            PyTorchBackend(make_config(device="cuda:0")).validate(make_config(), environment)

    def test_index_out_of_range_is_unavailable(self, register, environment, monkeypatch) -> None:
        register(tiny_spec())
        FakeCuda(monkeypatch, capability=(8, 9), count=1)
        with pytest.raises(UnavailableError, match="only 1 CUDA device"):
            PyTorchBackend(make_config(device="cuda:3")).validate(make_config(), environment)

    @pytest.mark.parametrize("precision", ["bf16", "tf32"])
    def test_precision_below_capability_is_unsupported(
        self, register, environment, monkeypatch, precision: str
    ) -> None:
        register(tiny_spec())
        FakeCuda(monkeypatch, capability=(7, 0))
        backend = PyTorchBackend(make_config(device="cuda:0", precision=precision))
        with pytest.raises(UnsupportedConfigurationError, match=r"SM 7.0"):
            backend.validate(make_config(), environment)

    def test_supported_precision_records_capability(
        self, register, environment, monkeypatch
    ) -> None:
        register(tiny_spec())
        FakeCuda(monkeypatch, capability=(8, 9))
        backend = PyTorchBackend(make_config(device="cuda:0", precision="fp16"))
        backend.validate(make_config(), environment)
        s = backend.descriptor.settings
        assert s["cuda_capability"] == "8.9"
        assert s["tensor_core_for_precision"] is True


class FakeEvent:
    def __init__(self, name: str, log: list[str], elapsed: float) -> None:
        self.name, self.log, self.elapsed = name, log, elapsed

    def record(self, stream: Any) -> None:
        self.log.append(f"{self.name}.record({stream})")

    def synchronize(self) -> None:
        self.log.append(f"{self.name}.synchronize")

    def elapsed_time(self, other: FakeEvent) -> float:
        self.log.append(f"{self.name}.elapsed_time({other.name})")
        return self.elapsed


def fake_torch(log: list[str], elapsed: float = 4.25) -> Any:
    names = iter(["start", "end"])
    return SimpleNamespace(
        cuda=SimpleNamespace(
            current_stream=lambda device: "stream0",
            Event=lambda enable_timing: FakeEvent(next(names), log, elapsed),
            synchronize=lambda device: log.append(f"device_sync({device})"),
        )
    )


class TestCudaEventTimerStructural:
    """The timing contract, verified against a fake event API -- not a GPU."""

    def test_call_order(self) -> None:
        log: list[str] = []
        timer = CudaEventTimer(fake_torch(log), "cuda:0")
        timer.start()
        assert log == ["device_sync(cuda:0)", "start.record(stream0)"]
        log.clear()
        timer.stop()
        assert log == ["end.record(stream0)", "end.synchronize", "start.elapsed_time(end)"]

    def test_primary_is_event_time_not_host_time(self) -> None:
        timer = CudaEventTimer(fake_torch([], elapsed=4.25), "cuda:0")
        timer.start()
        assert timer.stop() == 4.25
        assert timer.mechanism is TimingMechanism.CUDA_EVENT

    def test_host_time_is_secondary_and_drained(self) -> None:
        timer = CudaEventTimer(fake_torch([]), "cuda:0")
        for _ in range(3):
            timer.start()
            timer.stop()
        drained = timer.drain_secondary()
        assert drained is not None
        mechanism, samples = drained
        assert mechanism is TimingMechanism.WALL_CLOCK_SYNCHRONIZED
        assert len(samples) == 3 and all(s >= 0 for s in samples)
        assert timer.drain_secondary() == (TimingMechanism.WALL_CLOCK_SYNCHRONIZED, [])

    def test_stop_before_start_raises(self) -> None:
        with pytest.raises(RuntimeError, match="before start"):
            CudaEventTimer(fake_torch([]), "cuda:0").stop()


# ---------------------------------------------------------------------------------
# real ResNet-50
# ---------------------------------------------------------------------------------


@pytest.mark.slow
class TestRealResNet50:
    def test_random_init_resnet50_runs_on_cpu(self, engine) -> None:
        cfg = make_config("resnet50", weights="random", batch_size=1, warmup=1, iterations=3)
        result = engine.run(PyTorchBackend(cfg), cfg)
        assert result.status is BenchmarkStatus.OK, result.errors
        assert result.model_info is not None
        assert result.model_info.architecture == "ResNet"
        assert result.model_info.parameter_count == 25_557_032
        restored = BenchmarkResult.model_validate_json(result.model_dump_json())
        assert restored == result
