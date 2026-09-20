"""Tests for the TensorRT build layer and backend.

Three kinds, labelled so none can be mistaken for another:

* STRUCTURAL: a fake ``tensorrt`` + CUDA runtime, for wiring that must hold regardless
  of hardware (precision policy, optimization profile, lifecycle order, cleanup,
  manifest contents, error surfacing). These prove wiring, **not** GPU behaviour.
* CONFIG: pure configuration validation, no fake needed.
* REAL: skipped unless TensorRT is genuinely installed.

The whole module runs with TensorRT absent, which is the normal state of the CPU
development machine and of CI.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_benchlab.core.backend import DeviceKind
from gpu_benchlab.core.config import ExperimentConfig, parse_config
from gpu_benchlab.core.engine import BenchmarkEngine
from gpu_benchlab.core.errors import (
    BackendError,
    ConfigurationError,
    UnavailableError,
)
from gpu_benchlab.core.schema import BenchmarkStatus
from gpu_benchlab.core.timing import TimingMechanism
from gpu_benchlab.export import tensorrt_build
from gpu_benchlab.export.tensorrt_build import TensorRtManifest, TrtBuildConfig
from gpu_benchlab.hardware.detect import detect_environment

CHW = (3, 224, 224)


# ============================================================================== fakes


class FakeTensor:
    def __init__(self, name: str, shape: list[int]) -> None:
        self.name = name
        self.shape = shape
        self.dtype = "DataType.FLOAT"


class FakeNetwork:
    def __init__(self, fake: FakeTrt) -> None:
        self.fake = fake
        self.num_inputs = 1
        self.num_outputs = 1
        self.num_layers = 126

    def get_input(self, index: int) -> FakeTensor:
        return FakeTensor("input", [-1, *CHW])

    def get_output(self, index: int) -> FakeTensor:
        return FakeTensor("logits", [-1, 1000])

    def get_flag(self, flag: Any) -> bool:
        return flag == "STRONGLY_TYPED"


class FakeProfile:
    def __init__(self) -> None:
        self.shapes: dict[str, tuple[Any, Any, Any]] = {}

    def set_shape(self, name: str, minimum: Any, optimum: Any, maximum: Any) -> None:
        self.shapes[name] = (tuple(minimum), tuple(optimum), tuple(maximum))


class FakeBuilderConfig:
    def __init__(self, fake: FakeTrt) -> None:
        self.fake = fake
        self.flags = {"TF32": True}  # TensorRT enables TF32 by default
        self.builder_optimization_level = 3
        self.pool_limits = {"WORKSPACE": 23_659_151_360}
        self.profiles: list[FakeProfile] = []

    def get_flag(self, flag: str) -> bool:
        return self.flags.get(flag, False)

    def set_flag(self, flag: str) -> None:
        self.flags[flag] = True

    def clear_flag(self, flag: str) -> None:
        self.flags[flag] = False

    def set_memory_pool_limit(self, kind: str, value: int) -> None:
        self.pool_limits[kind] = value

    def get_memory_pool_limit(self, kind: str) -> int:
        return self.pool_limits[kind]

    def add_optimization_profile(self, profile: FakeProfile) -> None:
        self.profiles.append(profile)
        self.fake.profile = profile


class FakeParser:
    def __init__(self, network: Any, logger: Any, fake: FakeTrt) -> None:
        self.fake = fake
        self.num_errors = len(fake.parser_errors)

    def parse(self, data: bytes) -> bool:
        self.num_errors = len(self.fake.parser_errors)
        return not self.fake.parser_errors

    def get_error(self, index: int) -> str:
        return self.fake.parser_errors[index]


class FakeEngine:
    def __init__(self, fake: FakeTrt) -> None:
        self.fake = fake

    def get_tensor_profile_shape(self, name: str, index: int) -> list[tuple[int, ...]]:
        return [
            (self.fake.min_batch, *CHW),
            (self.fake.opt_batch, *CHW),
            (self.fake.max_batch, *CHW),
        ]

    def create_execution_context(self) -> Any:
        return None if self.fake.context_fails else FakeContext(self.fake)


class FakeContext:
    def __init__(self, fake: FakeTrt) -> None:
        self.fake = fake

    def set_input_shape(self, name: str, shape: Any) -> None:
        self.fake.runtime_shape = tuple(shape)

    def set_tensor_address(self, name: str, address: int) -> None:
        self.fake.addresses[name] = address

    def execute_async_v3(self, stream: Any) -> bool:
        self.fake.executions += 1
        return not self.fake.execute_fails


class FakeBuilder:
    def __init__(self, fake: FakeTrt) -> None:
        self.fake = fake

    def create_network(self, flags: int) -> FakeNetwork:
        return FakeNetwork(self.fake)

    def create_builder_config(self) -> FakeBuilderConfig:
        config = FakeBuilderConfig(self.fake)
        self.fake.builder_config = config
        return config

    def create_optimization_profile(self) -> FakeProfile:
        return FakeProfile()

    def build_serialized_network(self, network: Any, config: Any) -> Any:
        return None if self.fake.build_fails else self.fake.plan_bytes


class FakeRuntime:
    def __init__(self, logger: Any, fake: FakeTrt) -> None:
        self.fake = fake

    def deserialize_cuda_engine(self, plan: bytes) -> Any:
        return None if self.fake.deserialize_fails else FakeEngine(self.fake)


class FakeTrt:
    """A stand-in for the ``tensorrt`` module, shaped like TensorRT 11.x."""

    def __init__(self) -> None:
        self.__version__ = "11.3.0.99"
        self.__file__ = "/fake/tensorrt/__init__.py"
        self.parser_errors: list[str] = []
        self.build_fails = False
        self.deserialize_fails = False
        self.context_fails = False
        self.execute_fails = False
        self.plan_bytes = b"FAKE-TRT-ENGINE" * 16
        self.min_batch, self.opt_batch, self.max_batch = 1, 8, 8
        self.builder_config: FakeBuilderConfig | None = None
        self.profile: FakeProfile | None = None
        self.runtime_shape: tuple[int, ...] | None = None
        self.addresses: dict[str, int] = {}
        self.executions = 0

        self.BuilderFlag = SimpleNamespace(
            TF32="TF32",
            STRICT_NANS="STRICT_NANS",
            DISABLE_TIMING_CACHE="DISABLE_TIMING_CACHE",
            DISABLE_COMPILATION_CACHE="DISABLE_COMPILATION_CACHE",
        )
        self.MemoryPoolType = SimpleNamespace(WORKSPACE="WORKSPACE")
        self.NetworkDefinitionCreationFlag = SimpleNamespace(STRONGLY_TYPED="STRONGLY_TYPED")
        self.Logger = lambda severity=None: SimpleNamespace(severity=severity)
        self.Logger.WARNING = "WARNING"  # type: ignore[attr-defined]

    def Builder(self, logger: Any) -> FakeBuilder:  # noqa: N802 - mirrors the TensorRT API
        return FakeBuilder(self)

    def OnnxParser(self, network: Any, logger: Any) -> FakeParser:  # noqa: N802
        return FakeParser(network, logger, self)

    def Runtime(self, logger: Any) -> FakeRuntime:  # noqa: N802
        return FakeRuntime(logger, self)


class FakeCudart:
    """A stand-in for ``cuda.bindings.runtime``, tracking allocations and streams."""

    def __init__(self) -> None:
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.streams: list[int] = []
        self.destroyed_streams: list[int] = []
        self.events: list[int] = []
        self.destroyed_events: list[int] = []
        self.device_count = 1
        self.copies: list[tuple[Any, int]] = []
        self.synchronizations = 0
        self._next = 0x1000
        self.cudaMemcpyKind = SimpleNamespace(
            cudaMemcpyHostToDevice="H2D", cudaMemcpyDeviceToHost="D2H"
        )
        self.cudaMemoryType = SimpleNamespace(cudaMemoryTypeDevice="device")

    def _handle(self) -> int:
        self._next += 0x100
        return self._next

    def cudaGetErrorString(self, error: Any) -> tuple[int, bytes]:  # noqa: N802
        return 0, b"fake cuda error"

    def cudaGetDeviceCount(self) -> tuple[int, int]:  # noqa: N802
        return 0, self.device_count

    def cudaSetDevice(self, index: int) -> tuple[int]:  # noqa: N802
        return (0,)

    def cudaGetDeviceProperties(self, index: int) -> tuple[int, Any]:  # noqa: N802
        return 0, SimpleNamespace(
            name=b"FAKE L4", major=8, minor=9, multiProcessorCount=58, totalGlobalMem=2**34
        )

    def cudaRuntimeGetVersion(self) -> tuple[int, int]:  # noqa: N802
        return 0, 13040

    def cudaDriverGetVersion(self) -> tuple[int, int]:  # noqa: N802
        return 0, 13000

    def cudaMalloc(self, size: int) -> tuple[int, int]:  # noqa: N802
        handle = self._handle()
        self.allocations[handle] = size
        return 0, handle

    def cudaFree(self, pointer: int) -> tuple[int]:  # noqa: N802
        self.freed.append(pointer)
        return (0,)

    def cudaMemcpy(self, dst: Any, src: Any, size: int, kind: Any) -> tuple[int]:  # noqa: N802
        if kind == "D2H":
            # A real copy writes bytes into the destination; zeros stand in for logits.
            # Without this the backend would read its own NaN pre-fill and fail, which
            # is exactly the behaviour test_a_missing_device_copy_is_caught relies on.
            ctypes.memset(int(dst), 0, size)
        self.copies.append((kind, size))
        return (0,)

    def cudaStreamCreate(self) -> tuple[int, int]:  # noqa: N802
        handle = self._handle()
        self.streams.append(handle)
        return 0, handle

    def cudaStreamSynchronize(self, stream: Any) -> tuple[int]:  # noqa: N802
        self.synchronizations += 1
        return (0,)

    def cudaStreamDestroy(self, stream: Any) -> tuple[int]:  # noqa: N802
        self.destroyed_streams.append(stream)
        return (0,)

    def cudaEventCreate(self) -> tuple[int, int]:  # noqa: N802
        handle = self._handle()
        self.events.append(handle)
        return 0, handle

    def cudaEventRecord(self, event: Any, stream: Any) -> tuple[int]:  # noqa: N802
        return (0,)

    def cudaEventSynchronize(self, event: Any) -> tuple[int]:  # noqa: N802
        return (0,)

    def cudaEventElapsedTime(self, start: Any, end: Any) -> tuple[int, float]:  # noqa: N802
        return 0, 2.5

    def cudaEventDestroy(self, event: Any) -> tuple[int]:  # noqa: N802
        self.destroyed_events.append(event)
        return (0,)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeTrt:
    trt, cudart = FakeTrt(), FakeCudart()
    monkeypatch.setattr(tensorrt_build, "import_tensorrt", lambda: (trt, cudart))
    from gpu_benchlab.backends import tensorrt_backend

    monkeypatch.setattr(tensorrt_backend, "import_tensorrt", lambda: (trt, cudart))
    trt.cudart = cudart  # type: ignore[attr-defined]
    return trt


@pytest.fixture
def environment():
    return detect_environment(include_frameworks=False)


def cfg(
    device: str | None = "cuda:0",
    *,
    batch: int = 1,
    precision: str = "fp32",
    options: dict[str, Any] | None = None,
) -> ExperimentConfig:
    data: dict[str, Any] = {
        "name": "trt-test",
        "backend": "tensorrt",
        "precision": precision,
        "batch_size": batch,
        "model": {"name": "resnet50"},
        "benchmark": {"warmup_iterations": 2, "measurement_iterations": 3},
        "backend_options": options or {},
    }
    if device is not None:
        data["device"] = device
    return parse_config(data)


def onnx_manifest_stub(tmp_path: Path) -> tuple[Path, Any]:
    """A canonical-looking ONNX artifact and its manifest, without touching torch."""
    import hashlib

    from gpu_benchlab.core.backend import ModelInfo
    from gpu_benchlab.core.provenance import capture_provenance
    from gpu_benchlab.core.schema import utc_now_iso
    from gpu_benchlab.export.onnx_export import (
        ArtifactInfo,
        ExporterInfo,
        OnnxManifest,
        TensorSpec,
    )

    artifact = tmp_path / "resnet50-canonical.onnx"
    artifact.write_bytes(b"ONNX-BYTES")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = OnnxManifest(
        created_utc=utc_now_iso(),
        artifact=ArtifactInfo(filename=artifact.name, sha256=digest, bytes=artifact.stat().st_size),
        model=ModelInfo(
            name="resnet50",
            architecture="ResNet",
            weights="IMAGENET1K_V2",
            weights_sha256="11ad3fa6" + "0" * 56,
            parameter_count=25_557_032,
            expected_parameter_count=25_557_032,
        ),
        exporter=ExporterInfo(
            torch_version="2.14.0+cu130",
            torchvision_version="0.29.0",
            onnx_version="1.23.0",
            onnxscript_version="0.7.2",
            opset=20,
            optimize=True,
            trace_batch_size=1,
        ),
        input=TensorSpec(name="input", dtype="float32", shape=["batch", 3, 224, 224]),
        output=TensorSpec(name="logits", dtype="float32", shape=["batch", 1000]),
        export_seconds=1.0,
        provenance=capture_provenance(),
    )
    return artifact, manifest


# ========================================================================= config only


class TestConfiguration:
    def test_cpu_device_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match=r"(?i)cuda"):
            from gpu_benchlab.backends.tensorrt_backend import TensorRtBackend

            TensorRtBackend(cfg("cpu"))

    def test_missing_device_is_rejected(self) -> None:
        from gpu_benchlab.backends.tensorrt_backend import TensorRtBackend

        with pytest.raises(ConfigurationError, match="explicit CUDA device"):
            TensorRtBackend(cfg(None))

    def test_bare_cuda_means_device_zero(self) -> None:
        from gpu_benchlab.backends.tensorrt_backend import TensorRtBackend

        assert TensorRtBackend(cfg("cuda")).descriptor.device == "cuda:0"

    def test_profile_must_be_ordered(self) -> None:
        from gpu_benchlab.backends.tensorrt_backend import TensorRtBackend

        with pytest.raises(ConfigurationError, match="min_batch <= opt_batch <= max_batch"):
            TensorRtBackend(cfg(options={"min_batch": 4, "opt_batch": 2, "max_batch": 8}))

    def test_unknown_option_is_rejected(self) -> None:
        from gpu_benchlab.backends.tensorrt_backend import TensorRtBackend

        with pytest.raises(ConfigurationError, match="Invalid backend_options"):
            TensorRtBackend(cfg(options={"fp16": True}))

    def test_descriptor_is_cuda_before_validate(self) -> None:
        from gpu_benchlab.backends.tensorrt_backend import TensorRtBackend

        descriptor = TensorRtBackend(cfg()).descriptor
        assert descriptor.device_kind is DeviceKind.CUDA and not descriptor.is_simulated


class TestUnavailable:
    def test_missing_tensorrt_is_unavailable_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch, environment
    ) -> None:
        """REAL-ish: the import failure path, with tensorrt genuinely absent or faked out."""
        import builtins

        real_import = builtins.__import__

        def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "tensorrt":
                raise ImportError("No module named 'tensorrt'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        with pytest.raises(UnavailableError, match=r"\[tensorrt\] extra"):
            tensorrt_build.import_tensorrt()

    def test_engine_run_without_tensorrt_is_unavailable(self, environment) -> None:
        """The engine records unavailable rather than raising, when TensorRT is missing."""
        from gpu_benchlab.backends.tensorrt_backend import TensorRtBackend

        try:
            import tensorrt  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("TensorRT is installed; this asserts the absent-package path")
        config = cfg()
        result = BenchmarkEngine(environment=environment).run(TensorRtBackend(config), config)
        assert result.status is BenchmarkStatus.UNAVAILABLE
        assert result.errors[0].phase == "validate"
        assert result.raw_samples.latency_ms == []
        assert result.timing_mechanism is None


# ========================================================================== structural


class TestBuildLayer:
    """STRUCTURAL: the build layer against a fake TensorRT."""

    def test_tf32_is_cleared_for_ieee_fp32(self, fake: FakeTrt, tmp_path: Path) -> None:
        artifact, manifest = onnx_manifest_stub(tmp_path)
        _, built = tensorrt_build.build_engine(
            TrtBuildConfig(model="resnet50", weights="IMAGENET1K_V2"),
            artifact,
            manifest,
            directory=tmp_path / "engines",
        )
        assert built.builder_settings["tf32_enabled_by_default"] is True
        assert built.builder_settings["tf32_flag_after_policy"] is False
        assert built.precision_policy == "ieee_fp32"

    def test_tf32_policy_leaves_the_flag_set(self, fake: FakeTrt, tmp_path: Path) -> None:
        """The negative-control policy must differ from IEEE FP32, or it proves nothing."""
        artifact, manifest = onnx_manifest_stub(tmp_path)
        _, built = tensorrt_build.build_engine(
            TrtBuildConfig(model="resnet50", weights="IMAGENET1K_V2", precision="tf32"),
            artifact,
            manifest,
            directory=tmp_path / "engines",
        )
        assert built.builder_settings["tf32_flag_after_policy"] is True

    def test_optimization_profile_uses_configured_batches(
        self, fake: FakeTrt, tmp_path: Path
    ) -> None:
        artifact, manifest = onnx_manifest_stub(tmp_path)
        _, built = tensorrt_build.build_engine(
            TrtBuildConfig(
                model="resnet50", weights="IMAGENET1K_V2", min_batch=1, opt_batch=4, max_batch=16
            ),
            artifact,
            manifest,
            directory=tmp_path / "engines",
        )
        assert built.profile.min == [1, *CHW]
        assert built.profile.opt == [4, *CHW]
        assert built.profile.max == [16, *CHW]
        assert fake.profile is not None
        assert fake.profile.shapes["input"] == ((1, *CHW), (4, *CHW), (16, *CHW))

    def test_parser_errors_are_all_surfaced(self, fake: FakeTrt, tmp_path: Path) -> None:
        fake.parser_errors = ["unsupported op Foo", "node Bar failed"]
        artifact, manifest = onnx_manifest_stub(tmp_path)
        with pytest.raises(BackendError) as excinfo:
            tensorrt_build.build_engine(
                TrtBuildConfig(model="resnet50", weights="IMAGENET1K_V2"),
                artifact,
                manifest,
                directory=tmp_path / "engines",
            )
        assert "unsupported op Foo" in str(excinfo.value)
        assert "node Bar failed" in str(excinfo.value)

    def test_build_failure_is_a_backend_error(self, fake: FakeTrt, tmp_path: Path) -> None:
        fake.build_fails = True
        artifact, manifest = onnx_manifest_stub(tmp_path)
        with pytest.raises(BackendError, match="no engine"):
            tensorrt_build.build_engine(
                TrtBuildConfig(model="resnet50", weights="IMAGENET1K_V2"),
                artifact,
                manifest,
                directory=tmp_path / "engines",
            )

    def test_changed_artifact_is_refused(self, fake: FakeTrt, tmp_path: Path) -> None:
        artifact, manifest = onnx_manifest_stub(tmp_path)
        artifact.write_bytes(b"DIFFERENT-BYTES")
        with pytest.raises(BackendError, match="changed on disk"):
            tensorrt_build.build_engine(
                TrtBuildConfig(model="resnet50", weights="IMAGENET1K_V2"),
                artifact,
                manifest,
                directory=tmp_path / "engines",
            )

    def test_manifest_records_provenance_and_round_trips(
        self, fake: FakeTrt, tmp_path: Path
    ) -> None:
        artifact, manifest = onnx_manifest_stub(tmp_path)
        path, built = tensorrt_build.build_engine(
            TrtBuildConfig(model="resnet50", weights="IMAGENET1K_V2"),
            artifact,
            manifest,
            directory=tmp_path / "engines",
        )
        assert path.exists() and path.suffix == ".plan"
        assert built.engine_size_bytes == len(fake.plan_bytes)
        assert built.tensorrt_version == "11.3.0.99"
        assert built.compute_capability == "8.9"
        assert built.cuda_runtime_version == 13040 and built.cuda_driver_version == 13000
        assert built.source_onnx_sha256 == manifest.artifact.sha256
        assert built.provenance.git_commit
        stored = json.loads((path.with_suffix(".manifest.json")).read_text(encoding="utf-8"))
        assert TensorRtManifest.model_validate(stored) == built

    def test_engine_filename_encodes_what_changes_the_bytes(self) -> None:
        config = TrtBuildConfig(model="resnet50", weights="IMAGENET1K_V2", opt_batch=4)
        stem = config.stem("resnet50-canonical", "11.3.0.99", "8.9")
        assert "trt11.3.0.99" in stem and "sm89" in stem
        assert "b1_4_8" in stem and "ieee_fp32" in stem

    def test_cached_engine_is_reused_only_when_it_matches(self, tmp_path: Path) -> None:
        base = TensorRtManifest(
            created_utc="2026-09-20T00:00:00+00:00",
            engine_file="e.plan",
            engine_sha256="a" * 64,
            engine_size_bytes=10,
            build_seconds=1.0,
            source_onnx_sha256="b" * 64,
            model=onnx_manifest_stub(tmp_path)[1].model,
            input=onnx_manifest_stub(tmp_path)[1].input,
            output=onnx_manifest_stub(tmp_path)[1].output,
            tensorrt_version="11.3.0.99",
            cuda_runtime_version=13040,
            cuda_driver_version=13000,
            gpu_name="NVIDIA L4",
            compute_capability="8.9",
            precision_policy="ieee_fp32",
            profile=tensorrt_build.TrtProfile(min=[1, *CHW], opt=[8, *CHW], max=[8, *CHW]),
            builder_settings={},
            provenance=onnx_manifest_stub(tmp_path)[1].provenance,
        )
        assert base.matches(base)
        assert not base.matches(base.model_copy(update={"tensorrt_version": "11.2.1.2"}))
        assert not base.matches(base.model_copy(update={"compute_capability": "9.0"}))
        assert not base.matches(base.model_copy(update={"precision_policy": "tf32"}))
        assert not base.matches(base.model_copy(update={"source_onnx_sha256": "c" * 64}))


# ================================================================ backend (structural)


@pytest.fixture
def staged_engine(fake: FakeTrt, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A built engine on disk plus its manifest, without touching torch or a GPU."""
    artifact, onnx = onnx_manifest_stub(tmp_path)
    path, manifest = tensorrt_build.build_engine(
        TrtBuildConfig(model="resnet50", weights="IMAGENET1K_V2"),
        artifact,
        onnx,
        directory=tmp_path / "engines",
    )
    from gpu_benchlab.backends import tensorrt_backend

    # load() fetches the canonical artifact and build() makes the engine: both are
    # stubbed so these tests touch neither torch, the weights cache nor a GPU.
    monkeypatch.setattr(
        tensorrt_backend, "ensure_artifact", lambda *a, **k: (artifact, onnx, False)
    )
    monkeypatch.setattr(tensorrt_backend, "ensure_engine", lambda *a, **k: (path, manifest, False))
    return manifest


def run_backend(config: ExperimentConfig, environment: Any) -> Any:
    from gpu_benchlab.backends.tensorrt_backend import TensorRtBackend

    return BenchmarkEngine(environment=environment).run(TensorRtBackend(config), config)


class TestBackendLifecycle:
    """STRUCTURAL: lifecycle, timing wiring and cleanup against the fake TensorRT."""

    def test_successful_run_records_engine_and_precision_provenance(
        self, fake: FakeTrt, staged_engine: Any, environment
    ) -> None:
        result = run_backend(cfg(batch=1), environment)
        assert result.status is BenchmarkStatus.OK
        settings = result.backend.settings
        assert settings["precision_policy"].startswith("ieee_fp32")
        assert settings["builder.tf32_flag_after_policy"] is False
        assert settings["engine_sha256"] == staged_engine.engine_sha256
        assert settings["source_onnx_sha256"] == staged_engine.source_onnx_sha256
        assert settings["io_device_resident"] is True
        assert settings["sanity_check"] == "passed"
        assert result.device_kind is DeviceKind.CUDA

    def test_timing_uses_cuda_events_with_a_host_secondary_series(
        self, fake: FakeTrt, staged_engine: Any, environment
    ) -> None:
        result = run_backend(cfg(batch=1), environment)
        assert result.timing_mechanism is TimingMechanism.CUDA_EVENT
        assert result.secondary_timing_mechanism is TimingMechanism.WALL_CLOCK_SYNCHRONIZED
        assert result.raw_samples.latency_ms == [2.5, 2.5, 2.5]  # from the fake event timer
        assert len(result.raw_samples.secondary_latency_ms) == 3
        assert len(result.raw_samples.warmup_latency_ms) == 2

    def test_engine_build_is_not_inference(
        self, fake: FakeTrt, staged_engine: Any, environment
    ) -> None:
        """Build and deserialization are recorded, and never inside the measured loop."""
        result = run_backend(cfg(batch=1), environment)
        assert result.backend.settings["engine_build_seconds"] >= 0
        assert result.backend.settings["engine_deserialization_ms"] >= 0
        assert all(sample == 2.5 for sample in result.raw_samples.latency_ms)

    def test_execution_uses_the_stream_and_synchronizes(
        self, fake: FakeTrt, staged_engine: Any, environment
    ) -> None:
        run_backend(cfg(batch=1), environment)
        # sanity pass + (2 warmup + 3 measured) enqueues
        assert fake.executions == 1 + 2 + 3
        assert fake.cudart.synchronizations >= 1 + 2 + 3
        assert fake.runtime_shape == (1, 3, 224, 224)
        assert set(fake.addresses) == {"input", "logits"}

    def test_cleanup_releases_device_resources(
        self, fake: FakeTrt, staged_engine: Any, environment
    ) -> None:
        run_backend(cfg(batch=1), environment)
        cudart = fake.cudart
        assert sorted(cudart.freed) == sorted(cudart.allocations)
        assert cudart.destroyed_streams == cudart.streams
        assert sorted(cudart.destroyed_events) == sorted(cudart.events)

    def test_deserialization_failure_is_a_build_phase_error(
        self, fake: FakeTrt, staged_engine: Any, environment
    ) -> None:
        fake.deserialize_fails = True
        result = run_backend(cfg(batch=1), environment)
        assert result.status is BenchmarkStatus.FAILED
        assert result.errors[0].phase == "build"
        assert "deserialize" in result.errors[0].message
        assert result.raw_samples.latency_ms == []

    def test_missing_execution_context_is_a_build_phase_error(
        self, fake: FakeTrt, staged_engine: Any, environment
    ) -> None:
        fake.context_fails = True
        result = run_backend(cfg(batch=1), environment)
        assert result.status is BenchmarkStatus.FAILED
        assert result.errors[0].phase == "build"

    def test_failed_sanity_inference_is_a_prepare_phase_error(
        self, fake: FakeTrt, staged_engine: Any, environment
    ) -> None:
        fake.execute_fails = True
        result = run_backend(cfg(batch=1), environment)
        assert result.status is BenchmarkStatus.FAILED
        assert result.errors[0].phase == "prepare"
        assert result.raw_samples.latency_ms == [], "nothing is measured after a bad sanity run"

    def test_cleanup_still_happens_after_a_failure(
        self, fake: FakeTrt, staged_engine: Any, environment
    ) -> None:
        fake.execute_fails = True
        run_backend(cfg(batch=1), environment)
        cudart = fake.cudart
        assert sorted(cudart.freed) == sorted(cudart.allocations)
        assert cudart.destroyed_streams == cudart.streams

    def test_batch_outside_the_configured_profile_is_unsupported(
        self, fake: FakeTrt, staged_engine: Any, environment
    ) -> None:
        result = run_backend(cfg(batch=16), environment)
        assert result.status is BenchmarkStatus.UNSUPPORTED
        assert result.errors[0].phase == "validate"
        assert "optimization profile" in result.errors[0].message
        assert result.raw_samples.latency_ms == []

    def test_batch_outside_the_engines_own_profile_is_unsupported(
        self, fake: FakeTrt, staged_engine: Any, environment
    ) -> None:
        """The engine's recorded profile is authoritative, not just the requested one."""
        fake.max_batch = 2  # the engine on disk turns out narrower than the configuration
        result = run_backend(cfg(batch=8), environment)
        assert result.status is BenchmarkStatus.UNSUPPORTED
        assert result.errors[0].phase == "prepare"
        assert "outside the engine" in result.errors[0].message

    def test_non_fp32_precision_is_unsupported_for_now(
        self, fake: FakeTrt, staged_engine: Any, environment
    ) -> None:
        result = run_backend(cfg(batch=1, precision="fp16"), environment)
        assert result.status is BenchmarkStatus.UNSUPPORTED
        assert result.errors[0].phase == "validate"
        assert "IEEE FP32 engines only" in result.errors[0].message

    def test_no_cuda_device_is_unavailable_never_cpu(
        self, fake: FakeTrt, staged_engine: Any, environment
    ) -> None:
        fake.cudart.device_count = 0
        result = run_backend(cfg(batch=1), environment)
        assert result.status is BenchmarkStatus.UNAVAILABLE
        assert result.errors[0].phase == "validate"
        assert "Not falling back to CPU" in result.errors[0].message
        assert result.raw_samples.latency_ms == []

    def test_device_index_beyond_the_visible_devices_is_unavailable(
        self, fake: FakeTrt, staged_engine: Any, environment
    ) -> None:
        result = run_backend(cfg("cuda:3", batch=1), environment)
        assert result.status is BenchmarkStatus.UNAVAILABLE
        assert "only 1 CUDA device" in result.errors[0].message

    def test_a_missing_device_copy_is_caught_by_the_sanity_pass(
        self, fake: FakeTrt, staged_engine: Any, environment, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: the output buffer is NaN-filled, so a copy that never happens fails.

        With np.empty it passed or failed depending on what the uninitialised memory
        happened to contain -- it passed on one machine and failed on another.
        """
        monkeypatch.setattr(fake.cudart, "cudaMemcpy", lambda *a, **k: (0,))
        result = run_backend(cfg(batch=1), environment)
        assert result.status is BenchmarkStatus.FAILED
        assert result.errors[0].phase == "prepare"
        assert "non-finite" in result.errors[0].message


class TestEnginePathsAreDistinct:
    """Regression: engines that differ must not share a filename."""

    def test_the_version_dots_do_not_truncate_the_filename(self, tmp_path: Path) -> None:
        """Path.with_suffix() ate everything after the last dot of the TensorRT version.

        "...-trt11.3.0.99-sm89-b1_8_8-ieee_fp32" collapsed to "...-trt11.3.0.plan",
        dropping the architecture, profile and precision from the name, so the FP32 and
        TF32 engines overwrote each other.
        """
        config = TrtBuildConfig(model="resnet50", weights="IMAGENET1K_V2")
        plan, manifest = tensorrt_build.engine_paths(
            config, "resnet50-canonical", "11.3.0.99", "8.9", tmp_path
        )
        assert plan.name == "resnet50-canonical-trt11.3.0.99-sm89-b1_8_8-ieee_fp32.plan"
        assert manifest.name == (
            "resnet50-canonical-trt11.3.0.99-sm89-b1_8_8-ieee_fp32.manifest.json"
        )

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ({"precision": "ieee_fp32"}, {"precision": "tf32"}),
            ({"opt_batch": 1}, {"opt_batch": 8}),
            ({"max_batch": 8}, {"max_batch": 16}),
        ],
    )
    def test_engines_that_differ_get_different_files(
        self, first: dict[str, Any], second: dict[str, Any], tmp_path: Path
    ) -> None:
        base = {"model": "resnet50", "weights": "IMAGENET1K_V2", "min_batch": 1, "max_batch": 8}
        a, _ = tensorrt_build.engine_paths(
            TrtBuildConfig(**{**base, **first}), "stem", "11.3.0.99", "8.9", tmp_path
        )
        b, _ = tensorrt_build.engine_paths(
            TrtBuildConfig(**{**base, **second}), "stem", "11.3.0.99", "8.9", tmp_path
        )
        assert a != b, "different engines would overwrite each other"

    def test_different_architectures_get_different_files(self, tmp_path: Path) -> None:
        config = TrtBuildConfig(model="resnet50", weights="IMAGENET1K_V2")
        ada, _ = tensorrt_build.engine_paths(config, "stem", "11.3.0.99", "8.9", tmp_path)
        hopper, _ = tensorrt_build.engine_paths(config, "stem", "11.3.0.99", "9.0", tmp_path)
        assert ada != hopper


class TestPhaseAttribution:
    """The engine build belongs to the build phase, not to model load."""

    def test_load_fetches_the_artifact_and_build_makes_the_engine(
        self,
        fake: FakeTrt,
        staged_engine: Any,
        environment,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gpu_benchlab.backends import tensorrt_backend
        from gpu_benchlab.backends.tensorrt_backend import TensorRtBackend

        stub_dir = tmp_path / "phase-attribution"
        stub_dir.mkdir(parents=True, exist_ok=True)
        artifact, onnx = onnx_manifest_stub(stub_dir)
        calls: list[str] = []
        staged = tensorrt_backend.ensure_engine

        def record_artifact(*args: Any, **kwargs: Any) -> Any:
            calls.append("ensure_artifact")
            return artifact, onnx, False

        def record_engine(*args: Any, **kwargs: Any) -> Any:
            calls.append("ensure_engine")
            return staged(*args, **kwargs)

        monkeypatch.setattr(tensorrt_backend, "ensure_artifact", record_artifact)
        monkeypatch.setattr(tensorrt_backend, "ensure_engine", record_engine)

        config = cfg(batch=1)
        backend = TensorRtBackend(config)
        try:
            backend.validate(config, environment)
            backend.load()
            assert calls == ["ensure_artifact"], "load() must not build or fetch an engine"
            assert "engine_sha256" not in backend.descriptor.settings
            backend.build()
            assert calls == ["ensure_artifact", "ensure_engine"]
            assert backend.descriptor.settings["engine_sha256"] == staged_engine.engine_sha256
            assert backend.descriptor.settings["engine_deserialization_ms"] >= 0
        finally:
            backend.close()
