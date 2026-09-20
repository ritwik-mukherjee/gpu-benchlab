"""Tests for ONNX export, the PyTorch-vs-ORT correctness check and the ORT backend.

Three kinds of test, labelled so none can be mistaken for another:

* REAL: real torch export, real onnxruntime (CPU EP) inference.
* REAL-FALLBACK: real onnxruntime session creation with only the availability
  pre-check patched, reproducing ORT's genuine silent CUDA->CPU substitution.
* STRUCTURAL: a fake onnxruntime module, for CUDA-only structure (IOBinding, TF32
  provider options, partial placement). These prove wiring, not GPU behaviour.

A tiny registered CNN keeps exports fast; real ResNet-50 tests run only when its
pinned weights are already cached locally.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")
pytest.importorskip("onnxscript")

from gpu_benchlab.backends import ort_backend  # noqa: E402
from gpu_benchlab.backends.ort_backend import CPU_EP, CUDA_EP, OnnxRuntimeBackend  # noqa: E402
from gpu_benchlab.core.backend import DeviceKind  # noqa: E402
from gpu_benchlab.core.config import ExperimentConfig, parse_config  # noqa: E402
from gpu_benchlab.core.engine import CPU_RESULT_NOTE, BenchmarkEngine  # noqa: E402
from gpu_benchlab.core.errors import (  # noqa: E402
    BackendError,
    ConfigurationError,
    UnavailableError,
)
from gpu_benchlab.core.inputs import INPUT_GENERATOR  # noqa: E402
from gpu_benchlab.core.schema import BenchmarkStatus  # noqa: E402
from gpu_benchlab.core.timing import TimingMechanism  # noqa: E402
from gpu_benchlab.export.onnx_export import (  # noqa: E402
    ExportConfig,
    artifact_paths,
    default_onnx_dir,
    ensure_artifact,
    export_onnx,
    load_manifest,
    validate_artifact,
)
from gpu_benchlab.hardware.detect import detect_environment  # noqa: E402
from gpu_benchlab.models.registry import (  # noqa: E402
    RANDOM_WEIGHTS,
    ModelSpec,
    register_model,
    unregister_model,
)
from gpu_benchlab.models.weights import default_cache_dir  # noqa: E402

pytestmark = [pytest.mark.torch, pytest.mark.onnx]

TINY = "tiny-onnx"
TINY_PARAMS = (3 * 4 * 3 * 3 + 4) + (4 * 10 + 10)  # 162


class TinyOnnxNet(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(4, 10)

    def forward(self, x: Any) -> Any:
        return self.fc(torch.flatten(self.pool(torch.relu(self.conv(x))), 1))


@pytest.fixture(scope="module", autouse=True)
def tiny_model() -> Iterator[None]:
    register_model(
        ModelSpec(
            name=TINY,
            family="vision",
            architecture="TinyOnnxNet",
            build=TinyOnnxNet,
            input_shape=(3, 16, 16),
            output_shape=(10,),
            expected_parameters=TINY_PARAMS,
            default_weights=RANDOM_WEIGHTS,
            test_only=True,
        )
    )
    yield
    unregister_model(TINY)


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """An isolated cache holding one tiny export, shared by the module."""
    root = tmp_path_factory.mktemp("cache")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GPU_BENCHLAB_CACHE", str(root))
        export_onnx(ExportConfig(model=TINY))
        yield root


@pytest.fixture(scope="module")
def environment():
    return detect_environment(include_frameworks=False)


@pytest.fixture(scope="module")
def cuda_ep_active(cache: Path) -> bool:
    """Whether a session here really runs on the CUDA EP.

    Being listed proves nothing (Phase 4: ORT lists the EP and then silently builds a
    CPU session), so this creates a real session and asks it. On a GPU machine this is
    True, which makes the REAL-FALLBACK substitution test inapplicable: there is
    nothing to substitute.
    """
    if CUDA_EP not in ort.get_available_providers():
        return False
    artifact, _ = artifact_paths(ExportConfig(model=TINY), directory=cache / "onnx")
    try:
        session = ort.InferenceSession(str(artifact), providers=[CUDA_EP])
    except Exception:  # noqa: BLE001 - any failure means the EP is not usable here
        return False
    return CUDA_EP in session.get_providers()


@pytest.fixture
def engine(environment) -> BenchmarkEngine:
    return BenchmarkEngine(environment=environment)


def cfg(
    device: str | None = "cpu",
    *,
    precision: str = "fp32",
    batch: int = 2,
    options: dict[str, Any] | None = None,
    model: str = TINY,
) -> ExperimentConfig:
    data: dict[str, Any] = {
        "name": "ort-test",
        "backend": "onnxruntime",
        "precision": precision,
        "batch_size": batch,
        "model": {"name": model},
        "benchmark": {"warmup_iterations": 2, "measurement_iterations": 4},
        "backend_options": {"allow_export": False, **(options or {})},
    }
    if device is not None:
        data["device"] = device
    return parse_config(data)


def run(engine: BenchmarkEngine, config: ExperimentConfig):
    return engine.run(OnnxRuntimeBackend(config), config)


# ================================================================================ export


class TestExport:
    """REAL: torch.onnx.export (dynamo) of the tiny model."""

    def test_manifest_records_provenance(self, cache: Path) -> None:
        path, manifest_path = artifact_paths(ExportConfig(model=TINY), cache / "onnx")
        m = load_manifest(manifest_path)
        assert m.model.name == TINY and m.model.weights == RANDOM_WEIGHTS
        assert m.model.parameter_count == TINY_PARAMS
        assert m.exporter.opset == 20
        assert m.exporter.external_data is False
        assert m.exporter.exporter == "torch.onnx.export(dynamo=True)"
        assert m.exporter.torch_version == torch.__version__
        assert m.exporter.onnx_version and m.exporter.onnxscript_version
        assert m.exporter.trace_input_generator == INPUT_GENERATOR
        assert m.input.shape == ["batch", 3, 16, 16] and m.input.dtype == "float32"
        assert m.output.shape == ["batch", 10]
        assert m.artifact.filename == path.name
        assert m.provenance.benchlab_version

    def test_validation_checks(self, cache: Path) -> None:
        path, manifest_path = artifact_paths(ExportConfig(model=TINY), cache / "onnx")
        checks = validate_artifact(path, load_manifest(manifest_path))
        joined = " | ".join(checks)
        for expected in ("sha256", "no external data", "full_check", "opset 20", "input", "output"):
            assert expected in joined

    def test_no_external_data_files(self, cache: Path) -> None:
        assert not list((cache / "onnx").glob("*.data"))

    def test_reexport_is_byte_identical(self, cache: Path, tmp_path: Path) -> None:
        first = load_manifest(artifact_paths(ExportConfig(model=TINY), cache / "onnx")[1])
        _, again = export_onnx(ExportConfig(model=TINY), directory=tmp_path)
        assert again.artifact.sha256 == first.artifact.sha256

    def test_static_batch_export(self, tmp_path: Path) -> None:
        path, m = export_onnx(
            ExportConfig(model=TINY, dynamic_batch=False, trace_batch_size=2), directory=tmp_path
        )
        assert path.name == f"{TINY}-random-opset20-b2.onnx"
        assert m.input.shape == [2, 3, 16, 16] and m.exporter.dynamic_shapes == {}

    def test_tampered_artifact_is_refused(self, cache: Path, tmp_path: Path) -> None:
        src, manifest_path = artifact_paths(ExportConfig(model=TINY), cache / "onnx")
        bad = tmp_path / src.name
        data = bytearray(src.read_bytes())
        data[-1] ^= 0xFF
        bad.write_bytes(bytes(data))
        with pytest.raises(BackendError, match="does not match its manifest"):
            validate_artifact(bad, load_manifest(manifest_path))

    def test_opset_mismatch_is_refused(self, cache: Path) -> None:
        path, manifest_path = artifact_paths(ExportConfig(model=TINY), cache / "onnx")
        m = load_manifest(manifest_path)
        lying = m.model_copy(update={"exporter": m.exporter.model_copy(update={"opset": 18})})
        with pytest.raises(BackendError, match="opset 20 != manifest 18"):
            validate_artifact(path, lying)

    def test_missing_artifact_is_unavailable(self, cache: Path, tmp_path: Path) -> None:
        m = load_manifest(artifact_paths(ExportConfig(model=TINY), cache / "onnx")[1])
        with pytest.raises(UnavailableError, match="not found"):
            validate_artifact(tmp_path / "nope.onnx", m)

    def test_ensure_artifact_reuses_and_respects_allow_export(
        self, cache: Path, tmp_path: Path
    ) -> None:
        _, _, exported = ensure_artifact(ExportConfig(model=TINY), directory=cache / "onnx")
        assert exported is False
        with pytest.raises(UnavailableError, match="export is disabled"):
            ensure_artifact(ExportConfig(model=TINY), directory=tmp_path, allow_export=False)

    def test_cache_location_follows_env(self, cache: Path) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("GPU_BENCHLAB_CACHE", str(cache))
            assert default_onnx_dir() == cache / "onnx"


# =========================================================================== correctness


class TestCorrectnessTiny:
    """REAL: PyTorch vs ORT on the tiny model."""

    def test_report_and_stored_outputs(self, cache: Path, tmp_path: Path) -> None:
        from gpu_benchlab.export.verify import save_report, verify_against_pytorch

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("GPU_BENCHLAB_CACHE", str(cache))
            report, outputs = verify_against_pytorch(
                ExportConfig(model=TINY), batch_sizes=(1, 4), seeds=(0, 1), allow_export=False
            )
        assert report.cases_total == 4
        assert report.cases_passed == 4, [c.comparison.reasons for c in report.cases]
        assert report.passed == (report.negative_control_rejected and report.cases_passed == 4)
        assert report.candidate.detail["active_providers"] == CPU_EP
        assert report.artifact_sha256 == report.manifest.artifact.sha256

        directory = save_report(report, outputs, tmp_path)
        stored = np.load(directory / "outputs.npz")
        assert {"b1_s0_reference", "b4_s1_candidate", "negative_control_fp16"} <= set(stored.files)
        # Independent re-derivation of one case from the stored arrays.
        ref, cand = stored["b4_s1_reference"], stored["b4_s1_candidate"]
        case = next(c for c in report.cases if (c.batch_size, c.seed) == (4, 1))
        assert float(np.abs(cand.astype(np.float64) - ref).max()) == pytest.approx(
            case.comparison.max_abs_error
        )
        assert (
            json.loads((directory / "report.json").read_text(encoding="utf-8"))["passed"]
            == report.passed
        )

    def test_refuses_to_compare_different_weights(
        self, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gpu_benchlab.export import verify
        from gpu_benchlab.models import torch_loader

        real = torch_loader.load_reference_model

        def other_weights(*a: Any, **k: Any) -> Any:
            loaded = real(*a, **k)
            info = loaded.info.model_copy(update={"weights_sha256": "f" * 64})
            return torch_loader.LoadedModel(module=loaded.module, info=info)

        monkeypatch.setenv("GPU_BENCHLAB_CACHE", str(cache))
        monkeypatch.setattr(torch_loader, "load_reference_model", other_weights)
        with pytest.raises(BackendError, match="Refusing to compare"):
            verify.verify_against_pytorch(ExportConfig(model=TINY), allow_export=False)


def _resnet_cached() -> bool:
    return (default_cache_dir() / "resnet50-11ad3fa6.pth").is_file()


@pytest.mark.slow
@pytest.mark.skipif(not _resnet_cached(), reason="pinned ResNet-50 weights not cached locally")
class TestCorrectnessResNet50:
    """REAL: the Phase 4 headline, on the pinned weights."""

    def test_resnet50_matches_pytorch_and_rejects_fp16(self) -> None:
        from gpu_benchlab.export.verify import verify_against_pytorch

        report, _ = verify_against_pytorch(ExportConfig(model="resnet50"))
        assert report.cases_passed == report.cases_total == 9
        assert report.negative_control_rejected, "tolerance must reject FP16 execution"
        assert report.passed
        assert all(c.comparison.top1_agreement == 1.0 for c in report.cases)


# =============================================================================== backend


class TestBackendConstruction:
    def test_device_required(self) -> None:
        with pytest.raises(ConfigurationError, match="explicit device"):
            OnnxRuntimeBackend(cfg(device=None))

    @pytest.mark.parametrize("device", ["gpu", "cuda:x", "tpu"])
    def test_bad_device(self, device: str) -> None:
        with pytest.raises(ConfigurationError, match="Unrecognised device"):
            OnnxRuntimeBackend(cfg(device=device))

    def test_unknown_option(self) -> None:
        with pytest.raises(ConfigurationError, match="backend_options"):
            OnnxRuntimeBackend(cfg(options={"use_tf33": True}))

    def test_cuda_maps_to_cuda_ep(self) -> None:
        b = OnnxRuntimeBackend(cfg(device="cuda:1"))
        assert b.descriptor.execution_provider == CUDA_EP and b.descriptor.device == "cuda:1"


class TestBackendCpu:
    """REAL: ORT CPU EP inference through the unchanged engine."""

    def test_run(
        self, engine: BenchmarkEngine, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GPU_BENCHLAB_CACHE", str(cache))
        r = run(engine, cfg())
        assert r.status is BenchmarkStatus.OK, r.errors
        assert r.device_kind is DeviceKind.CPU and not r.is_simulated
        assert r.timing_mechanism is TimingMechanism.WALL_CLOCK
        assert r.backend.execution_provider == CPU_EP
        assert len(r.raw_samples.latency_ms) == 4 and len(r.raw_samples.warmup_latency_ms) == 2
        assert CPU_RESULT_NOTE in r.notes
        assert "not suitable for backend performance ranking" in CPU_RESULT_NOTE

    def test_settings_and_provenance(
        self, engine: BenchmarkEngine, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GPU_BENCHLAB_CACHE", str(cache))
        r = run(engine, cfg())
        s = r.backend.settings
        manifest = load_manifest(artifact_paths(ExportConfig(model=TINY), cache / "onnx")[1])
        assert s["requested_provider"] == CPU_EP and s["active_providers"] == CPU_EP
        assert s["node_placement.CPUExecutionProvider"] > 0
        assert not any(k.startswith("node_placement.CUDA") for k in s)
        assert s["artifact_sha256"] == manifest.artifact.sha256
        assert s["opset"] == 20 and s["dynamic_batch"] is True
        assert s["input_shape"] == "2x3x16x16" and s["input_dtype"] == "float32"
        assert s["input_generator"] == INPUT_GENERATOR
        assert s["io_binding"] is False and s["sanity_check"] == "passed"
        assert s["onnxruntime_version"] == ort.__version__
        assert r.model_info is not None and r.model_info.parameter_count == TINY_PARAMS

    def test_phases_separate_load_build_prepare(
        self, engine: BenchmarkEngine, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GPU_BENCHLAB_CACHE", str(cache))
        p = run(engine, cfg()).phases
        assert p.model_load_ms and p.engine_build_ms and p.prepare_inputs_ms

    @pytest.mark.parametrize("precision", ["fp16", "bf16", "int8", "fp8"])
    def test_non_fp32_precisions_unsupported(
        self, engine: BenchmarkEngine, cache: Path, monkeypatch: pytest.MonkeyPatch, precision: str
    ) -> None:
        monkeypatch.setenv("GPU_BENCHLAB_CACHE", str(cache))
        r = run(engine, cfg(precision=precision))
        assert r.status is BenchmarkStatus.UNSUPPORTED and r.errors[0].phase == "validate"

    def test_tf32_on_cpu_unsupported(
        self, engine: BenchmarkEngine, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GPU_BENCHLAB_CACHE", str(cache))
        assert run(engine, cfg(precision="tf32")).status is BenchmarkStatus.UNSUPPORTED

    def test_missing_artifact_without_export_is_unavailable(
        self, engine: BenchmarkEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GPU_BENCHLAB_CACHE", str(tmp_path))
        r = run(engine, cfg())
        assert r.status is BenchmarkStatus.UNAVAILABLE and r.errors[0].phase == "validate"

    def test_artifact_changed_after_validation_fails_load(
        self, environment, cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        own = tmp_path / "own"
        monkeypatch.setenv("GPU_BENCHLAB_CACHE", str(own))
        export_onnx(ExportConfig(model=TINY))
        backend = OnnxRuntimeBackend(cfg())
        backend.validate(cfg(), environment)
        path = artifact_paths(ExportConfig(model=TINY))[0]
        path.write_bytes(path.read_bytes() + b"\x00")
        with pytest.raises(BackendError, match="changed on disk"):
            backend.load()

    def test_ort_missing_is_unavailable(self, environment, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        with pytest.raises(UnavailableError, match="onnxruntime is not installed"):
            OnnxRuntimeBackend(cfg()).validate(cfg(), environment)


class TestNoSilentCudaFallback:
    def test_cpu_only_package_rejects_cuda_before_anything_runs(
        self, engine: BenchmarkEngine, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REAL: the installed CPU-only package does not list the CUDA EP."""
        if CUDA_EP in ort.get_available_providers():
            pytest.skip("a CUDA-capable onnxruntime package is installed")
        monkeypatch.setenv("GPU_BENCHLAB_CACHE", str(cache))
        r = run(engine, cfg(device="cuda:0"))
        assert r.status is BenchmarkStatus.UNAVAILABLE
        assert r.errors[0].phase == "validate"
        assert "Not falling back to CPU" in r.errors[0].message
        assert r.raw_samples.latency_ms == [] and r.timing_mechanism is None

    def test_real_ort_silent_substitution_is_caught_at_build(
        self,
        engine: BenchmarkEngine,
        cache: Path,
        cuda_ep_active: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REAL-FALLBACK: only the availability list is patched.

        The session is then created by real onnxruntime, which -- asked for the CUDA EP
        it cannot provide -- silently builds a CPU session (observed with both the
        CPU package and onnxruntime-gpu 1.30 without CUDA libraries). The backend must
        refuse at build, not benchmark the CPU.
        """
        if cuda_ep_active:
            pytest.skip("the CUDA EP really loads here, so ORT substitutes nothing")
        monkeypatch.setenv("GPU_BENCHLAB_CACHE", str(cache))
        real = ort.get_available_providers()
        monkeypatch.setattr(ort, "get_available_providers", lambda: [CUDA_EP, *real])
        with pytest.warns(UserWarning, match="not in available provider"):
            r = run(engine, cfg(device="cuda:0"))
        assert r.status is BenchmarkStatus.UNAVAILABLE
        assert r.errors[0].phase == "build"
        assert "substituted" in r.errors[0].message
        assert r.raw_samples.latency_ms == [], "nothing may be measured on the CPU"
        assert r.timing_mechanism is None


# ======================================================================== CUDA structural


class FakeSession:
    def __init__(self, fake: FakeOrt, providers: list[Any]) -> None:
        self.fake = fake
        self.provider, self.options = providers[0]
        self.profiled = fake.last_so.enable_profiling
        self.prefix = fake.last_so.profile_file_prefix

    def get_providers(self) -> list[str]:
        return [self.provider, CPU_EP] if self.fake.ep_loads else [CPU_EP]

    def get_provider_options(self) -> dict[str, dict[str, str]]:
        return {self.provider: dict(self.options)}

    def run(self, names: Any, feed: dict[str, Any]) -> list[Any]:
        batch = next(iter(feed.values())).shape[0]
        return [np.zeros((batch, 10), dtype=np.float32)]

    def io_binding(self) -> Any:
        fake = self.fake

        class Binding:
            def bind_ortvalue_input(self, name: str, value: Any) -> None:
                fake.bound_input = (name, value.device, value.array.shape)

            def bind_output(self, name: str, device: str, device_id: int) -> None:
                fake.bound_output = (name, device, device_id)

            def copy_outputs_to_cpu(self) -> list[Any]:
                return [np.zeros((fake.bound_input[2][0], 10), dtype=np.float32)]

        return Binding()

    def run_with_iobinding(self, binding: Any) -> None:
        self.fake.iobinding_runs += 1

    def end_profiling(self) -> str:
        events = [
            {"cat": "Node", "name": f"n{i}_kernel_time", "args": {"provider": ep}}
            for i, ep in enumerate(self.fake.placement)
        ]
        path = Path(self.prefix + ".json")
        path.write_text(json.dumps(events), encoding="utf-8")
        return str(path)


class FakeOrt:
    """Stands in for a CUDA-capable onnxruntime. STRUCTURAL tests only."""

    __version__ = "fake-1.30"

    def __init__(self, *, ep_loads: bool = True, placement: list[str] | None = None) -> None:
        self.ep_loads = ep_loads
        self.placement = placement or [CUDA_EP] * 6
        self.sessions: list[FakeSession] = []
        self.bound_input: Any = None
        self.bound_output: Any = None
        self.iobinding_runs = 0
        self.GraphOptimizationLevel = SimpleNamespace(
            ORT_DISABLE_ALL=0, ORT_ENABLE_BASIC=1, ORT_ENABLE_EXTENDED=2, ORT_ENABLE_ALL=99
        )
        fake = self

        class SessionOptions:
            def __init__(self) -> None:
                self.enable_profiling = False
                self.profile_file_prefix = ""
                fake.last_so = self

        class OrtValue:
            @staticmethod
            def ortvalue_from_numpy(array: Any, device: str, device_id: int) -> Any:
                return SimpleNamespace(array=array, device=(device, device_id))

        self.SessionOptions = SessionOptions
        self.OrtValue = OrtValue

    def get_available_providers(self) -> list[str]:
        return [CUDA_EP, CPU_EP]

    def InferenceSession(self, model: Any, so: Any, providers: list[Any]) -> FakeSession:  # noqa: N802
        s = FakeSession(self, providers)
        self.sessions.append(s)
        return s


def gpu_env(environment, cc: tuple[int, int] | None):
    """The real environment with a synthetic GPU entry, for capability checks only."""
    if cc is None:
        return environment.model_copy(update={"gpus": []})
    from gpu_benchlab.hardware.types import GPUInfo

    gpu = GPUInfo(
        index=0,
        name="Fake GPU",
        compute_capability=f"{cc[0]}.{cc[1]}",
        compute_capability_major=cc[0],
        compute_capability_minor=cc[1],
    )
    return environment.model_copy(update={"gpus": [gpu]})


class TestCudaStructural:
    """STRUCTURAL: wiring of the CUDA path against a fake ORT. Not GPU evidence."""

    @pytest.fixture
    def fake(self, monkeypatch: pytest.MonkeyPatch, cache: Path) -> FakeOrt:
        monkeypatch.setenv("GPU_BENCHLAB_CACHE", str(cache))
        f = FakeOrt()
        monkeypatch.setattr(ort_backend, "_import_ort", lambda: f)
        return f

    def test_fp32_disables_tf32_explicitly(self, fake: FakeOrt, environment) -> None:
        r = BenchmarkEngine(environment=environment).run(
            OnnxRuntimeBackend(cfg("cuda:0")), cfg("cuda:0")
        )
        assert r.status is BenchmarkStatus.OK, r.errors
        bench = fake.sessions[0]
        assert bench.options == {
            "device_id": "0",
            "use_tf32": "0",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
        }
        assert r.backend.settings["active_provider_option.use_tf32"] == "0"

    def test_io_binding_keeps_data_on_device(self, fake: FakeOrt, environment) -> None:
        BenchmarkEngine(environment=environment).run(
            OnnxRuntimeBackend(cfg("cuda:0")), cfg("cuda:0")
        )
        assert fake.bound_input[1] == ("cuda", 0)
        assert fake.bound_output == ("logits", "cuda", 0)
        assert fake.iobinding_runs == 1 + 2 + 4  # sanity + warmup + measured

    def test_device_index_reaches_provider(self, fake: FakeOrt, environment) -> None:
        BenchmarkEngine(environment=environment).run(
            OnnxRuntimeBackend(cfg("cuda:3")), cfg("cuda:3")
        )
        assert fake.sessions[0].options["device_id"] == "3"

    def test_ep_that_fails_to_load_is_unavailable(self, fake: FakeOrt, environment) -> None:
        fake.ep_loads = False
        r = BenchmarkEngine(environment=environment).run(
            OnnxRuntimeBackend(cfg("cuda:0")), cfg("cuda:0")
        )
        assert r.status is BenchmarkStatus.UNAVAILABLE and r.errors[0].phase == "build"

    def test_partial_placement_is_unsupported_by_default(self, fake: FakeOrt, environment) -> None:
        fake.placement = [CUDA_EP] * 5 + [CPU_EP] * 2
        r = BenchmarkEngine(environment=environment).run(
            OnnxRuntimeBackend(cfg("cuda:0")), cfg("cuda:0")
        )
        assert r.status is BenchmarkStatus.UNSUPPORTED
        assert r.errors[0].phase == "prepare"
        assert "2 node(s)" in r.errors[0].message
        assert r.backend.settings["node_placement.CPUExecutionProvider"] == 2

    def test_partial_placement_allowed_is_recorded(self, fake: FakeOrt, environment) -> None:
        fake.placement = [CUDA_EP] * 5 + [CPU_EP] * 2
        c = cfg("cuda:0", options={"allow_partial_placement": True})
        r = BenchmarkEngine(environment=environment).run(OnnxRuntimeBackend(c), c)
        assert r.status is BenchmarkStatus.OK
        s = r.backend.settings
        assert s["node_placement.CUDAExecutionProvider"] == 5
        assert s["node_placement.CPUExecutionProvider"] == 2

    @pytest.mark.parametrize(("cc", "ok"), [((8, 6), True), ((7, 5), False), (None, False)])
    def test_tf32_requires_known_sm80(self, fake: FakeOrt, environment, cc, ok: bool) -> None:
        c = cfg("cuda:0", precision="tf32")
        r = BenchmarkEngine(environment=gpu_env(environment, cc)).run(OnnxRuntimeBackend(c), c)
        if ok:
            assert r.status is BenchmarkStatus.OK, r.errors
            assert fake.sessions[0].options["use_tf32"] == "1"
        else:
            assert r.status is BenchmarkStatus.UNSUPPORTED and r.errors[0].phase == "validate"
