"""Tests for the Phase 3 changes to the engine / backend contract.

Uses small purpose-built backends rather than torch, so these run everywhere.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gpu_benchlab.backends.fake import FakeBackend
from gpu_benchlab.cli.run_cmd import build_backend
from gpu_benchlab.core.backend import Backend, BackendDescriptor, DeviceKind, ModelInfo
from gpu_benchlab.core.config import BenchmarkConfig, ExperimentConfig, ModelConfig, parse_config
from gpu_benchlab.core.engine import CPU_RESULT_NOTE, SIMULATION_WARNING, BenchmarkEngine
from gpu_benchlab.core.errors import (
    BackendError,
    ConfigurationError,
    OutOfMemoryError,
    UnavailableError,
)
from gpu_benchlab.core.schema import RESULT_SCHEMA_VERSION, BenchmarkResult, BenchmarkStatus
from gpu_benchlab.core.storage import ResultStore
from gpu_benchlab.core.timing import ScriptedTimer, Timer, TimingMechanism
from gpu_benchlab.hardware.detect import detect_environment

FIXTURE_1_0 = Path(__file__).parent.parent / "fixtures" / "result_schema_1_0.json"


@pytest.fixture(scope="module")
def environment():
    return detect_environment(include_frameworks=False)


@pytest.fixture
def engine(environment) -> BenchmarkEngine:
    return BenchmarkEngine(environment=environment)


def cfg(warmup: int = 2, iterations: int = 5) -> ExperimentConfig:
    return ExperimentConfig(
        name="contract",
        model=ModelConfig(name="m"),
        backend="probe",
        benchmark=BenchmarkConfig(warmup_iterations=warmup, measurement_iterations=iterations),
    )


class ProbeBackend(Backend):
    """A configurable real-but-trivial backend for exercising the contract.

    `fail_at` raises in the named lifecycle step. It uses a scripted timer, so its
    results are test fixtures for the engine contract, never measurements.
    """

    def __init__(
        self,
        *,
        fail_at: str | None = None,
        exc: Exception | None = None,
        device_kind: DeviceKind = DeviceKind.CPU,
        timer: Timer | None = None,
        fail_on_execute_call: int | None = None,
    ) -> None:
        self.fail_at = fail_at
        self.exc = exc or BackendError(f"boom in {fail_at}")
        self.kind = device_kind
        self.timer = timer
        self.fail_on_execute_call = fail_on_execute_call
        self.events: list[str] = []
        self.seen_environment: Any = None
        self.settings: dict[str, Any] = {"stage": "constructed"}
        self.calls = 0
        self.in_context = False
        self.context_during_execute: list[bool] = []

    def _maybe_fail(self, step: str) -> None:
        self.events.append(step)
        if self.fail_at == step:
            raise self.exc

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            name="probe", device=self.kind.value, device_kind=self.kind, settings=self.settings
        )

    @property
    def model_info(self) -> ModelInfo | None:
        return ModelInfo(
            name="m",
            architecture="Probe",
            weights="random",
            parameter_count=1,
            expected_parameter_count=1,
        )

    def validate(self, config: Any, environment: Any) -> None:
        self.seen_environment = environment
        self._maybe_fail("validate")

    def load(self) -> None:
        self._maybe_fail("load")
        self.settings = {"stage": "applied-in-load"}

    def build(self) -> None:
        self._maybe_fail("build")

    def prepare(self, config: Any) -> Any:
        self._maybe_fail("prepare")
        return object()

    @contextlib.contextmanager
    def _ctx(self) -> Iterator[None]:
        self.events.append("context-enter")
        self.in_context = True
        try:
            yield
        finally:
            self.in_context = False
            self.events.append("context-exit")

    def execution_context(self) -> contextlib.AbstractContextManager[object]:
        return self._ctx()

    def execute(self, inputs: Any) -> Any:
        self.calls += 1
        self.context_during_execute.append(self.in_context)
        if self.fail_on_execute_call is not None and self.calls >= self.fail_on_execute_call:
            raise self.exc
        return inputs

    def make_timer(self) -> Timer:
        return self.timer or ScriptedTimer([1.0] * 1000)

    def close(self) -> None:
        self.events.append("close")
        if self.fail_at == "close":
            raise self.exc


class SecondaryTimer(ScriptedTimer):
    """Scripted primary samples plus a scripted secondary series."""

    def __init__(self, primary: list[float], secondary: list[float]) -> None:
        super().__init__(primary)
        self._secondary_source = list(secondary)
        self._collected: list[float] = []

    def stop(self) -> float:
        value = super().stop()
        self._collected.append(self._secondary_source.pop(0))
        return value

    def drain_secondary(self) -> tuple[TimingMechanism, list[float]] | None:
        out, self._collected = self._collected, []
        return TimingMechanism.WALL_CLOCK_SYNCHRONIZED, out


class TestUnavailableStatus:
    def test_unavailable_error_maps_to_unavailable(self, engine: BenchmarkEngine) -> None:
        backend = ProbeBackend(fail_at="validate", exc=UnavailableError("no cuda here"))
        result = engine.run(backend, cfg())
        assert result.status is BenchmarkStatus.UNAVAILABLE
        assert result.errors[0].type == "UnavailableError"
        assert result.errors[0].phase == "validate"

    def test_unavailable_run_claims_no_timing_mechanism(self, engine: BenchmarkEngine) -> None:
        """Nothing was timed, so no mechanism may be claimed."""
        backend = ProbeBackend(fail_at="validate", exc=UnavailableError("absent"))
        result = engine.run(backend, cfg())
        assert result.timing_mechanism is None
        assert result.latency is None
        assert result.raw_samples.latency_ms == []

    def test_unavailable_run_does_not_load_or_execute(self, engine: BenchmarkEngine) -> None:
        backend = ProbeBackend(fail_at="validate", exc=UnavailableError("absent"))
        engine.run(backend, cfg())
        assert "load" not in backend.events
        assert backend.calls == 0
        assert backend.events[-1] == "close", "close() still runs"


class TestPhaseTagging:
    @pytest.mark.parametrize("step", ["validate", "load", "build", "prepare"])
    def test_setup_failures_are_tagged_with_their_phase(
        self, engine: BenchmarkEngine, step: str
    ) -> None:
        result = engine.run(ProbeBackend(fail_at=step), cfg())
        assert result.status is BenchmarkStatus.FAILED
        assert result.errors[0].phase == step

    def test_failure_during_warmup(self, engine: BenchmarkEngine) -> None:
        result = engine.run(ProbeBackend(fail_on_execute_call=1), cfg(warmup=2))
        assert result.errors[0].phase == "warmup"

    def test_failure_during_measurement(self, engine: BenchmarkEngine) -> None:
        result = engine.run(ProbeBackend(fail_on_execute_call=4), cfg(warmup=2, iterations=5))
        assert result.errors[0].phase == "measure"

    def test_oom_raised_in_prepare_is_tagged_prepare(self, engine: BenchmarkEngine) -> None:
        """Phase 2 bug: this used to be tagged 'execute' regardless of origin."""
        result = engine.run(FakeBackend(seed=1, oom_above_batch_size=0), cfg())
        assert result.status is BenchmarkStatus.FAILED
        assert result.errors[0].type == "OutOfMemoryError"
        assert result.errors[0].phase == "prepare"

    def test_oom_during_measurement_is_failed_and_tagged(self, engine: BenchmarkEngine) -> None:
        backend = ProbeBackend(fail_on_execute_call=3, exc=OutOfMemoryError("CUDA out of memory"))
        result = engine.run(backend, cfg(warmup=0, iterations=5))
        assert result.status is BenchmarkStatus.FAILED
        assert result.errors[0].type == "OutOfMemoryError"
        assert result.errors[0].phase == "measure"

    def test_invalid_samples_are_tagged_statistics(self, engine: BenchmarkEngine) -> None:
        backend = ProbeBackend(timer=ScriptedTimer([1.0, float("nan"), 1.0]))
        result = engine.run(backend, cfg(warmup=0, iterations=3))
        assert result.errors[0].phase == "statistics"

    def test_close_failure_is_tagged_close(self, engine: BenchmarkEngine) -> None:
        result = engine.run(ProbeBackend(fail_at="close"), cfg())
        assert any(e.phase == "close" for e in result.errors)


class TestEnvironmentPassedToValidate:
    def test_validate_receives_the_engine_environment(
        self, engine: BenchmarkEngine, environment
    ) -> None:
        backend = ProbeBackend()
        engine.run(backend, cfg())
        assert backend.seen_environment is environment


class TestExecutionContext:
    def test_entered_once_around_warmup_and_measurement(self, engine: BenchmarkEngine) -> None:
        backend = ProbeBackend()
        engine.run(backend, cfg(warmup=3, iterations=4))
        assert backend.events.count("context-enter") == 1
        assert backend.events.count("context-exit") == 1
        assert all(backend.context_during_execute), "every iteration ran inside the context"
        assert len(backend.context_during_execute) == 7

    def test_context_is_not_active_during_setup(self, engine: BenchmarkEngine) -> None:
        backend = ProbeBackend()
        engine.run(backend, cfg())
        enter = backend.events.index("context-enter")
        assert backend.events.index("prepare") < enter
        assert backend.events.index("context-exit") < backend.events.index("close")

    def test_context_exits_even_when_execution_fails(self, engine: BenchmarkEngine) -> None:
        backend = ProbeBackend(fail_on_execute_call=2)
        engine.run(backend, cfg())
        assert "context-exit" in backend.events


class TestSecondaryTiming:
    def test_secondary_samples_are_stored_separately(self, engine: BenchmarkEngine) -> None:
        timer = SecondaryTimer(
            primary=[1.0, 1.0, 2.0, 2.0, 2.0], secondary=[9.0, 9.0, 3.0, 3.0, 3.0]
        )
        result = engine.run(ProbeBackend(timer=timer), cfg(warmup=2, iterations=3))
        assert result.raw_samples.latency_ms == [2.0, 2.0, 2.0]
        assert result.raw_samples.secondary_latency_ms == [3.0, 3.0, 3.0]
        assert result.raw_samples.secondary_warmup_latency_ms == [9.0, 9.0]
        assert result.secondary_timing_mechanism is TimingMechanism.WALL_CLOCK_SYNCHRONIZED

    def test_secondary_never_becomes_the_headline(self, engine: BenchmarkEngine) -> None:
        timer = SecondaryTimer(primary=[2.0] * 3, secondary=[50.0] * 3)
        result = engine.run(ProbeBackend(timer=timer), cfg(warmup=0, iterations=3))
        assert result.latency is not None and result.latency.mean_ms == pytest.approx(2.0)
        assert result.secondary_latency is not None
        assert result.secondary_latency.mean_ms == pytest.approx(50.0)
        assert result.timing_mechanism is TimingMechanism.SCRIPTED

    def test_no_secondary_by_default(self, engine: BenchmarkEngine) -> None:
        result = engine.run(ProbeBackend(), cfg())
        assert result.secondary_timing_mechanism is None
        assert result.secondary_latency is None
        assert result.raw_samples.secondary_latency_ms == []


class TestDescriptorReadAfterRun:
    def test_settings_reflect_what_was_applied(self, engine: BenchmarkEngine) -> None:
        """Effective settings exist only after load(); the result must carry those."""
        result = engine.run(ProbeBackend(), cfg())
        assert result.backend.settings == {"stage": "applied-in-load"}

    def test_model_info_is_recorded(self, engine: BenchmarkEngine) -> None:
        result = engine.run(ProbeBackend(), cfg())
        assert result.model_info is not None and result.model_info.architecture == "Probe"


class TestCpuLabelling:
    def test_cpu_result_is_labelled(self, engine: BenchmarkEngine) -> None:
        result = engine.run(ProbeBackend(device_kind=DeviceKind.CPU), cfg())
        assert result.device_kind is DeviceKind.CPU
        assert CPU_RESULT_NOTE in result.notes

    def test_cpu_result_is_never_a_gpu_measurement(self) -> None:
        """Construct a successful, non-simulated CPU result and check the predicates."""
        data = json.loads(FIXTURE_1_0.read_text(encoding="utf-8"))
        data.update(is_simulated=False, device_kind="cpu", schema_version="1.1")
        result = BenchmarkResult.model_validate(data)
        assert result.is_real_measurement is True
        assert result.is_gpu_measurement is False

    def test_cuda_result_can_be_a_gpu_measurement(self) -> None:
        data = json.loads(FIXTURE_1_0.read_text(encoding="utf-8"))
        data.update(is_simulated=False, device_kind="cuda", schema_version="1.1")
        assert BenchmarkResult.model_validate(data).is_gpu_measurement is True

    def test_simulated_is_neither(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1), cfg())
        assert result.device_kind is DeviceKind.SIMULATED
        assert SIMULATION_WARNING in result.notes
        assert CPU_RESULT_NOTE not in result.notes
        assert result.is_gpu_measurement is False

    def test_cpu_results_get_a_cpu_prefix_on_disk(self, tmp_path: Path, environment) -> None:
        engine = BenchmarkEngine(environment=environment, experiment_id_factory=lambda: "c1")
        result = engine.run(ProbeBackend(device_kind=DeviceKind.CPU), cfg())
        directory = ResultStore(tmp_path).save(result)
        assert directory.name == "cpu-c1"
        for name in ("result.json", "metadata.json", "raw.json", "summary.json"):
            data = json.loads((directory / name).read_text(encoding="utf-8"))
            assert data["device_kind"] == "cpu", f"{name} does not label the device"
        assert ResultStore(tmp_path).load("c1").experiment_id == "c1"


class TestSchemaVersioning:
    def test_current_version_is_1_1(self) -> None:
        assert RESULT_SCHEMA_VERSION == "1.1"

    def test_a_real_schema_1_0_result_still_loads(self) -> None:
        """Committed Phase 2 output, unmodified."""
        data = json.loads(FIXTURE_1_0.read_text(encoding="utf-8"))
        assert data["schema_version"] == "1.0"
        result = BenchmarkResult.model_validate(data)
        assert result.device_kind is None
        assert result.model_info is None
        assert result.secondary_latency is None
        assert result.raw_samples.secondary_latency_ms == []

    def test_new_results_are_stamped_1_1(self, engine: BenchmarkEngine) -> None:
        assert engine.run(ProbeBackend(), cfg()).schema_version == "1.1"


class TestConfigAdditions:
    def test_weights_and_seed(self) -> None:
        c = parse_config(
            {"name": "x", "backend": "fake", "model": {"name": "m", "weights": "random", "seed": 7}}
        )
        assert c.model.weights == "random"
        assert c.model.seed == 7

    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            parse_config({"name": "x", "backend": "fake", "model": {"name": "m", "seed": -1}})

    def test_backend_options_default_empty(self) -> None:
        assert (
            parse_config({"name": "x", "backend": "fake", "model": {"name": "m"}}).backend_options
            == {}
        )

    def test_fake_backend_rejects_options(self) -> None:
        c = parse_config(
            {"name": "x", "backend": "fake", "model": {"name": "m"}, "backend_options": {"a": 1}}
        )
        with pytest.raises(ConfigurationError, match="accepts no backend_options"):
            build_backend(c, seed=0)
