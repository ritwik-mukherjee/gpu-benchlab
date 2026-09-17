"""Tests for the benchmark engine.

Covers the guarantees the engine makes: every experiment yields a result, phases
never leak into each other, raw samples are preserved, failures are recorded
rather than raised, and simulation is contagious.
"""

from __future__ import annotations

import pytest

from gpu_benchlab.backends.fake import FakeBackend
from gpu_benchlab.core.config import BenchmarkConfig, ExperimentConfig, ModelConfig
from gpu_benchlab.core.engine import NO_WARMUP_WARNING, BenchmarkEngine
from gpu_benchlab.core.schema import BenchmarkStatus, ThroughputUnit
from gpu_benchlab.core.timing import TimingMechanism
from gpu_benchlab.hardware.capability import Precision
from gpu_benchlab.hardware.detect import detect_environment


@pytest.fixture(scope="module")
def environment():
    # Detected once: it is slow and identical for every test here.
    return detect_environment(include_frameworks=False)


@pytest.fixture
def engine(environment) -> BenchmarkEngine:
    return BenchmarkEngine(environment=environment)


def make_config(
    *,
    warmup: int = 5,
    iterations: int = 20,
    batch_size: int = 1,
    precision: Precision = Precision.FP32,
) -> ExperimentConfig:
    return ExperimentConfig(
        name="test-experiment",
        model=ModelConfig(name="sim-model", input_shape=[3, 224, 224]),
        backend="fake",
        precision=precision,
        batch_size=batch_size,
        benchmark=BenchmarkConfig(warmup_iterations=warmup, measurement_iterations=iterations),
    )


class TestSuccessfulRun:
    def test_status_ok(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1), make_config())
        assert result.status is BenchmarkStatus.OK
        assert result.errors == []

    def test_sample_count_matches_configuration(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1), make_config(warmup=7, iterations=33))
        assert len(result.raw_samples.latency_ms) == 33
        assert len(result.raw_samples.warmup_latency_ms) == 7

    def test_backend_executed_warmup_plus_measurement(self, engine: BenchmarkEngine) -> None:
        backend = FakeBackend(seed=1)
        engine.run(backend, make_config(warmup=7, iterations=33))
        assert backend.execution_count == 40

    def test_statistics_derive_from_stored_samples(self, engine: BenchmarkEngine) -> None:
        """The stored aggregate must be recomputable from the stored raw data."""
        from gpu_benchlab.core.statistics import LatencyStatistics

        result = engine.run(FakeBackend(seed=1), make_config())
        assert result.latency is not None
        recomputed = LatencyStatistics.from_samples(result.raw_samples.latency_ms)
        assert recomputed.model_dump() == result.latency.model_dump()

    def test_warmup_samples_excluded_from_statistics(self, engine: BenchmarkEngine) -> None:
        """Warmup is slower by construction; if it leaked in, min would drop."""
        result = engine.run(FakeBackend(seed=1), make_config(warmup=5, iterations=20))
        assert result.latency is not None
        warmup = result.raw_samples.warmup_latency_ms
        assert max(warmup) > result.latency.max_ms, "warmup should be the slow phase"
        assert result.latency.count == 20

    def test_warmup_samples_are_retained(self, engine: BenchmarkEngine) -> None:
        """Retained so warmup convergence can be determined empirically."""
        result = engine.run(FakeBackend(seed=1), make_config(warmup=5))
        assert len(result.raw_samples.warmup_latency_ms) == 5

    def test_warmup_decays_towards_steady_state(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1, jitter_ms=0.0), make_config(warmup=6))
        warmup = result.raw_samples.warmup_latency_ms
        assert warmup == sorted(warmup, reverse=True), "warmup should decay monotonically"

    def test_closes_the_backend(self, engine: BenchmarkEngine) -> None:
        backend = FakeBackend(seed=1)
        engine.run(backend, make_config())
        assert backend.closed is True


class TestPhaseSeparation:
    def test_phases_are_recorded_separately(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1), make_config())
        phases = result.phases
        assert phases.model_load_ms is not None
        assert phases.engine_build_ms is not None
        assert phases.prepare_inputs_ms is not None
        assert phases.warmup_total_ms is not None
        assert phases.measurement_total_ms is not None

    def test_engine_build_never_leaks_into_inference_latency(self, engine: BenchmarkEngine) -> None:
        """The core anti-regression test for PRD §29.

        One backend has a genuinely expensive engine build (a real 60ms), the
        other has none. Everything else is identical, including the seed. If
        build time leaked into the measurement, the latency statistics would
        differ. They must be byte-identical, while engine_build_ms differs by
        the full cost.
        """
        config = make_config(warmup=2, iterations=10)

        baseline = engine.run(FakeBackend(seed=1, jitter_ms=0.0), config)
        with_build = engine.run(
            FakeBackend(seed=1, jitter_ms=0.0, engine_build_ms=60.0, model_load_ms=20.0),
            config,
        )

        assert baseline.latency is not None
        assert with_build.latency is not None

        # The expensive build really happened...
        assert with_build.phases.engine_build_ms is not None
        assert with_build.phases.engine_build_ms > 50.0
        assert with_build.phases.model_load_ms is not None
        assert with_build.phases.model_load_ms > 15.0
        assert (baseline.phases.engine_build_ms or 0.0) < 10.0

        # ...and none of it reached the inference measurement.
        assert with_build.latency.model_dump() == baseline.latency.model_dump()
        assert with_build.raw_samples.latency_ms == baseline.raw_samples.latency_ms

    def test_zero_warmup_records_zero_total(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1), make_config(warmup=0))
        assert result.phases.warmup_total_ms == 0.0
        assert result.raw_samples.warmup_latency_ms == []


class TestSteadyStateFlag:
    def test_warmup_present_means_steady_state(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1), make_config(warmup=5))
        assert result.measures_steady_state is True
        assert NO_WARMUP_WARNING not in result.notes

    def test_zero_warmup_is_flagged_loudly(self, engine: BenchmarkEngine) -> None:
        """Reporting cold start as steady state must require deliberate action."""
        result = engine.run(FakeBackend(seed=1), make_config(warmup=0))
        assert result.measures_steady_state is False
        assert NO_WARMUP_WARNING in result.notes


class TestSimulationMarking:
    """Four independent markers, so a fake result cannot pass as a measurement."""

    def test_result_is_stamped_simulated(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1), make_config())
        assert result.is_simulated is True

    def test_backend_descriptor_is_stamped(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1), make_config())
        assert result.backend.is_simulated is True

    def test_timing_mechanism_is_scripted(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1), make_config())
        assert result.timing_mechanism is TimingMechanism.SCRIPTED

    def test_warning_note_is_attached(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1), make_config())
        assert any("SIMULATED RESULT" in note for note in result.notes)

    def test_is_real_measurement_is_false(self, engine: BenchmarkEngine) -> None:
        """The single check consumers should use before trusting a number."""
        result = engine.run(FakeBackend(seed=1), make_config())
        assert result.succeeded is True
        assert result.is_real_measurement is False


class TestThroughput:
    def test_derived_from_mean_latency(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1), make_config(batch_size=4))
        assert result.latency is not None
        by_mean = [t for t in result.throughput if t.basis == "batch_size / mean_latency_s"]
        assert len(by_mean) == 1
        expected = 4 / (result.latency.mean_ms / 1000.0)
        assert by_mean[0].value == pytest.approx(expected)

    def test_unit_is_always_named(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1), make_config())
        assert result.throughput
        for metric in result.throughput:
            assert isinstance(metric.unit, ThroughputUnit)
            assert metric.basis

    def test_observed_throughput_suppressed_under_scripted_timing(
        self, engine: BenchmarkEngine
    ) -> None:
        """Scripted durations and real wall time are not commensurable.

        Emitting their quotient would be a fabricated number, so nothing is emitted.
        """
        result = engine.run(FakeBackend(seed=1), make_config())
        bases = {t.basis for t in result.throughput}
        assert "measurement_iterations * batch_size / measurement_total_s" not in bases


class TestFailureHandling:
    def test_unsupported_precision_is_recorded_not_raised(self, engine: BenchmarkEngine) -> None:
        backend = FakeBackend(seed=1, unsupported_precisions=frozenset({Precision.FP8}))
        result = engine.run(backend, make_config(precision=Precision.FP8))
        assert result.status is BenchmarkStatus.UNSUPPORTED
        assert result.latency is None
        assert len(result.errors) == 1
        assert result.errors[0].phase == "validate"

    def test_oom_is_a_failed_result_not_a_crash(self, engine: BenchmarkEngine) -> None:
        """An OOM at a batch size is informative data about that batch size."""
        backend = FakeBackend(seed=1, oom_above_batch_size=8)
        result = engine.run(backend, make_config(batch_size=16))
        assert result.status is BenchmarkStatus.FAILED
        assert result.errors[0].type == "OutOfMemoryError"
        assert "out of memory" in result.errors[0].message.lower()

    def test_oom_below_threshold_succeeds(self, engine: BenchmarkEngine) -> None:
        backend = FakeBackend(seed=1, oom_above_batch_size=8)
        result = engine.run(backend, make_config(batch_size=8))
        assert result.status is BenchmarkStatus.OK

    def test_load_failure_is_recorded(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1, fail_on_load=True), make_config())
        assert result.status is BenchmarkStatus.FAILED
        assert "load" in result.errors[0].message.lower()

    def test_build_failure_is_recorded(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1, fail_on_build=True), make_config())
        assert result.status is BenchmarkStatus.FAILED
        assert "build" in result.errors[0].message.lower()

    def test_mid_run_execution_failure_is_recorded(self, engine: BenchmarkEngine) -> None:
        backend = FakeBackend(seed=1, fail_on_execute_iteration=10)
        result = engine.run(backend, make_config(warmup=2, iterations=50))
        assert result.status is BenchmarkStatus.FAILED
        assert result.latency is None

    def test_nan_sample_fails_rather_than_reporting_statistics(
        self, engine: BenchmarkEngine
    ) -> None:
        """A NaN means the timer misbehaved; no statistic may be derived from it."""
        backend = FakeBackend(seed=1, emit_nan_at_iteration=3)
        result = engine.run(backend, make_config(warmup=0, iterations=10))
        assert result.status is BenchmarkStatus.FAILED
        assert result.latency is None
        assert result.errors[0].type == "InvalidSampleError"

    def test_errors_carry_a_traceback_and_timestamp(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1, fail_on_load=True), make_config())
        error = result.errors[0]
        assert error.traceback and "Traceback" in error.traceback
        assert error.timestamp_utc

    def test_backend_closed_even_after_failure(self, engine: BenchmarkEngine) -> None:
        backend = FakeBackend(seed=1, fail_on_load=True)
        engine.run(backend, make_config())
        assert backend.closed is True

    def test_failed_run_still_records_full_environment(self, engine: BenchmarkEngine) -> None:
        """A failure is a result; it must be as traceable as a success."""
        result = engine.run(FakeBackend(seed=1, fail_on_load=True), make_config())
        assert result.environment.schema_version
        assert result.configuration.name == "test-experiment"
        assert result.provenance.benchlab_version

    def test_engine_never_raises(self, engine: BenchmarkEngine) -> None:
        for backend in (
            FakeBackend(seed=1, fail_on_load=True),
            FakeBackend(seed=1, fail_on_build=True),
            FakeBackend(seed=1, oom_above_batch_size=0),
            FakeBackend(seed=1, emit_nan_at_iteration=0),
        ):
            engine.run(backend, make_config(warmup=0, iterations=5))


class TestProvenanceAndIdentity:
    def test_experiment_ids_are_unique(self, engine: BenchmarkEngine) -> None:
        ids = {engine.run(FakeBackend(seed=1), make_config()).experiment_id for _ in range(5)}
        assert len(ids) == 5

    def test_group_id_and_repeat_index_are_preserved(self, engine: BenchmarkEngine) -> None:
        result = engine.run(FakeBackend(seed=1), make_config(), group_id="grp-1", repeat_index=2)
        assert result.group_id == "grp-1"
        assert result.repeat_index == 2

    def test_deterministic_id_factory_is_honoured(self, environment) -> None:
        engine = BenchmarkEngine(environment=environment, experiment_id_factory=lambda: "fixed")
        assert engine.run(FakeBackend(seed=1), make_config()).experiment_id == "fixed"

    def test_configuration_is_stored_verbatim(self, engine: BenchmarkEngine) -> None:
        config = make_config(batch_size=4, warmup=3, iterations=11)
        result = engine.run(FakeBackend(seed=1), config)
        assert result.configuration == config


class TestDeterminism:
    def test_same_seed_gives_identical_samples(self, engine: BenchmarkEngine) -> None:
        config = make_config()
        first = engine.run(FakeBackend(seed=99), config)
        second = engine.run(FakeBackend(seed=99), config)
        assert first.raw_samples.latency_ms == second.raw_samples.latency_ms

    def test_different_seeds_give_different_samples(self, engine: BenchmarkEngine) -> None:
        config = make_config()
        first = engine.run(FakeBackend(seed=1), config)
        second = engine.run(FakeBackend(seed=2), config)
        assert first.raw_samples.latency_ms != second.raw_samples.latency_ms
