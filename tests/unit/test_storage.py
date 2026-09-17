"""Tests for result persistence and the result schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_benchlab.backends.fake import FakeBackend
from gpu_benchlab.core.config import BenchmarkConfig, ExperimentConfig, ModelConfig
from gpu_benchlab.core.engine import BenchmarkEngine
from gpu_benchlab.core.schema import RESULT_SCHEMA_VERSION, BenchmarkResult, BenchmarkStatus
from gpu_benchlab.core.storage import ResultStore, result_directory_name
from gpu_benchlab.hardware.detect import detect_environment


@pytest.fixture(scope="module")
def environment():
    return detect_environment(include_frameworks=False)


def make_config(**kw: object) -> ExperimentConfig:
    return ExperimentConfig(
        name="storage-test",
        model=ModelConfig(name="sim-model", input_shape=[3, 224, 224]),
        backend="fake",
        benchmark=BenchmarkConfig(warmup_iterations=3, measurement_iterations=15),
        **kw,  # type: ignore[arg-type]
    )


@pytest.fixture
def result(environment) -> BenchmarkResult:
    engine = BenchmarkEngine(environment=environment, experiment_id_factory=lambda: "abc123")
    return engine.run(FakeBackend(seed=7), make_config())


class TestSchema:
    def test_version_is_stamped(self, result: BenchmarkResult) -> None:
        assert result.schema_version == RESULT_SCHEMA_VERSION

    def test_round_trips_through_json(self, result: BenchmarkResult) -> None:
        restored = BenchmarkResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_raw_samples_survive_round_trip_exactly(self, result: BenchmarkResult) -> None:
        """Float precision must not degrade; statistics are recomputed from these."""
        restored = BenchmarkResult.model_validate_json(result.model_dump_json())
        assert restored.raw_samples.latency_ms == result.raw_samples.latency_ms

    def test_result_is_frozen(self, result: BenchmarkResult) -> None:
        with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
            result.status = BenchmarkStatus.FAILED  # type: ignore[misc]

    def test_contains_every_required_section(self, result: BenchmarkResult) -> None:
        data = json.loads(result.model_dump_json())
        for key in (
            "schema_version",
            "experiment_id",
            "timestamp_utc",
            "status",
            "is_simulated",
            "configuration",
            "backend",
            "environment",
            "timing_mechanism",
            "phases",
            "latency",
            "throughput",
            "raw_samples",
            "provenance",
            "errors",
            "notes",
        ):
            assert key in data, f"missing schema section: {key}"

    def test_environment_is_embedded_not_referenced(self, result: BenchmarkResult) -> None:
        data = json.loads(result.model_dump_json())
        assert data["environment"]["host"]["hostname"]
        assert "detection_status" in data["environment"]

    def test_summary_dict(self, result: BenchmarkResult) -> None:
        summary = result.summary_dict()
        assert summary["experiment_id"] == "abc123"
        assert summary["is_simulated"] is True
        assert summary["samples"] == 15


class TestStoreWrite:
    def test_writes_all_four_files(self, tmp_path: Path, result: BenchmarkResult) -> None:
        directory = ResultStore(tmp_path).save(result)
        for name in ("result.json", "metadata.json", "raw.json", "summary.json"):
            assert (directory / name).is_file(), f"missing {name}"
        assert (directory / "logs").is_dir()

    def test_simulated_results_get_a_prefixed_directory(
        self, tmp_path: Path, result: BenchmarkResult
    ) -> None:
        """Visible in a file listing before anyone opens anything."""
        directory = ResultStore(tmp_path).save(result)
        assert directory.name == "sim-abc123"
        assert result_directory_name(result).startswith("sim-")

    def test_every_file_is_valid_json(self, tmp_path: Path, result: BenchmarkResult) -> None:
        directory = ResultStore(tmp_path).save(result)
        for name in ("result.json", "metadata.json", "raw.json", "summary.json"):
            json.loads((directory / name).read_text(encoding="utf-8"))

    def test_every_file_carries_the_simulated_flag(
        self, tmp_path: Path, result: BenchmarkResult
    ) -> None:
        """No single file can be read in isolation and mistaken for real data."""
        directory = ResultStore(tmp_path).save(result)
        for name in ("result.json", "metadata.json", "raw.json", "summary.json"):
            data = json.loads((directory / name).read_text(encoding="utf-8"))
            assert data["is_simulated"] is True, f"{name} does not mark simulation"

    def test_raw_file_names_its_units(self, tmp_path: Path, result: BenchmarkResult) -> None:
        directory = ResultStore(tmp_path).save(result)
        raw = json.loads((directory / "raw.json").read_text(encoding="utf-8"))
        assert raw["units"] == "milliseconds"
        assert len(raw["latency_ms"]) == 15
        assert len(raw["warmup_latency_ms"]) == 3

    def test_summary_matches_result(self, tmp_path: Path, result: BenchmarkResult) -> None:
        """The split views are derived from one object and cannot disagree."""
        directory = ResultStore(tmp_path).save(result)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        assert result.latency is not None
        assert summary["latency_ms"]["p50_ms"] == pytest.approx(result.latency.p50_ms)
        assert summary["status"] == result.status.value

    def test_creates_missing_directories(self, tmp_path: Path, result: BenchmarkResult) -> None:
        store = ResultStore(tmp_path / "deep" / "nested")
        assert store.save(result).is_dir()


class TestStoreRead:
    def test_round_trips_through_disk(self, tmp_path: Path, result: BenchmarkResult) -> None:
        store = ResultStore(tmp_path)
        store.save(result)
        assert store.load("abc123") == result

    def test_load_accepts_the_prefixed_name(self, tmp_path: Path, result: BenchmarkResult) -> None:
        store = ResultStore(tmp_path)
        store.save(result)
        assert store.load("sim-abc123").experiment_id == "abc123"

    def test_missing_result_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="nope"):
            ResultStore(tmp_path).load("nope")

    def test_list_empty_root(self, tmp_path: Path) -> None:
        assert ResultStore(tmp_path / "absent").list_results() == []

    def test_lists_saved_results(self, tmp_path: Path, environment) -> None:
        store = ResultStore(tmp_path)
        for i in range(3):
            engine = BenchmarkEngine(
                environment=environment, experiment_id_factory=lambda i=i: f"exp{i}"
            )
            store.save(engine.run(FakeBackend(seed=i), make_config()))
        assert len(store.list_results()) == 3

    def test_corrupt_result_does_not_break_the_listing(
        self, tmp_path: Path, result: BenchmarkResult
    ) -> None:
        store = ResultStore(tmp_path)
        store.save(result)
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / "result.json").write_text("{not json", encoding="utf-8")
        assert len(store.list_results()) == 1


class TestFailedResultPersistence:
    def test_failed_result_is_saved_with_its_error(self, tmp_path: Path, environment) -> None:
        """A failure is a result and must be as durable as a success."""
        engine = BenchmarkEngine(environment=environment, experiment_id_factory=lambda: "fail1")
        result = engine.run(FakeBackend(seed=1, fail_on_load=True), make_config())

        directory = ResultStore(tmp_path).save(result)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))

        assert summary["status"] == "failed"
        assert summary["latency_ms"] is None
        assert summary["errors"][0]["type"] == "BackendError"

    def test_unsupported_result_is_saved(self, tmp_path: Path, environment) -> None:
        from gpu_benchlab.hardware.capability import Precision

        engine = BenchmarkEngine(environment=environment, experiment_id_factory=lambda: "uns1")
        backend = FakeBackend(seed=1, unsupported_precisions=frozenset({Precision.FP32}))
        result = engine.run(backend, make_config())

        store = ResultStore(tmp_path)
        store.save(result)
        assert store.load("uns1").status is BenchmarkStatus.UNSUPPORTED
