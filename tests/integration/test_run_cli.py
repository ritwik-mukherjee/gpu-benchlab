"""End-to-end tests for `gpu-bench run`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gpu_benchlab.cli.main import app
from gpu_benchlab.cli.run_cmd import EXIT_BENCHMARK_NOT_OK, EXIT_CONFIG_ERROR, EXIT_OK

runner = CliRunner()

CONFIG = """
name: cli-test
backend: fake
precision: fp32
batch_size: 2
model:
  name: sim-model
  input_shape: [3, 224, 224]
benchmark:
  warmup_iterations: 3
  measurement_iterations: 40
"""


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "exp.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def invoke(config_file: Path, results_dir: Path, *extra: str):
    return runner.invoke(
        app,
        ["run", "-c", str(config_file), "--results-dir", str(results_dir), *extra],
    )


class TestSuccessfulRun:
    def test_exits_zero(self, config_file: Path, tmp_path: Path) -> None:
        result = invoke(config_file, tmp_path / "results")
        assert result.exit_code == EXIT_OK, result.stdout

    def test_reports_latency_percentiles(self, config_file: Path, tmp_path: Path) -> None:
        out = invoke(config_file, tmp_path / "results").stdout
        for label in ("p50", "p90", "p95", "p99", "Latency", "Throughput"):
            assert label in out

    def test_reports_phases_separately(self, config_file: Path, tmp_path: Path) -> None:
        out = invoke(config_file, tmp_path / "results").stdout
        assert "Engine build" in out
        assert "Warmup (total)" in out

    def test_writes_result_files(self, config_file: Path, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        invoke(config_file, results_dir)
        directories = list(results_dir.glob("sim-*"))
        assert len(directories) == 1
        for name in ("result.json", "metadata.json", "raw.json", "summary.json"):
            assert (directories[0] / name).is_file()

    def test_no_save_writes_nothing(self, config_file: Path, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        result = invoke(config_file, results_dir, "--no-save")
        assert result.exit_code == EXIT_OK
        assert not results_dir.exists() or not list(results_dir.glob("sim-*"))

    def test_raw_samples_match_configured_iterations(
        self, config_file: Path, tmp_path: Path
    ) -> None:
        results_dir = tmp_path / "results"
        invoke(config_file, results_dir)
        raw_path = next(results_dir.glob("sim-*/raw.json"))
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        assert len(raw["latency_ms"]) == 40
        assert len(raw["warmup_latency_ms"]) == 3


class TestSimulationIsUnmissable:
    def test_banner_is_printed(self, config_file: Path, tmp_path: Path) -> None:
        out = invoke(config_file, tmp_path / "results").stdout
        assert "SIMULATED" in out
        assert "NOT REAL PERFORMANCE DATA" in out

    def test_output_states_these_are_not_measurements(
        self, config_file: Path, tmp_path: Path
    ) -> None:
        out = invoke(config_file, tmp_path / "results").stdout.lower()
        assert "not measurements" in out

    def test_directory_is_prefixed(self, config_file: Path, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        invoke(config_file, results_dir)
        assert all(d.name.startswith("sim-") for d in results_dir.iterdir())

    def test_stored_result_is_flagged(self, config_file: Path, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        invoke(config_file, results_dir)
        data = json.loads(next(results_dir.glob("sim-*/result.json")).read_text(encoding="utf-8"))
        assert data["is_simulated"] is True
        assert data["timing_mechanism"] == "scripted"


class TestDeterminism:
    def test_same_seed_reproduces_samples(self, config_file: Path, tmp_path: Path) -> None:
        samples = []
        for i in range(2):
            results_dir = tmp_path / f"run{i}"
            invoke(config_file, results_dir, "--seed", "123")
            raw = json.loads(next(results_dir.glob("sim-*/raw.json")).read_text(encoding="utf-8"))
            samples.append(raw["latency_ms"])
        assert samples[0] == samples[1]

    def test_different_seed_changes_samples(self, config_file: Path, tmp_path: Path) -> None:
        samples = []
        for i, seed in enumerate(("1", "2")):
            results_dir = tmp_path / f"run{i}"
            invoke(config_file, results_dir, "--seed", seed)
            raw = json.loads(next(results_dir.glob("sim-*/raw.json")).read_text(encoding="utf-8"))
            samples.append(raw["latency_ms"])
        assert samples[0] != samples[1]


class TestConfigErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        result = invoke(tmp_path / "nope.yaml", tmp_path / "results")
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "not found" in result.stdout

    def test_invalid_config(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            "name: x\nbackend: fake\nmodel:\n  name: m\nbatch_size: -1\n", encoding="utf-8"
        )
        result = invoke(path, tmp_path / "results")
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "Configuration error" in result.stdout

    def test_unimplemented_backend_says_which_phase(self, tmp_path: Path) -> None:
        """A backend that is genuinely not implemented must name its phase.

        This asserted `tensorrt` until Phase 6 implemented it (`8ca5c7b`), at which
        point `tensorrt` left `known_but_unimplemented` and the assertion went stale.
        `tensorrt_llm` is the backend that is still unimplemented, so it now carries
        the contract; `tensorrt`'s own contract is the test below.
        """
        path = tmp_path / "trtllm.yaml"
        path.write_text("name: x\nbackend: tensorrt_llm\nmodel:\n  name: m\n", encoding="utf-8")
        result = invoke(path, tmp_path / "results")
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "not implemented yet" in result.stdout
        assert "Phase 10" in result.stdout

    def test_tensorrt_is_implemented_and_refuses_the_default_cpu_device(
        self, tmp_path: Path
    ) -> None:
        """`tensorrt` now reaches its backend, which has no CPU path.

        The config names no device, so the default is CPU. TensorRT must refuse that
        with a configuration error -- never accept it and quietly execute somewhere
        else -- and must no longer claim to be unimplemented.
        """
        path = tmp_path / "trt.yaml"
        path.write_text("name: x\nbackend: tensorrt\nmodel:\n  name: m\n", encoding="utf-8")
        result = invoke(path, tmp_path / "results")
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "requires an explicit CUDA device" in result.stdout
        assert "no CPU execution path" in result.stdout
        assert "not implemented yet" not in result.stdout

    def test_unknown_backend(self, tmp_path: Path) -> None:
        path = tmp_path / "nope.yaml"
        path.write_text("name: x\nbackend: wat\nmodel:\n  name: m\n", encoding="utf-8")
        result = invoke(path, tmp_path / "results")
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "Unknown backend" in result.stdout


class TestFailurePaths:
    def test_zero_warmup_is_flagged_in_output(self, tmp_path: Path) -> None:
        path = tmp_path / "nowarm.yaml"
        path.write_text(
            "name: x\nbackend: fake\nmodel:\n  name: m\n"
            "benchmark:\n  warmup_iterations: 0\n  measurement_iterations: 10\n",
            encoding="utf-8",
        )
        out = invoke(path, tmp_path / "results").stdout
        assert "cold start" in out.lower()

    def test_low_confidence_is_surfaced(self, tmp_path: Path) -> None:
        path = tmp_path / "few.yaml"
        path.write_text(
            "name: x\nbackend: fake\nmodel:\n  name: m\n"
            "benchmark:\n  warmup_iterations: 1\n  measurement_iterations: 5\n",
            encoding="utf-8",
        )
        out = invoke(path, tmp_path / "results").stdout
        assert "low confidence" in out


class TestExampleConfig:
    def test_shipped_example_runs(self, tmp_path: Path) -> None:
        """The example in the repo must actually work."""
        example = Path("examples/simulated-smoke-test.yaml")
        assert example.is_file(), "shipped example config is missing"
        result = invoke(example, tmp_path / "results")
        assert result.exit_code == EXIT_OK, result.stdout


class TestExitCodes:
    def test_benchmark_failure_exits_nonzero(self, tmp_path: Path) -> None:
        """A suite runner must be able to detect a non-ok benchmark from the shell."""
        assert EXIT_BENCHMARK_NOT_OK != EXIT_OK
        assert EXIT_CONFIG_ERROR != EXIT_OK
