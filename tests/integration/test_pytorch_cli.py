"""End-to-end CLI tests for the PyTorch backend and `gpu-bench models`."""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gpu_benchlab.cli.main import app
from gpu_benchlab.cli.run_cmd import EXIT_BENCHMARK_NOT_OK, EXIT_OK

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

runner = CliRunner()


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "exp.yaml"
    path.write_text(text, encoding="utf-8")
    return path


CPU_RESNET = """
name: cli-resnet50-cpu
backend: pytorch
device: cpu
precision: fp32
batch_size: 1
model:
  name: resnet50
  weights: random
benchmark:
  warmup_iterations: 1
  measurement_iterations: 4
"""


class TestModelsCommand:
    def test_list_shows_resnet50_and_its_pinned_parameter_count(self) -> None:
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0
        assert "resnet50" in result.stdout
        assert "25,557,032" in result.stdout
        assert "IMAGENET1K_V2" in result.stdout

    def test_fetch_unknown_model_fails_cleanly(self) -> None:
        result = runner.invoke(app, ["models", "fetch", "nope"])
        assert result.exit_code == 1
        assert "Unknown model" in result.stdout

    def test_fetch_random_is_rejected(self) -> None:
        result = runner.invoke(app, ["models", "fetch", "resnet50", "--weights", "random"])
        assert result.exit_code == 1


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA path")
class TestCudaExampleOnThisMachine:
    def test_shipped_cuda_example_is_unavailable_here(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "run",
                "-c",
                "examples/resnet50-pytorch-cuda-fp16.yaml",
                "--results-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == EXIT_BENCHMARK_NOT_OK
        assert "unavailable" in result.stdout
        assert "UnavailableError" in result.stdout
        assert "Latency" not in result.stdout, "no latency table for a run that never ran"

        # The terminal wraps long messages inside table cells; the stored record is
        # the authoritative, unwrapped one.
        stored = json.loads(next(tmp_path.glob("*/result.json")).read_text(encoding="utf-8"))
        assert "Not falling back to CPU" in stored["errors"][0]["message"]
        assert stored["status"] == "unavailable"
        assert stored["device_kind"] == "cuda"
        assert stored["timing_mechanism"] is None
        assert stored["raw_samples"]["latency_ms"] == []


@pytest.mark.slow
class TestCpuResNet50ThroughCli:
    def test_runs_and_is_labelled_cpu_everywhere(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        out = runner.invoke(
            app, ["run", "-c", str(write(tmp_path, CPU_RESNET)), "--results-dir", str(results)]
        )
        assert out.exit_code == EXIT_OK, out.stdout
        assert "CPU RESULT" in out.stdout
        assert "NOT GPU PERFORMANCE" in out.stdout

        directory = next(results.iterdir())
        assert directory.name.startswith("cpu-")
        for name in ("result.json", "metadata.json", "raw.json", "summary.json"):
            data = json.loads((directory / name).read_text(encoding="utf-8"))
            assert data["device_kind"] == "cpu", name
            assert data["is_simulated"] is False, name

        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        assert result["model_info"]["parameter_count"] == 25_557_032
        assert result["backend"]["settings"]["fp32_precision.cudnn.conv"] == "ieee"
        assert any("CPU RESULT" in n for n in result["notes"])

    def test_stored_statistics_rederive_from_raw_samples(self, tmp_path: Path) -> None:
        """Recompute from raw.json with the stdlib only, not the project's code."""
        results = tmp_path / "results"
        runner.invoke(
            app, ["run", "-c", str(write(tmp_path, CPU_RESNET)), "--results-dir", str(results)]
        )
        directory = next(results.iterdir())
        raw = json.loads((directory / "raw.json").read_text(encoding="utf-8"))["latency_ms"]
        lat = json.loads((directory / "summary.json").read_text(encoding="utf-8"))["latency_ms"]

        def pct(values: list[float], p: float) -> float:
            v = sorted(values)
            k = (len(v) - 1) * p / 100
            f = int(k)
            c = min(f + 1, len(v) - 1)
            return v[f] + (v[c] - v[f]) * (k - f)

        assert lat["mean_ms"] == pytest.approx(st.fmean(raw), abs=1e-9)
        assert lat["median_ms"] == pytest.approx(st.median(raw), abs=1e-9)
        assert lat["stddev_ms"] == pytest.approx(st.stdev(raw), abs=1e-9)
        for p in (50, 90, 95, 99):
            assert lat[f"p{p}_ms"] == pytest.approx(pct(raw, p), abs=1e-9)
