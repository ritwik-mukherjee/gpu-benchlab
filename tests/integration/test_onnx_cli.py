"""End-to-end CLI tests for `gpu-bench onnx` and `backend: onnxruntime`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from gpu_benchlab.cli.main import app  # noqa: E402
from gpu_benchlab.cli.run_cmd import EXIT_BENCHMARK_NOT_OK, EXIT_OK  # noqa: E402
from gpu_benchlab.models.weights import default_cache_dir  # noqa: E402

pytestmark = [pytest.mark.torch, pytest.mark.onnx]
runner = CliRunner()

RESNET_CACHED = (default_cache_dir() / "resnet50-11ad3fa6.pth").is_file()
needs_resnet = pytest.mark.skipif(not RESNET_CACHED, reason="pinned ResNet-50 weights not cached")


class TestOnnxCommands:
    def test_help_lists_export_and_verify(self) -> None:
        out = runner.invoke(app, ["onnx", "--help"]).stdout
        assert "export" in out and "verify" in out

    def test_unknown_model_fails_cleanly(self) -> None:
        result = runner.invoke(app, ["onnx", "export", "nope"])
        assert result.exit_code == 1
        assert "Unknown model" in result.stdout


@pytest.mark.skipif(
    "CUDAExecutionProvider" in ort.get_available_providers(),
    reason="checks the CPU-only package path",
)
def test_shipped_cuda_example_is_unavailable(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "-c",
            "examples/resnet50-onnxruntime-cuda-fp32.yaml",
            "--results-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == EXIT_BENCHMARK_NOT_OK
    stored = json.loads(next(tmp_path.glob("*/result.json")).read_text(encoding="utf-8"))
    assert stored["status"] == "unavailable"
    assert stored["backend"]["execution_provider"] == "CUDAExecutionProvider"
    assert stored["raw_samples"]["latency_ms"] == []
    assert "Not falling back to CPU" in stored["errors"][0]["message"]


@pytest.mark.slow
@needs_resnet
class TestResNet50ThroughCli:
    """REAL: pinned ResNet-50, exported and run by ONNX Runtime on the CPU EP."""

    def test_verify_passes_and_stores_evidence(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["onnx", "verify", "resnet50", "--results-dir", str(tmp_path)])
        assert result.exit_code == EXIT_OK, result.stdout
        assert "PASSED" in result.stdout and "9/9" in result.stdout
        assert "rejected, as required" in result.stdout
        directory = next(tmp_path.glob("correctness-*"))
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        outputs = np.load(directory / "outputs.npz")
        assert report["passed"] is True and len(outputs.files) == 9 * 2 + 1

    def test_cpu_run_is_labelled_not_for_ranking(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "run",
                "-c",
                "examples/resnet50-onnxruntime-cpu-fp32.yaml",
                "--results-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == EXIT_OK, result.stdout
        flat = " ".join(result.stdout.split())
        assert "not suitable for backend performance ranking" in flat
        directory = next(tmp_path.iterdir())
        assert directory.name.startswith("cpu-")
        stored = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        assert stored["backend"]["name"] == "onnxruntime"
        assert stored["backend"]["settings"]["active_providers"] == "CPUExecutionProvider"
        assert stored["model_info"]["weights_sha256"].startswith("11ad3fa6")
        assert len(stored["raw_samples"]["latency_ms"]) == 100
