"""End-to-end CLI tests.

These invoke the real `gpu-bench` commands against the real machine. They assert
on behaviour that must hold regardless of whether a GPU is present, because the
test suite has to be meaningful on both CPU-only CI runners and GPU machines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gpu_benchlab import __version__
from gpu_benchlab.cli.main import EXIT_DETECTION_FAILED, EXIT_NO_GPU, EXIT_OK, app
from gpu_benchlab.hardware import detect_environment

runner = CliRunner()


@pytest.fixture(scope="module")
def gpu_present() -> bool:
    return detect_environment(include_frameworks=False).has_nvidia_gpu


class TestVersion:
    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == EXIT_OK
        assert __version__ in result.stdout


class TestHardwareCommand:
    def test_runs_and_exits_with_a_defined_code(self) -> None:
        result = runner.invoke(app, ["hardware", "--skip-frameworks"])
        assert result.exit_code in (EXIT_OK, EXIT_NO_GPU, EXIT_DETECTION_FAILED)

    def test_exit_code_matches_gpu_presence(self, gpu_present: bool) -> None:
        result = runner.invoke(app, ["hardware", "--skip-frameworks"])
        if gpu_present:
            assert result.exit_code == EXIT_OK
        else:
            assert result.exit_code != EXIT_OK

    def test_human_output_always_states_a_verdict(self) -> None:
        """A user must never have to guess whether benchmarks can run here."""
        result = runner.invoke(app, ["hardware", "--skip-frameworks"])
        assert "Verdict" in result.stdout

    def test_reports_host_facts(self) -> None:
        result = runner.invoke(app, ["hardware", "--skip-frameworks"])
        assert "Host" in result.stdout
        assert "Python" in result.stdout

    def test_json_output_is_valid_json(self) -> None:
        result = runner.invoke(app, ["hardware", "--json", "--skip-frameworks"])
        data = json.loads(result.stdout)
        assert data["schema_version"]
        assert "detection_status" in data
        assert "host" in data

    def test_output_file_is_written(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "environment.json"
        runner.invoke(app, ["hardware", "--skip-frameworks", "-o", str(target)])
        assert target.exists(), "parent directories must be created"
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["benchlab_version"] == __version__

    def test_no_gpu_output_never_implies_a_measurement(self, gpu_present: bool) -> None:
        """On a GPU-less machine the tool must refuse, not estimate (CLAUDE.md §0)."""
        if gpu_present:
            pytest.skip("this machine has a GPU; the refusal path is not exercised")
        result = runner.invoke(app, ["hardware", "--skip-frameworks"])
        lowered = result.stdout.lower()
        assert "no usable nvidia gpu" in lowered
        assert "cannot be produced" in lowered


class TestDoctorCommand:
    def test_runs(self) -> None:
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code in (EXIT_OK, EXIT_NO_GPU, EXIT_DETECTION_FAILED)

    def test_lists_every_framework(self) -> None:
        result = runner.invoke(app, ["doctor"])
        for name in ("torch", "onnxruntime", "tensorrt"):
            assert name in result.stdout

    def test_reports_gpu_check(self) -> None:
        result = runner.invoke(app, ["doctor"])
        assert "NVIDIA GPU present" in result.stdout


class TestHelp:
    def test_bare_invocation_shows_help(self) -> None:
        result = runner.invoke(app, [])
        assert "hardware" in result.stdout
        assert "doctor" in result.stdout
