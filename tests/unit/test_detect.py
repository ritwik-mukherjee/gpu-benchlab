"""Tests for the environment detection orchestrator and its schema.

The report is embedded into every benchmark result, so it must round-trip
losslessly through JSON and must never claim a GPU it did not enumerate.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from gpu_benchlab.hardware import (
    ENVIRONMENT_SCHEMA_VERSION,
    DetectionStatus,
    EnvironmentReport,
    detect_environment,
    probe_host,
)


class TestHostProbe:
    def test_reports_real_host_facts(self) -> None:
        host = probe_host()
        assert host.hostname
        assert host.os_name
        assert host.architecture
        assert host.python_version.startswith("3.")
        assert host.python_executable

    def test_core_counts_are_sane_when_reported(self) -> None:
        host = probe_host()
        if host.cpu_logical_cores is not None:
            assert host.cpu_logical_cores >= 1
        if host.cpu_physical_cores is not None and host.cpu_logical_cores is not None:
            assert host.cpu_physical_cores <= host.cpu_logical_cores


class TestDetectEnvironment:
    """These run against the real machine, whatever it is."""

    def test_detection_never_raises(self) -> None:
        detect_environment(include_frameworks=False)

    def test_schema_version_is_stamped(self) -> None:
        report = detect_environment(include_frameworks=False)
        assert report.schema_version == ENVIRONMENT_SCHEMA_VERSION

    def test_skip_frameworks_is_honoured(self) -> None:
        report = detect_environment(include_frameworks=False)
        assert report.frameworks == []

    def test_frameworks_probed_when_requested(self) -> None:
        report = detect_environment(include_frameworks=True)
        names = {fw.name for fw in report.frameworks}
        assert {"torch", "onnxruntime", "tensorrt", "tensorrt_llm"} <= names

    def test_has_nvidia_gpu_is_consistent_with_status(self) -> None:
        """The key invariant: never claim a GPU unless one was actually enumerated."""
        report = detect_environment(include_frameworks=False)
        if report.has_nvidia_gpu:
            assert report.detection_status is DetectionStatus.OK
            assert len(report.gpus) > 0
        else:
            assert report.detection_status is not DetectionStatus.OK or not report.gpus

    def test_failure_carries_an_explanation(self) -> None:
        report = detect_environment(include_frameworks=False)
        if report.detection_status is not DetectionStatus.OK:
            assert report.detection_error, "a non-OK status must explain itself"

    def test_timestamp_is_utc_iso(self) -> None:
        report = detect_environment(include_frameworks=False)
        assert "T" in report.timestamp_utc
        assert report.timestamp_utc.endswith("+00:00") or report.timestamp_utc.endswith("Z")


class TestSerialisation:
    def test_round_trips_through_json(self) -> None:
        report = detect_environment(include_frameworks=True)
        payload = report.model_dump_json()
        restored = EnvironmentReport.model_validate_json(payload)
        assert restored == report

    def test_json_is_a_plain_object(self) -> None:
        report = detect_environment(include_frameworks=False)
        data = json.loads(report.model_dump_json())
        for key in ("schema_version", "timestamp_utc", "host", "gpus", "detection_status"):
            assert key in data

    def test_enum_serialises_as_its_value(self) -> None:
        report = detect_environment(include_frameworks=False)
        data = json.loads(report.model_dump_json())
        assert isinstance(data["detection_status"], str)
        assert data["detection_status"] in {s.value for s in DetectionStatus}

    def test_report_is_immutable(self) -> None:
        """A report embedded in a result must not be mutable after the fact."""
        report = detect_environment(include_frameworks=False)
        with pytest.raises(ValidationError):
            report.schema_version = "9.9"  # type: ignore[misc]


class TestFrameworkLookup:
    def test_framework_accessor(self) -> None:
        report = detect_environment(include_frameworks=True)
        assert report.framework("torch") is not None
        assert report.framework("definitely-not-a-framework") is None


class TestPowerSource:
    """Power source was the dominant confounder in the 2026-09-18 A/B run."""

    def _battery(self, plugged: bool | None, percent: float):
        from types import SimpleNamespace

        return SimpleNamespace(power_plugged=plugged, percent=percent, secsleft=0)

    def test_on_battery(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import psutil

        monkeypatch.setattr(psutil, "sensors_battery", lambda: self._battery(False, 72.0))
        host = probe_host()
        assert host.power_plugged is False
        assert host.battery_percent == 72.0

    def test_on_ac(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import psutil

        monkeypatch.setattr(psutil, "sensors_battery", lambda: self._battery(True, 98.0))
        assert probe_host().power_plugged is True

    def test_unknown_plug_state_is_none_not_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """psutil may report power_plugged=None; that must not become 'on battery'."""
        import psutil

        monkeypatch.setattr(psutil, "sensors_battery", lambda: self._battery(None, 50.0))
        assert probe_host().power_plugged is None

    def test_no_battery_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import psutil

        monkeypatch.setattr(psutil, "sensors_battery", lambda: None)
        host = probe_host()
        assert host.power_plugged is None
        assert host.battery_percent is None

    def test_environment_schema_is_1_1_and_1_0_still_loads(self) -> None:
        import json
        from pathlib import Path

        from gpu_benchlab.hardware import EnvironmentReport

        assert ENVIRONMENT_SCHEMA_VERSION == "1.2"
        fixture = Path(__file__).parent.parent / "fixtures" / "result_schema_1_0.json"
        env = json.loads(fixture.read_text(encoding="utf-8"))["environment"]
        assert env["schema_version"] == "1.0"
        report = EnvironmentReport.model_validate(env)
        assert report.host.power_plugged is None
