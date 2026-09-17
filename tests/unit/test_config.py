"""Tests for experiment configuration and its validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_benchlab.core.config import (
    DEFAULT_MEASUREMENT_ITERATIONS,
    DEFAULT_WARMUP_ITERATIONS,
    BenchmarkConfig,
    ExperimentConfig,
    ModelConfig,
    load_config,
    parse_config,
)
from gpu_benchlab.core.errors import ConfigurationError
from gpu_benchlab.hardware.capability import Precision

MINIMAL = {"name": "exp", "model": {"name": "m"}, "backend": "fake"}


class TestDefaults:
    def test_documented_defaults(self) -> None:
        """Referenced by docs/methodology.md §4 as an unvalidated hypothesis."""
        assert DEFAULT_WARMUP_ITERATIONS == 10
        assert DEFAULT_MEASUREMENT_ITERATIONS == 100

    def test_minimal_config_is_valid(self) -> None:
        config = parse_config(MINIMAL)
        assert config.precision is Precision.FP32
        assert config.batch_size == 1
        assert config.benchmark.warmup_iterations == DEFAULT_WARMUP_ITERATIONS
        assert config.benchmark.repeats == 1


class TestValidation:
    def test_batch_size_must_be_positive(self) -> None:
        with pytest.raises(ConfigurationError, match="batch_size"):
            parse_config({**MINIMAL, "batch_size": 0})

    def test_measurement_iterations_must_be_positive(self) -> None:
        with pytest.raises(ConfigurationError, match="measurement_iterations"):
            parse_config({**MINIMAL, "benchmark": {"measurement_iterations": 0}})

    def test_warmup_may_be_zero(self) -> None:
        """Permitted, but it means cold start is included -- flagged, not blocked."""
        config = parse_config({**MINIMAL, "benchmark": {"warmup_iterations": 0}})
        assert config.measures_steady_state is False

    def test_negative_warmup_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            parse_config({**MINIMAL, "benchmark": {"warmup_iterations": -1}})

    def test_unknown_precision_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="precision"):
            parse_config({**MINIMAL, "precision": "fp13"})

    def test_typo_in_field_name_is_rejected(self) -> None:
        """extra='forbid': a silently ignored typo would run the wrong experiment."""
        with pytest.raises(ConfigurationError, match="batch_sizes"):
            parse_config({**MINIMAL, "batch_sizes": 4})

    def test_missing_required_field_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="model"):
            parse_config({"name": "exp", "backend": "fake"})

    def test_empty_name_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            parse_config({**MINIMAL, "name": ""})

    def test_all_errors_reported_not_just_the_first(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            parse_config({"name": "", "model": {"name": "m"}, "backend": "", "batch_size": -1})
        message = str(exc_info.value)
        assert message.count("- ") >= 3

    def test_input_shape_must_be_positive(self) -> None:
        with pytest.raises(ConfigurationError, match="positive"):
            parse_config({**MINIMAL, "model": {"name": "m", "input_shape": [3, 0, 224]}})

    def test_empty_input_shape_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="must not be empty"):
            parse_config({**MINIMAL, "model": {"name": "m", "input_shape": []}})

    def test_sequence_length_must_be_positive(self) -> None:
        with pytest.raises(ConfigurationError):
            parse_config({**MINIMAL, "model": {"name": "m", "sequence_length": 0}})


class TestDerivedProperties:
    def test_full_input_shape_prepends_batch(self) -> None:
        config = parse_config(
            {**MINIMAL, "batch_size": 8, "model": {"name": "m", "input_shape": [3, 224, 224]}}
        )
        assert config.full_input_shape == [8, 3, 224, 224]

    def test_full_input_shape_is_none_without_shape(self) -> None:
        assert parse_config(MINIMAL).full_input_shape is None

    def test_measures_steady_state(self) -> None:
        assert parse_config(MINIMAL).measures_steady_state is True


class TestImmutabilityAndSerialisation:
    def test_config_is_frozen(self) -> None:
        """A config stored with a result must match what actually ran."""
        config = parse_config(MINIMAL)
        with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
            config.batch_size = 99  # type: ignore[misc]

    def test_round_trips(self) -> None:
        config = ExperimentConfig(
            name="exp",
            model=ModelConfig(name="m", input_shape=[3, 224, 224], revision="abc"),
            backend="fake",
            precision=Precision.FP16,
            batch_size=4,
            benchmark=BenchmarkConfig(warmup_iterations=3, measurement_iterations=20),
        )
        restored = ExperimentConfig.model_validate_json(config.model_dump_json())
        assert restored == config


class TestYamlLoading:
    def test_loads_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "exp.yaml"
        path.write_text(
            "name: resnet-fake-fp16\n"
            "backend: fake\n"
            "precision: fp16\n"
            "batch_size: 4\n"
            "model:\n"
            "  name: sim-model\n"
            "  input_shape: [3, 224, 224]\n"
            "benchmark:\n"
            "  warmup_iterations: 5\n"
            "  measurement_iterations: 50\n",
            encoding="utf-8",
        )
        config = load_config(path)
        assert config.name == "resnet-fake-fp16"
        assert config.precision is Precision.FP16
        assert config.batch_size == 4
        assert config.benchmark.measurement_iterations == 50

    def test_accepts_optional_experiment_wrapper(self, tmp_path: Path) -> None:
        path = tmp_path / "wrapped.yaml"
        path.write_text(
            "experiment:\n  name: wrapped\n  backend: fake\n  model:\n    name: m\n",
            encoding="utf-8",
        )
        assert load_config(path).name == "wrapped"

    def test_missing_file_is_a_clear_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_empty_file_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="empty"):
            load_config(path)

    def test_malformed_yaml_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("name: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Could not parse YAML"):
            load_config(path)

    def test_non_mapping_toplevel_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="mapping"):
            load_config(path)

    def test_error_names_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "invalid.yaml"
        path.write_text("name: x\nbackend: fake\nmodel:\n  name: m\nbatch_size: -5\n", "utf-8")
        with pytest.raises(ConfigurationError, match=r"invalid\.yaml"):
            load_config(path)
