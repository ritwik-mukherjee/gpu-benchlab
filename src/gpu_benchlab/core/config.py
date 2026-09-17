"""Experiment configuration and its validation.

Configurations are validated *before* execution (PRD §21). A malformed config is
a user error surfaced immediately, not a benchmark outcome — unlike a config that
is well-formed but unsupported by the hardware, which is a recorded result.

The defaults for ``warmup_iterations`` and ``measurement_iterations`` are
documented in :mod:`docs/methodology.md` §4 and are an explicitly **unvalidated
starting hypothesis**, not a measured constant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from gpu_benchlab.core.errors import ConfigurationError
from gpu_benchlab.hardware.capability import Precision

__all__ = [
    "DEFAULT_MEASUREMENT_ITERATIONS",
    "DEFAULT_WARMUP_ITERATIONS",
    "BenchmarkConfig",
    "ExperimentConfig",
    "ModelConfig",
    "load_config",
]

# See docs/methodology.md §4. These are a starting hypothesis to be replaced with
# an empirically determined value once the engine can measure warmup convergence.
DEFAULT_WARMUP_ITERATIONS = 10
DEFAULT_MEASUREMENT_ITERATIONS = 100


class ModelConfig(BaseModel):
    """Which model is being benchmarked, precisely enough to reproduce it."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    name: str = Field(min_length=1)
    source: str | None = Field(
        default=None, description="Where the model comes from, e.g. a hub id or a path."
    )
    revision: str | None = Field(
        default=None,
        description="Exact revision/commit of the model. Required for real reproducibility.",
    )
    format: str | None = Field(default=None, description='e.g. "pytorch", "onnx", "trt-engine".')
    input_shape: list[int] | None = Field(
        default=None,
        description="Per-sample input shape, excluding the batch dimension.",
    )
    sequence_length: int | None = Field(
        default=None, gt=0, description="For sequence models; None for vision models."
    )

    @field_validator("input_shape")
    @classmethod
    def _shape_must_be_positive(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("input_shape must not be empty; use null if not applicable.")
        if any(d <= 0 for d in value):
            raise ValueError(f"input_shape dimensions must all be positive, got {value}.")
        return value


class BenchmarkConfig(BaseModel):
    """Measurement parameters: how the benchmark is run, not what is run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    warmup_iterations: int = Field(
        default=DEFAULT_WARMUP_ITERATIONS,
        ge=0,
        description=(
            "Iterations executed and discarded before measurement. Zero is "
            "permitted but means steady-state performance is NOT being measured; "
            "this is recorded and flagged on the result."
        ),
    )
    measurement_iterations: int = Field(
        default=DEFAULT_MEASUREMENT_ITERATIONS,
        gt=0,
        description="Iterations whose individual durations are recorded.",
    )
    repeats: int = Field(
        default=1,
        gt=0,
        description=(
            "How many times to run the whole measurement. Each repeat is stored "
            "as its own result sharing a group id; repeats are never averaged "
            "together, because that would destroy the between-run variance."
        ),
    )


class ExperimentConfig(BaseModel):
    """A complete, self-contained description of one benchmark experiment.

    Stored verbatim alongside the result so the experiment can be reproduced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    name: str = Field(min_length=1)
    description: str | None = None

    model: ModelConfig
    backend: str = Field(min_length=1, description='Backend name, e.g. "pytorch", "tensorrt".')
    precision: Precision = Precision.FP32
    batch_size: int = Field(default=1, gt=0)
    device: str | None = Field(default=None, description='e.g. "cuda:0". None = backend default.')

    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    @property
    def measures_steady_state(self) -> bool:
        """False when warmup is disabled, in which case results include cold start."""
        return self.benchmark.warmup_iterations > 0

    @property
    def full_input_shape(self) -> list[int] | None:
        """Input shape including the batch dimension."""
        if self.model.input_shape is None:
            return None
        return [self.batch_size, *self.model.input_shape]


def _format_validation_error(exc: ValidationError, source: str) -> str:
    lines = [f"Invalid experiment configuration in {source}:"]
    for err in exc.errors():
        location = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {location}: {err['msg']}")
    return "\n".join(lines)


def parse_config(data: dict[str, Any], *, source: str = "<dict>") -> ExperimentConfig:
    """Validate a config mapping.

    Raises:
        ConfigurationError: with every validation problem listed, not just the first.
    """
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(_format_validation_error(exc, source)) from exc


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file.

    Raises:
        ConfigurationError: if the file is missing, unparseable, or invalid.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Could not parse YAML in {config_path}: {exc}") from exc

    if raw is None:
        raise ConfigurationError(f"Configuration file is empty: {config_path}")
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Configuration file must contain a mapping at the top level, "
            f"got {type(raw).__name__}: {config_path}"
        )

    # Accept an optional `experiment:` wrapper so configs read naturally either way.
    if "experiment" in raw and isinstance(raw["experiment"], dict):
        raw = raw["experiment"]

    return parse_config(raw, source=str(config_path))
