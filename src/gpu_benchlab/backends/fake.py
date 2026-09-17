"""Deterministic simulated backend, for testing the framework only.

.. warning::

   **This backend measures nothing.** It returns scripted durations from a seeded
   generator. It exists so the engine, statistics, schema, storage and failure
   paths can be tested exhaustively, instantly, and on hardware with no GPU.

   Every result it produces is stamped ``is_simulated=True``, carries a warning
   note, uses :attr:`TimingMechanism.SCRIPTED`, and is stored under a ``sim-``
   prefixed directory. Four independent markers, because the one thing this
   project must never do is let a fabricated number be mistaken for a measurement.

The simulated latency model is deliberately crude but has the shape real
measurements have, so the framework is exercised against realistic input:

* a **warmup ramp** — the first iterations are slower and decay towards steady
  state, so warmup logic and warmup-convergence analysis have something to bite on;
* **Gaussian jitter** around a steady-state mean;
* **occasional outliers**, so percentile and tail behaviour is exercised;
* configurable **failure injection** — OOM above a batch-size threshold,
  unsupported precisions, load/build/execute errors.

None of these values correspond to any real hardware.
"""

from __future__ import annotations

import random
import time
from typing import Any

from gpu_benchlab.core.backend import Backend, BackendDescriptor
from gpu_benchlab.core.config import ExperimentConfig
from gpu_benchlab.core.errors import (
    BackendError,
    OutOfMemoryError,
    UnsupportedConfigurationError,
)
from gpu_benchlab.core.timing import Timer, TimingMechanism
from gpu_benchlab.hardware.capability import Precision

__all__ = ["FakeBackend", "FakeTimer"]


class FakeTimer(Timer):
    """Returns simulated durations pulled from the backend's generator.

    Reports :attr:`TimingMechanism.SCRIPTED` so nothing downstream can mistake
    these for measurements.
    """

    def __init__(self, backend: FakeBackend) -> None:
        self._backend = backend
        self._started = False
        self.mechanism = TimingMechanism.SCRIPTED

    def start(self) -> None:
        self._started = True

    def stop(self) -> float:
        if not self._started:
            raise RuntimeError("Timer.stop() called before start().")
        self._started = False
        return self._backend.next_duration_ms()


class FakeBackend(Backend):
    """A simulated inference backend with reproducible, configurable behaviour.

    Args:
        seed: Seeds the generator. The same seed and parameters always produce
            exactly the same sample sequence.
        base_latency_ms: Simulated steady-state mean at batch size 1.
        jitter_ms: Standard deviation of the Gaussian noise.
        per_sample_latency_ms: Added per batch element, so batch scaling is not
            perfectly flat.
        warmup_penalty_ms: Extra latency on the first iteration, decaying
            geometrically by ``warmup_decay`` each iteration.
        warmup_decay: Decay factor for the warmup penalty, in (0, 1).
        outlier_probability: Chance per iteration of an outlier.
        outlier_multiplier: How much slower an outlier is.
        model_load_ms / engine_build_ms: Simulated setup costs. These really do
            elapse, because the engine times setup phases with a real clock.
        oom_above_batch_size: Raise OOM when batch size exceeds this.
        unsupported_precisions: Precisions to reject at validation time.
        fail_on_load / fail_on_build / fail_on_execute_iteration: Failure injection.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        base_latency_ms: float = 10.0,
        jitter_ms: float = 0.5,
        per_sample_latency_ms: float = 1.5,
        warmup_penalty_ms: float = 40.0,
        warmup_decay: float = 0.5,
        outlier_probability: float = 0.0,
        outlier_multiplier: float = 5.0,
        model_load_ms: float = 0.0,
        engine_build_ms: float = 0.0,
        oom_above_batch_size: int | None = None,
        unsupported_precisions: frozenset[Precision] | None = None,
        fail_on_load: bool = False,
        fail_on_build: bool = False,
        fail_on_execute_iteration: int | None = None,
        emit_nan_at_iteration: int | None = None,
        name: str = "fake",
    ) -> None:
        if not 0.0 < warmup_decay < 1.0:
            raise ValueError(f"warmup_decay must be in (0, 1), got {warmup_decay}")

        self._seed = seed
        self._rng = random.Random(seed)
        self._base_latency_ms = base_latency_ms
        self._jitter_ms = jitter_ms
        self._per_sample_latency_ms = per_sample_latency_ms
        self._warmup_penalty_ms = warmup_penalty_ms
        self._warmup_decay = warmup_decay
        self._outlier_probability = outlier_probability
        self._outlier_multiplier = outlier_multiplier
        self._model_load_ms = model_load_ms
        self._engine_build_ms = engine_build_ms
        self._oom_above_batch_size = oom_above_batch_size
        self._unsupported = unsupported_precisions or frozenset()
        self._fail_on_load = fail_on_load
        self._fail_on_build = fail_on_build
        self._fail_on_execute_iteration = fail_on_execute_iteration
        self._emit_nan_at_iteration = emit_nan_at_iteration
        self._name = name

        self._batch_size = 1
        self._iteration = 0
        self._executions = 0
        self.closed = False

    # -- identity ----------------------------------------------------------------------

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            name=self._name,
            version="simulated",
            device="simulated",
            is_simulated=True,
            detail=(
                "Simulated backend. Produces scripted durations from a seeded "
                f"generator (seed={self._seed}); measures no real hardware."
            ),
        )

    # -- lifecycle ---------------------------------------------------------------------

    def validate(self, config: Any) -> None:
        assert isinstance(config, ExperimentConfig)

        if config.precision in self._unsupported:
            raise UnsupportedConfigurationError(
                f"Simulated backend is configured to reject precision {config.precision.value}."
            )
        if (
            self._oom_above_batch_size is not None
            and config.batch_size > self._oom_above_batch_size
        ):
            # Deliberately NOT raised here: an OOM is discovered during execution,
            # and the engine must record it as a failure rather than as unsupported.
            return

    def load(self) -> None:
        if self._fail_on_load:
            raise BackendError("Simulated model load failure.")
        self._consume(self._model_load_ms)

    def build(self) -> None:
        if self._fail_on_build:
            raise BackendError("Simulated engine build failure.")
        self._consume(self._engine_build_ms)

    @staticmethod
    def _consume(duration_ms: float) -> None:
        """Actually spend the requested wall time.

        Setup phases are timed by the engine with a *real* clock (unlike the
        scripted per-iteration timer), so simulating an expensive model load or
        engine build requires really elapsing time. Kept to a few milliseconds
        in tests. This is the one place the fake backend is genuinely slow, and
        it exists so the "engine build must never leak into inference latency"
        guarantee can be tested against a build that actually costs something.
        """
        if duration_ms <= 0:
            return
        time.sleep(duration_ms / 1000.0)

    def prepare(self, config: Any) -> Any:
        assert isinstance(config, ExperimentConfig)
        self._batch_size = config.batch_size
        self._iteration = 0

        if (
            self._oom_above_batch_size is not None
            and config.batch_size > self._oom_above_batch_size
        ):
            raise OutOfMemoryError(
                f"Simulated CUDA out of memory at batch size {config.batch_size} "
                f"(simulated limit: {self._oom_above_batch_size})."
            )

        return {"batch_size": config.batch_size, "shape": config.full_input_shape}

    def execute(self, inputs: Any) -> Any:
        self._executions += 1
        if (
            self._fail_on_execute_iteration is not None
            and self._executions > self._fail_on_execute_iteration
        ):
            raise BackendError(f"Simulated execution failure at iteration {self._executions}.")
        return inputs

    def make_timer(self) -> Timer:
        return FakeTimer(self)

    def close(self) -> None:
        self.closed = True

    # -- simulated latency model -------------------------------------------------------

    def next_duration_ms(self) -> float:
        """Produce the next simulated duration. Not a measurement."""
        index = self._iteration
        self._iteration += 1

        if self._emit_nan_at_iteration is not None and index == self._emit_nan_at_iteration:
            return float("nan")

        steady = self._base_latency_ms + self._per_sample_latency_ms * (self._batch_size - 1)
        warmup_extra = self._warmup_penalty_ms * (self._warmup_decay**index)
        # S311: reproducibility is the requirement here, not unpredictability.
        # A seeded Mersenne Twister is exactly right; this is never security-related.
        jitter = self._rng.gauss(0.0, self._jitter_ms) if self._jitter_ms > 0 else 0.0

        duration = steady + warmup_extra + jitter

        if self._outlier_probability > 0 and self._rng.random() < self._outlier_probability:
            duration *= self._outlier_multiplier

        # A simulated duration must still obey the contract that durations are
        # non-negative, or it would trip validation for the wrong reason.
        return max(duration, 0.0)

    @property
    def execution_count(self) -> int:
        """Total execute() calls, including warmup. For test assertions."""
        return self._executions

    def reset(self) -> None:
        """Restore the generator to its initial state."""
        self._rng = random.Random(self._seed)
        self._iteration = 0
        self._executions = 0
        self.closed = False
