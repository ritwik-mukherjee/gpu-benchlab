"""The contract every inference backend implements.

The engine drives this interface and owns the phase structure (load -> build ->
warmup -> measure). The backend owns everything runtime-specific, including
**its own timer**, because only the backend knows whether its work needs a device
synchronization, a stream synchronization, CUDA events, or nothing at all. See
:mod:`gpu_benchlab.core.timing` for why that decision does not live in the engine.

Concrete backends live in ``gpu_benchlab.backends``. This module deliberately
imports nothing from them, so the dependency runs backends -> core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gpu_benchlab.core.timing import Timer, WallClockTimer

__all__ = ["Backend", "BackendDescriptor"]


class BackendDescriptor(BaseModel):
    """Identity of the backend that produced a result.

    ``is_simulated`` is the load-bearing field: it is stamped onto every result
    and propagates to storage and to the CLI, so a result produced by a fake
    backend can never be mistaken for a hardware measurement.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    version: str | None = None
    execution_provider: str | None = Field(
        default=None,
        description=(
            "For runtimes with pluggable providers (ONNX Runtime), the provider "
            "that actually executed. Recorded so a TensorRT-EP result is never "
            "labelled as ONNX Runtime CUDA."
        ),
    )
    device: str | None = Field(default=None, description='e.g. "cuda:0", "cpu".')
    is_simulated: bool = Field(
        default=False,
        description=(
            "True when this backend does not execute a real model on real "
            "hardware. Simulated results are test fixtures for the framework "
            "itself and are never valid performance data."
        ),
    )
    detail: str | None = None


class Backend(ABC):
    """An inference runtime the engine can benchmark.

    Lifecycle, driven by :class:`~gpu_benchlab.core.engine.BenchmarkEngine`::

        backend.validate(config)     # raises UnsupportedConfigurationError if invalid
        backend.load()               # timed as model_load
        backend.build()              # timed as engine_build (may be a no-op)
        inputs = backend.prepare(config)
        timer = backend.make_timer()
        ... warmup iterations ...
        ... measured iterations ...
        backend.close()

    Implementations should raise
    :class:`~gpu_benchlab.core.errors.UnsupportedConfigurationError` from
    :meth:`validate` rather than failing later, and
    :class:`~gpu_benchlab.core.errors.OutOfMemoryError` from :meth:`execute` so an
    OOM is recorded as a measurement about that batch size rather than a crash.
    """

    @property
    @abstractmethod
    def descriptor(self) -> BackendDescriptor:
        """Identity and version of this backend."""

    def validate(self, config: Any) -> None:
        """Check the configuration can run here.

        Raises:
            UnsupportedConfigurationError: if it cannot, with a reason.

        The default accepts everything. Backends override to check precision
        support, batch-size limits and model compatibility *before* execution,
        so an invalid combination produces a clear ``unsupported`` result rather
        than a confusing runtime error (PRD §32).
        """

    def load(self) -> None:
        """Load the model and construct the runtime object.

        Timed separately as the model-load phase. Default is a no-op.
        """

    def build(self) -> None:
        """Compile or build an execution engine, if the runtime has such a step.

        Timed separately as the engine-build phase. This exists so a TensorRT
        engine build -- which can take minutes -- can never leak into inference
        latency (PRD §29). Default is a no-op.
        """

    @abstractmethod
    def prepare(self, config: Any) -> Any:
        """Build the input batch and move it to the device.

        Called once, before warmup. Doing this outside the measurement loop keeps
        host-to-device transfer out of the reported inference latency unless the
        experiment explicitly asks for end-to-end measurement.

        Returns:
            An opaque handle passed to every :meth:`execute` call.
        """

    @abstractmethod
    def execute(self, inputs: Any) -> Any:
        """Run one inference iteration.

        Must contain only the work being measured: no logging, no allocation, no
        telemetry. May return asynchronously -- the timer handles synchronization.
        """

    def synchronize(self) -> None:
        """Block until all outstanding device work has completed.

        Default is a no-op, which is correct for CPU backends and for runtimes
        that synchronize internally before returning.
        """

    def make_timer(self) -> Timer:
        """Return a timer appropriate for this backend's execution model.

        The default is a host monotonic clock that calls :meth:`synchronize`
        before stopping -- correct for any backend that overrides
        :meth:`synchronize`, and correct for CPU backends that do not.

        A backend with access to device-side instrumentation (CUDA events) should
        override this to return a more accurate timer. The mechanism is recorded
        on the result either way.
        """
        return WallClockTimer(synchronize=self.synchronize)

    def close(self) -> None:
        """Release resources. Always called, including after a failure."""

    def __enter__(self) -> Backend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
