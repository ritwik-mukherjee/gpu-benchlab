"""Timing abstraction.

## The contract

A :class:`Timer` measures the duration of **one iteration** of work. The engine
drives it::

    timer.start()
    backend.execute(inputs)
    sample_ms = timer.stop()

`stop()` must return the elapsed time in **milliseconds as a float**, and must not
return until the measured work has genuinely completed.

## Why the generic layer does not synchronize

CUDA kernel launches are asynchronous, so a host-side clock stopped immediately
after `execute()` measures launch overhead rather than execution. The obvious fix
is to call `torch.cuda.synchronize()` in the engine — but that would be wrong here
for three reasons:

1. **It is not always the correct call.** ONNX Runtime synchronizes internally
   before `run()` returns; an extra device-wide sync would add cost that is not
   part of the workload. TensorRT wants a *stream* synchronization, not a device
   one. A CPU backend needs none at all.
2. **It is not always the best available mechanism.** Where CUDA events exist they
   measure device time directly and exclude host-side launch and synchronization
   overhead. A host clock cannot do that, no matter where the sync goes.
3. **It would hard-code a CUDA dependency into code that must import on a machine
   with no GPU.**

So the engine never synchronizes. Instead, **each backend supplies its own timer**
via ``Backend.make_timer()``, and that timer encapsulates whatever
synchronization or device-side instrumentation is correct for that runtime.
The mechanism used is recorded on every result (:class:`TimingMechanism`) so two
results measured on different bases are never silently compared.

## What this module provides

:class:`WallClockTimer` — a monotonic host-clock timer with an injectable
synchronization hook. It covers the CPU case (no hook) and the
"synchronize then stop the host clock" case (hook supplied) without knowing
anything about CUDA.

A CUDA-event timer belongs with the backend that owns the stream, and arrives in
Phase 3 with the PyTorch backend. :class:`TimingMechanism` already names it so the
schema does not need to change when it does.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from types import TracebackType

__all__ = [
    "NANOSECONDS_PER_MILLISECOND",
    "ScriptedTimer",
    "Timer",
    "TimingMechanism",
    "WallClockTimer",
]

NANOSECONDS_PER_MILLISECOND = 1_000_000.0


class TimingMechanism(str, Enum):
    """How a duration was measured. Recorded on every result."""

    WALL_CLOCK = "wall_clock"
    """Host monotonic clock, no synchronization. Correct only for synchronous work."""

    WALL_CLOCK_SYNCHRONIZED = "wall_clock_synchronized"
    """Host monotonic clock with a backend-supplied synchronization before stopping.

    Includes kernel launch overhead and the cost of the synchronization itself.
    """

    CUDA_EVENT = "cuda_event"
    """CUDA events recorded on the execution stream — device time only.

    Excludes host-side launch overhead. Not implemented until Phase 3.
    """

    SCRIPTED = "scripted"
    """Durations supplied by a test fixture. Never a real measurement.

    Any result carrying this mechanism is simulated by construction.
    """


class Timer(ABC):
    """Measures the wall duration of a single unit of work.

    Implementations must be reusable: ``start()``/``stop()`` may be called many
    times in sequence, once per benchmark iteration.
    """

    mechanism: TimingMechanism

    @abstractmethod
    def start(self) -> None:
        """Begin timing. Any required pre-synchronization happens here."""

    @abstractmethod
    def stop(self) -> float:
        """End timing and return the elapsed duration in milliseconds.

        Must not return until the measured work has actually completed.
        """

    def __enter__(self) -> Timer:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._last_ms = self.stop()

    def drain_secondary(self) -> tuple[TimingMechanism, list[float]] | None:
        """Return and clear any secondary samples collected since the last drain.

        A timer may record a second, independent measurement of each iteration
        -- for example host wall time alongside CUDA-event device time. It
        accumulates those internally so the engine adds no per-iteration work,
        and the engine drains them once after warmup and once after measurement.

        Returns ``None`` when the timer records no secondary measurement.
        """
        return None

    @property
    def last_ms(self) -> float:
        """Duration recorded by the most recent context-manager exit."""
        value = getattr(self, "_last_ms", None)
        if value is None:
            raise RuntimeError("Timer has not completed a measurement yet.")
        return float(value)


class WallClockTimer(Timer):
    """Monotonic host-clock timer with an optional synchronization hook.

    Uses :func:`time.perf_counter_ns` — monotonic, highest available resolution,
    and immune to wall-clock adjustment. ``time.time()`` is never used: it is
    subject to NTP correction and can move backwards.

    Args:
        synchronize: Called immediately before the clock is stopped, and again at
            ``start()`` to ensure prior work has drained. Backends pass their own
            device synchronization here. ``None`` means the work is already
            synchronous (CPU backends, or runtimes that synchronize internally).

    The reported :attr:`mechanism` reflects whether a hook was supplied, so the
    result records the difference rather than claiming a synchronized measurement
    that never synchronized.
    """

    def __init__(self, synchronize: Callable[[], None] | None = None) -> None:
        self._synchronize = synchronize
        self._start_ns: int | None = None
        self.mechanism = (
            TimingMechanism.WALL_CLOCK
            if synchronize is None
            else TimingMechanism.WALL_CLOCK_SYNCHRONIZED
        )

    def start(self) -> None:
        # Drain outstanding work first, so a previous iteration's tail is not
        # attributed to this iteration.
        if self._synchronize is not None:
            self._synchronize()
        self._start_ns = time.perf_counter_ns()

    def stop(self) -> float:
        if self._start_ns is None:
            raise RuntimeError("Timer.stop() called before start().")
        if self._synchronize is not None:
            self._synchronize()
        elapsed_ns = time.perf_counter_ns() - self._start_ns
        self._start_ns = None
        return elapsed_ns / NANOSECONDS_PER_MILLISECOND


class ScriptedTimer(Timer):
    """Returns pre-determined durations instead of measuring anything.

    Exists so the engine's loop, phase separation and statistics can be tested
    deterministically and instantly, without sleeping and without hardware.

    **This never produces a measurement.** Its mechanism is
    :attr:`TimingMechanism.SCRIPTED`, which marks any result derived from it as
    simulated.

    Args:
        durations_ms: One value per expected ``stop()`` call.
    """

    def __init__(self, durations_ms: list[float]) -> None:
        self._durations = list(durations_ms)
        self._index = 0
        self._started = False
        self.mechanism = TimingMechanism.SCRIPTED

    def start(self) -> None:
        self._started = True

    def stop(self) -> float:
        if not self._started:
            raise RuntimeError("Timer.stop() called before start().")
        if self._index >= len(self._durations):
            raise RuntimeError(
                f"ScriptedTimer exhausted after {len(self._durations)} durations; "
                "the engine asked for more iterations than were scripted."
            )
        value = self._durations[self._index]
        self._index += 1
        self._started = False
        return value

    @property
    def remaining(self) -> int:
        return len(self._durations) - self._index
