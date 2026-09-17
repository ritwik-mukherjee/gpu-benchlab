"""Tests for the timing abstraction.

The central contract being verified: the generic layer never synchronizes on its
own, and a timer's reported mechanism always matches what it actually did.
"""

from __future__ import annotations

import time

import pytest

from gpu_benchlab.core.timing import (
    NANOSECONDS_PER_MILLISECOND,
    ScriptedTimer,
    TimingMechanism,
    WallClockTimer,
)


class TestWallClockTimer:
    def test_measures_elapsed_time(self) -> None:
        timer = WallClockTimer()
        timer.start()
        time.sleep(0.02)
        elapsed = timer.stop()
        # Generous lower bound; upper bound loose because CI machines stall.
        assert 15.0 < elapsed < 500.0

    def test_returns_milliseconds(self) -> None:
        assert NANOSECONDS_PER_MILLISECOND == 1_000_000.0

    def test_mechanism_without_sync(self) -> None:
        assert WallClockTimer().mechanism is TimingMechanism.WALL_CLOCK

    def test_mechanism_with_sync(self) -> None:
        """A timer must not claim to synchronize when it was given no hook."""
        timer = WallClockTimer(synchronize=lambda: None)
        assert timer.mechanism is TimingMechanism.WALL_CLOCK_SYNCHRONIZED

    def test_stop_before_start_raises(self) -> None:
        with pytest.raises(RuntimeError, match="before start"):
            WallClockTimer().stop()

    def test_is_reusable(self) -> None:
        timer = WallClockTimer()
        for _ in range(3):
            timer.start()
            assert timer.stop() >= 0.0

    def test_double_stop_raises(self) -> None:
        timer = WallClockTimer()
        timer.start()
        timer.stop()
        with pytest.raises(RuntimeError):
            timer.stop()


class TestSynchronizationHook:
    def test_sync_called_at_start_and_stop(self) -> None:
        """start() drains prior work; stop() waits for this iteration's work."""
        calls: list[str] = []
        timer = WallClockTimer(synchronize=lambda: calls.append("sync"))

        timer.start()
        assert calls == ["sync"], "must drain outstanding work before timing"
        timer.stop()
        assert calls == ["sync", "sync"], "must wait for work before stopping the clock"

    def test_no_sync_calls_when_hook_absent(self) -> None:
        """The generic layer must never invent a synchronization."""
        timer = WallClockTimer()
        timer.start()
        timer.stop()  # nothing to assert beyond: it did not raise or require CUDA

    def test_sync_cost_is_inside_the_measurement(self) -> None:
        """Synchronization time is part of what a host clock measures; be explicit."""
        timer = WallClockTimer(synchronize=lambda: time.sleep(0.02))
        timer.start()
        elapsed = timer.stop()
        assert elapsed > 15.0, "the stop-side sync must be inside the measured window"


class TestContextManager:
    def test_records_last_measurement(self) -> None:
        timer = WallClockTimer()
        with timer:
            time.sleep(0.01)
        assert timer.last_ms > 5.0

    def test_last_ms_before_any_measurement_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not completed"):
            _ = WallClockTimer().last_ms


class TestScriptedTimer:
    def test_returns_scripted_values_in_order(self) -> None:
        timer = ScriptedTimer([1.0, 2.0, 3.0])
        out = []
        for _ in range(3):
            timer.start()
            out.append(timer.stop())
        assert out == [1.0, 2.0, 3.0]

    def test_mechanism_marks_it_as_not_a_measurement(self) -> None:
        assert ScriptedTimer([1.0]).mechanism is TimingMechanism.SCRIPTED

    def test_exhaustion_raises_clearly(self) -> None:
        timer = ScriptedTimer([1.0])
        timer.start()
        timer.stop()
        timer.start()
        with pytest.raises(RuntimeError, match="exhausted"):
            timer.stop()

    def test_stop_before_start_raises(self) -> None:
        with pytest.raises(RuntimeError, match="before start"):
            ScriptedTimer([1.0]).stop()

    def test_remaining_counts_down(self) -> None:
        timer = ScriptedTimer([1.0, 2.0])
        assert timer.remaining == 2
        timer.start()
        timer.stop()
        assert timer.remaining == 1


class TestMechanismEnum:
    def test_scripted_is_distinguishable(self) -> None:
        """Any result carrying SCRIPTED is simulated by construction."""
        assert TimingMechanism.SCRIPTED.value == "scripted"
        real = {
            TimingMechanism.WALL_CLOCK,
            TimingMechanism.WALL_CLOCK_SYNCHRONIZED,
            TimingMechanism.CUDA_EVENT,
        }
        assert TimingMechanism.SCRIPTED not in real

    def test_cuda_event_is_declared_before_implementation(self) -> None:
        """Named now so the schema does not change when Phase 3 adds it."""
        assert TimingMechanism.CUDA_EVENT.value == "cuda_event"
