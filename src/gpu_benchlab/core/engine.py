"""The benchmark engine.

Owns the phase structure and the measurement loop; owns no runtime-specific
knowledge. Synchronization and timing mechanism belong to the backend
(see :mod:`gpu_benchlab.core.timing`).

Guarantees this engine makes:

* **Every experiment produces a result.** Failures, OOMs and unsupported
  configurations become results with a status and a reason, never exceptions
  that escape and abort a suite (PRD §33).
* **Phases never leak into each other.** Model load, engine build, input
  preparation, warmup and steady-state measurement are timed separately.
* **Raw samples are always preserved**, including warmup samples, which are
  excluded from statistics but retained so warmup convergence can be analysed.
* **The measurement loop is bare.** No logging, no allocation, no telemetry, no
  branching that could be hoisted out. Anything else would distort what is being
  measured (PRD §36).
* **Simulation is contagious.** If the backend is simulated, the result is
  stamped simulated and carries a warning note.
"""

from __future__ import annotations

import traceback
import uuid
from collections.abc import Callable

from gpu_benchlab.core.backend import Backend, DeviceKind
from gpu_benchlab.core.config import ExperimentConfig
from gpu_benchlab.core.errors import (
    InvalidSampleError,
    UnavailableError,
    UnsupportedConfigurationError,
)
from gpu_benchlab.core.provenance import capture_provenance
from gpu_benchlab.core.schema import (
    BenchmarkResult,
    BenchmarkStatus,
    ErrorRecord,
    PhaseTimings,
    RawSamples,
    ThroughputMetric,
    ThroughputUnit,
    utc_now_iso,
)
from gpu_benchlab.core.statistics import LatencyStatistics
from gpu_benchlab.core.timing import Timer, TimingMechanism, WallClockTimer
from gpu_benchlab.hardware.detect import detect_environment
from gpu_benchlab.hardware.types import EnvironmentReport

__all__ = ["BenchmarkEngine"]

SIMULATION_WARNING = (
    "SIMULATED RESULT: produced by a simulated backend. These numbers exercise "
    "the benchmarking framework itself and are NOT measurements of any GPU, "
    "model or inference runtime. They must never be presented as performance data."
)

CPU_RESULT_NOTE = (
    "CPU RESULT: measured on the host CPU. This is a real measurement of CPU "
    "inference, NOT GPU performance, and must never be compared with a GPU result "
    "as a speedup or presented as NVIDIA GPU data."
)

NO_WARMUP_WARNING = (
    "warmup_iterations is 0, so these samples include cold-start effects "
    "(kernel autotuning, lazy allocation, clock ramp). This is NOT steady-state "
    "performance."
)

_MS_PER_SECOND = 1000.0


class BenchmarkEngine:
    """Runs one experiment against one backend and returns a complete result.

    Args:
        environment: Pre-detected environment. Detection is slow, so a suite
            runner detects once and passes it in. Detected on demand if omitted.
        experiment_id_factory: Overridable for deterministic tests.
    """

    def __init__(
        self,
        environment: EnvironmentReport | None = None,
        experiment_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._environment = environment
        self._new_id = experiment_id_factory or (lambda: uuid.uuid4().hex[:12])

    @property
    def environment(self) -> EnvironmentReport:
        if self._environment is None:
            self._environment = detect_environment(include_frameworks=True)
        return self._environment

    # -- public API --------------------------------------------------------------------

    def run(
        self,
        backend: Backend,
        config: ExperimentConfig,
        *,
        group_id: str | None = None,
        repeat_index: int = 0,
    ) -> BenchmarkResult:
        """Execute one experiment. Never raises.

        Every error is recorded with the exact lifecycle phase it came from
        (validate / load / build / prepare / warmup / measure / statistics / close).
        """
        experiment_id = self._new_id()
        phases = PhaseTimings()
        raw = RawSamples()
        errors: list[ErrorRecord] = []

        # None until a timer exists: a run that never timed anything must not
        # claim a timing mechanism.
        timing_mechanism: TimingMechanism | None = None
        secondary_mechanism: TimingMechanism | None = None
        status = BenchmarkStatus.OK
        latency: LatencyStatistics | None = None
        secondary_latency: LatencyStatistics | None = None
        throughput: list[ThroughputMetric] = []

        phase = "validate"
        try:
            backend.validate(config, self.environment)

            # Setup phases are one-off costs: a host clock plus the backend's own
            # synchronization is the right instrument for them.
            setup_timer = WallClockTimer(synchronize=backend.synchronize)

            phase = "load"
            setup_timer.start()
            backend.load()
            model_load_ms = setup_timer.stop()

            phase = "build"
            setup_timer.start()
            backend.build()
            engine_build_ms = setup_timer.stop()

            phase = "prepare"
            setup_timer.start()
            inputs = backend.prepare(config)
            prepare_ms = setup_timer.stop()

            timer = backend.make_timer()
            timing_mechanism = timer.mechanism

            with backend.execution_context():
                phase = "warmup"
                warmup_samples, warmup_total_ms = self._warmup(backend, timer, inputs, config)
                warmup_secondary = timer.drain_secondary()

                phase = "measure"
                measured, measurement_total_ms = self._measure(backend, timer, inputs, config)
                measured_secondary = timer.drain_secondary()

            phases = PhaseTimings(
                model_load_ms=model_load_ms,
                engine_build_ms=engine_build_ms,
                prepare_inputs_ms=prepare_ms,
                warmup_total_ms=warmup_total_ms,
                measurement_total_ms=measurement_total_ms,
            )
            raw = RawSamples(
                latency_ms=measured,
                warmup_latency_ms=warmup_samples,
                secondary_latency_ms=measured_secondary[1] if measured_secondary else [],
                secondary_warmup_latency_ms=warmup_secondary[1] if warmup_secondary else [],
            )

            phase = "statistics"
            latency = LatencyStatistics.from_samples(measured)
            if measured_secondary is not None:
                secondary_mechanism = measured_secondary[0]
                secondary_latency = LatencyStatistics.from_samples(measured_secondary[1])
            throughput = self._throughput(latency, measurement_total_ms, config, timing_mechanism)

        except UnsupportedConfigurationError as exc:
            status = BenchmarkStatus.UNSUPPORTED
            errors.append(self._error_record(exc, phase))
        except UnavailableError as exc:
            # A missing device or runtime. Nothing ran, nothing broke, and nothing
            # is substituted: a CUDA request is never quietly run on the CPU.
            status = BenchmarkStatus.UNAVAILABLE
            errors.append(self._error_record(exc, phase))
        except InvalidSampleError as exc:
            # Timing data is untrustworthy, so no statistics may be reported from it.
            status = BenchmarkStatus.FAILED
            errors.append(self._error_record(exc, phase))
            latency = None
            secondary_latency = None
        except Exception as exc:  # noqa: BLE001 - any backend failure becomes a result
            # Includes OOM (ours and torch.OutOfMemoryError): a real finding about
            # this configuration, recorded with the phase it happened in.
            status = BenchmarkStatus.FAILED
            errors.append(self._error_record(exc, phase))
        finally:
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(self._error_record(exc, "close"))

        # Read after the run: effective settings only exist once applied.
        descriptor = backend.descriptor
        device_kind = descriptor.device_kind
        if descriptor.is_simulated:
            device_kind = DeviceKind.SIMULATED

        notes: list[str] = []
        if descriptor.is_simulated:
            notes.append(SIMULATION_WARNING)
        if device_kind is DeviceKind.CPU:
            notes.append(CPU_RESULT_NOTE)
        if config.benchmark.warmup_iterations == 0:
            notes.append(NO_WARMUP_WARNING)

        return BenchmarkResult(
            experiment_id=experiment_id,
            group_id=group_id,
            repeat_index=repeat_index,
            timestamp_utc=utc_now_iso(),
            status=status,
            is_simulated=descriptor.is_simulated,
            device_kind=device_kind,
            configuration=config,
            backend=descriptor,
            model_info=backend.model_info,
            environment=self.environment,
            timing_mechanism=timing_mechanism,
            secondary_timing_mechanism=secondary_mechanism,
            measures_steady_state=config.measures_steady_state,
            phases=phases,
            latency=latency,
            secondary_latency=secondary_latency,
            throughput=throughput,
            raw_samples=raw,
            provenance=capture_provenance(),
            errors=errors,
            notes=notes,
        )

    # -- phases ------------------------------------------------------------------------

    @staticmethod
    def _warmup(
        backend: Backend, timer: Timer, inputs: object, config: ExperimentConfig
    ) -> tuple[list[float], float]:
        """Run and time warmup iterations.

        Warmup durations are recorded but excluded from every reported statistic.
        Keeping them makes it possible to determine the correct warmup count
        empirically instead of trusting the default.
        """
        count = config.benchmark.warmup_iterations
        if count == 0:
            return [], 0.0

        samples: list[float] = []
        execute = backend.execute
        start = timer.start
        stop = timer.stop

        total = WallClockTimer()
        total.start()
        for _ in range(count):
            start()
            execute(inputs)
            samples.append(stop())
        total_ms = total.stop()

        return samples, total_ms

    @staticmethod
    def _measure(
        backend: Backend, timer: Timer, inputs: object, config: ExperimentConfig
    ) -> tuple[list[float], float]:
        """The measurement loop.

        Deliberately bare: locals are bound before the loop, the list is
        pre-sized, and nothing else happens inside it. Every additional
        operation here is overhead attributed to the model under test.
        """
        count = config.benchmark.measurement_iterations
        samples: list[float] = [0.0] * count

        execute = backend.execute
        start = timer.start
        stop = timer.stop

        total = WallClockTimer()
        total.start()
        for i in range(count):
            start()
            execute(inputs)
            samples[i] = stop()
        total_ms = total.stop()

        return samples, total_ms

    # -- derived metrics ---------------------------------------------------------------

    @staticmethod
    def _throughput(
        latency: LatencyStatistics,
        measurement_total_ms: float,
        config: ExperimentConfig,
        mechanism: TimingMechanism,
    ) -> list[ThroughputMetric]:
        """Derive throughput from measured latency.

        Two figures are meaningful and they answer different questions, so both
        are reported with the formula that produced them rather than one being
        picked and called "throughput".
        """
        metrics: list[ThroughputMetric] = []
        batch = config.batch_size

        if latency.mean_ms > 0:
            metrics.append(
                ThroughputMetric(
                    value=batch / (latency.mean_ms / _MS_PER_SECOND),
                    unit=ThroughputUnit.SAMPLES_PER_SECOND,
                    basis="batch_size / mean_latency_s",
                )
            )

        # Observed throughput divides a real wall-clock window by the work done in
        # it. Under a scripted timer the per-iteration durations are fabricated
        # while the loop's wall time is real, so the two are not commensurable and
        # the quotient is meaningless. Emitting nothing is the honest outcome;
        # emitting a number we cannot justify is exactly what this project forbids.
        if measurement_total_ms > 0 and mechanism is not TimingMechanism.SCRIPTED:
            total_samples = config.benchmark.measurement_iterations * batch
            metrics.append(
                ThroughputMetric(
                    value=total_samples / (measurement_total_ms / _MS_PER_SECOND),
                    unit=ThroughputUnit.SAMPLES_PER_SECOND,
                    basis="measurement_iterations * batch_size / measurement_total_s",
                )
            )

        return metrics

    # -- errors ------------------------------------------------------------------------

    @staticmethod
    def _error_record(exc: BaseException, phase: str) -> ErrorRecord:
        return ErrorRecord(
            type=type(exc).__name__,
            message=str(exc),
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            phase=phase,
            timestamp_utc=utc_now_iso(),
        )
