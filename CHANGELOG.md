# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Phase 2: benchmark core

- **Timing abstraction** with an explicit contract. The engine never synchronizes;
  each backend supplies its own timer, because only it knows whether a device sync,
  a stream sync or CUDA events is correct (ADR 0004). `WallClockTimer` uses
  `perf_counter_ns` with an injectable synchronization hook. The mechanism used is
  recorded on every result.
- **Benchmark engine** with strict phase separation — model load, engine build,
  input preparation, warmup and steady-state measurement are timed independently, so
  an engine build can never leak into inference latency.
- **Latency statistics** from raw samples: min/max/mean/median/p50/p90/p95/p99/stddev.
  Percentile method and `ddof` are recorded. Percentiles from too few samples to be
  stable are flagged rather than presented as solid.
- **Invalid samples are rejected, never dropped.** Empty sets, NaN, infinity and
  negative durations fail the run instead of silently altering the statistics.
- **Raw sample preservation**, including warmup samples, which are excluded from
  statistics but retained so warmup convergence can be determined empirically.
- **Versioned result schema** (`schema_version` 1.0) embedding configuration, full
  environment, backend identity, phases, statistics, raw samples, throughput,
  provenance (git commit + dirty flag) and errors.
- **Errors as first-class outcomes.** OOM, unsupported precision, load/build/execute
  failures and invalid samples all produce a stored result with a status and reason;
  the engine never raises.
- **Throughput with its unit and formula named**, never a bare "throughput" number.
- **Validated YAML experiment configuration** — unknown fields rejected, all errors
  reported at once.
- **JSON result storage**: `result.json` (canonical) plus `metadata.json`,
  `raw.json` and `summary.json` views written from the same object.
- **Simulated backend** for testing the framework without hardware. Deterministic
  under a seed, with warmup ramp, jitter, outliers and failure injection.
- `gpu-bench run --config <file>` to execute an experiment.

### Safety — simulated results cannot pass as measurements

The simulated backend marks its output in five independent places: `is_simulated`
on the result and on the backend descriptor, `timing_mechanism: scripted`, a warning
note travelling with the result, a `sim-` prefixed storage directory, and the flag
repeated in all four stored JSON files. `BenchmarkResult.is_real_measurement` is the
single check consumers should use.

Observed-throughput is deliberately **not** emitted under scripted timing, because
dividing real wall time by fabricated per-iteration durations would produce a number
nothing supports.

### Changed

- Enabled `ruff` bandit (`S`) security rules across the source tree.
- mypy now targets 3.12 rather than 3.10 (numpy's stubs use PEP 695 syntax mypy
  cannot parse under an older target). Runtime support for 3.10 is verified by
  running the full suite on 3.10 in CI.

### Added — Phase 1: hardware detection

- NVIDIA GPU detection via NVML (`nvidia-ml-py`), reporting name, UUID, PCI bus,
  architecture, compute capability, VRAM, clocks, power, temperature and compute mode.
- Four distinct detection states — `ok`, `no_nvidia_device`, `driver_unavailable`,
  `library_unavailable` — plus `error`, because each needs different user action.
- Per-precision support matrix (FP32/TF32/FP16/BF16/INT8/FP8/INT4/FP4) derived from
  compute capability, distinguishing *supported* from *tensor-core accelerated*.
- Host detection: OS, CPU model, core counts, RAM, Python.
- Framework detection for PyTorch, ONNX Runtime, TensorRT and TensorRT-LLM,
  separating "installed" from "has a usable CUDA device", and recording ONNX Runtime
  execution providers.
- Versioned, immutable `EnvironmentReport` schema (`schema_version` 1.0) that
  round-trips through JSON.
- `gpu-bench hardware` (human and `--json` output, `-o` to write a file) and
  `gpu-bench doctor`, with meaningful exit codes: `0` GPU present, `1` no usable GPU,
  `2` detection failed.

### Added — Phase 0: project foundation

- Repository structure, Apache-2.0 license, README, CONTRIBUTING, CLAUDE.md.
- Architecture, methodology, roadmap, limitations and environment documentation.
- ADRs 0001 (JSON storage), 0002 (NVML binding), 0003 (CPU-installable core).
- CI on Ubuntu and Windows across Python 3.10 and 3.12, including a check that
  detection degrades honestly on GPU-less runners.
- Tooling: ruff (lint + format), mypy strict, pytest with GPU/backend markers.

### Notes

This release contains **no benchmark measurements**. No real inference backend exists
yet, and the development machine has no NVIDIA GPU. The only results produced so far
are simulated and marked as such. See [docs/limitations.md](docs/limitations.md) for
exactly what has and has not been executed on real hardware.
