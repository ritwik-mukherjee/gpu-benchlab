# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Phase 6: TensorRT backend (implemented, **not validated on any GPU**)

- **A third backend compiled from the same canonical ONNX artifact**, by SHA-256, that
  ONNX Runtime consumes — TensorRT gets no export path of its own.
  `export/tensorrt_build.py` (ONNX parser with every `get_error(i)` surfaced, builder
  config, one optimization profile spanning batch 1–8, engine cache keyed on a manifest,
  serialize/deserialize) and `backends/tensorrt_backend.py` (`TensorRtBackend`,
  `TensorRtOptions`, `TrtCudaEventTimer`).
- **FP32 means IEEE FP32.** TensorRT enables `BuilderFlag.TF32` by default, so an engine
  built without clearing it is TF32. The `ieee_fp32` policy clears the flag at build time
  and the engine manifest records `tf32_enabled_by_default`, `tf32_flag_after_policy`,
  `strict_nans`, the timing-cache state and `strongly_typed_network`.
- **The engine build is not inference.** Building happens in the `build` lifecycle phase
  as `engine_build_ms`; deserialization, context creation and buffer allocation happen in
  `prepare`. Input and output stay device-resident for the whole measured loop, the only
  host copy being the untimed sanity pass. Timing reuses the PyTorch contract (CUDA
  events primary, synchronized host time secondary) — no second timing methodology.
- A shape outside the engine's optimization profile is `unsupported`, never clamped; a
  missing CUDA device is `unavailable`, matching PyTorch and ONNX Runtime.
- `tests/unit/test_tensorrt.py` (37 test functions) and the correctness collector
  `analysis/gpu_validation/correctness_tensorrt.py`, plus the Phase 6 controlled configs.

**What this is not.** Every TensorRT test runs against a mocked `FakeTrt`/`FakeCudart`.
**No TensorRT library has been installed or executed by this project.** There is no
engine, no TensorRT measurement and no evidence directory. Phase 6A (the read-only
environment and library probe) and Phase 6G (the controlled benchmark) are outstanding,
and until 6A reports, the `[tensorrt]` pins are declared rather than verified.

### Changed — documentation reconciled with the repository (audit, 2026-09-20)

- **One phase number for TensorRT: Phase 6.** The commit messages, the plan filename and
  title, the planned evidence directory and the `p6-` experiment names already said 6,
  while README said "Phase 5" and the roadmap carried a duplicate "5B" row. In-process
  telemetry sampling, previously listed as Phase 6, moves into the Phase 7 row; Phases
  7–12 keep their numbers because ADRs and earlier plans reference them and those are
  historical records.
- README, `limitations.md`, `roadmap.md` and the Phase 6 plan header no longer describe
  TensorRT as unimplemented, and the roadmap's "immediate next steps" no longer describe
  Phase 3 as upcoming. README's status line and "current position" agreed with neither
  the repository nor each other; both now say Phase 6, implemented but unvalidated.
- **`[tensorrt]` extra pinned to `tensorrt==11.3.0.99` and `cuda-python` added.** The
  previous `tensorrt>=10.0` floor admitted TensorRT 10.x, where `create_network(0)`
  yields a *weakly* typed network — the build layer passes exactly that and then asserts
  `STRONGLY_TYPED`, which is only the default from 11.0. `cuda.bindings.runtime` is
  imported directly by the build layer and was not declared at all, so installing the
  extra produced an unrunnable backend. `docs/environment.md` §5 rewritten accordingly.

### Added — Phase 5B warmup-sufficiency finding (documentation only; no data changed)

- **19 of the 20 published Phase 5B runs report `warmup_sufficient: false`**, a value
  already stored in `analysis/controlled.json` but absent from the evidence README, the
  operator notes and the pre-registered stability gate. Settle indices are now tabulated
  per cell. Recorded strictly as a measured observation: no causal link to the PyTorch
  batch-1 spread failure is claimed, the ±2% criterion's own known jitter-sensitivity is
  restated alongside it, and the hypothesis that a longer warmup would help is marked
  untested. No raw sample, result or telemetry file was modified.

### Added — Phase 5B: controlled PyTorch vs ONNX Runtime comparison (NVIDIA L4)

- **The Phase 5A matrix rerun with the input path unified**, and nothing else changed:
  same configs, iteration counts, timing boundary, telemetry method and alternating run
  order. Evidence: `results/published/2026-09-20-phase5b-l4-controlled-inputs/`; Phase 5A's
  tree is untouched and keeps its caveat.
- **Input identity proven on the GPU before measuring:** PyTorch's device tensor and ORT's
  device-resident `OrtValue` are byte-identical to `core.inputs.synthetic_input` and to
  each other, at batch 1 and 8, with `input_generator` / `input_seed` recorded.
- **Results** (host-side series, median of 5 runs): PyTorch 5.650 ms / ORT 3.246 ms at
  batch 1; PyTorch 11.719 ms / ORT 12.676 ms at batch 8. Ranges are disjoint in both
  cells, and the ordering reverses with batch size — ORT 1.741× faster at batch 1,
  PyTorch 1.082× faster at batch 8. Versus Phase 5A every median moved by −1.22% to
  +0.27%, inside each cell's own run-to-run spread.
- **One cell missed the pre-registered stability gate and is published as measured:**
  PyTorch batch 1 at 5.87% spread (limit 5%), with the SM clock pinned at 2040 MHz and no
  power-cap flag, so the cause is unestablished rather than explained away.
- `analysis/gpu_validation/input_identity_gpu.py` collects the GPU-side identity evidence;
  the ORT backend retains the `OrtValue` it binds so a bound input can be read back
  (IOBinding exposes only bound outputs). Execution is unchanged.

### Changed — Phase 5A cleanup (methodology and evidence)

- **One input generator for every backend.** `PyTorchBackend` drew benchmark inputs from
  `torch.randn` while ONNX Runtime used numpy: same shape, dtype and distribution,
  different values. Both now take the canonical array from `core.inputs.synthetic_input`,
  so a cross-backend comparison differs only in the runtime under test. Each result
  records `input_generator`; runs published before this change used the old PyTorch
  stream and must not be pooled with later ones.
- **Correlation is no longer written as causation.** The power-cap findings in
  `limitations.md`, the Phase 5A evidence README and `ENGINEERING_LOG.md` now separate
  measured telemetry, observed association and interpretation, and state that no causal
  direction between throughput and the 72 W cap was tested.
- **`PyTorchOptions.cudnn_benchmark` described autotuning wrongly.** It happens at the
  first forward pass of a shape — the untimed sanity pass in `prepare()` — not during
  warmup. Docstring, methodology §7 and limitations now carry the measured numbers.

### Added — Phase 5A cleanup

- `tests/unit/test_input_identity.py`: 12 tests proving every executing backend receives
  bit-identical inputs, iterating `EXECUTING_BACKENDS` so a future backend is covered
  automatically, plus an explicit assertion of the precision-cast conversion boundary.
- `analysis/sanitize_evidence.py`: generates a redacted **copy** of an evidence tree
  (hostname, GPU UUID and serial, home paths, arbitrary `--replace` literals) and
  verifies the redaction; raw evidence is never modified.
- `limitations.md` §1b explains why a CUDA machine reports 7 skips where a CPU machine
  reports none, test by test.

### Added — Phase 5A: first validation on NVIDIA hardware (NVIDIA L4)

- **The first GPU measurements in this repository**, all from one `g2-standard-4`
  instance with an NVIDIA L4 (driver 580.159.04, CUDA 13.0, torch 2.14.0+cu130,
  onnxruntime-gpu 1.30.0). Evidence: `results/published/2026-09-20-phase5a-l4/`.
- **`CudaEventTimer` validated against real asynchronous GPU work:** event time scales
  2.0006×/4.0001× with the workload (CV 0.019%), kernel launch is 0.156% of execution,
  and the synchronized host interval contains the device interval in every trial. A
  deliberately unsynchronized host timer read 0.0161 ms where the truth was 19.68 ms.
- **ONNX Runtime CUDA EP proven to execute on the GPU:** 122/122 nodes on the CUDA EP,
  device-resident IOBinding, loaded CUDA libraries confirmed from `/proc/self/maps`, and
  its end-of-`Run` synchronization measured (+0.58%, against a control at 13.6%).
- **GPU correctness:** PyTorch CPU vs CUDA and PyTorch CUDA vs ORT CUDA both 9/9 under
  the unchanged Phase 4 tolerance; the FP16 control still rejected. **TF32 measured for
  the first time:** 11.8–14.7× the tolerance, while top-1 agreement stayed 100%.
- **First controlled GPU benchmark:** ResNet-50 FP32, batch 1 and 8, 5 alternating
  repeats per cell, host-side timing for both backends, with 100 ms NVML telemetry.
  Run-to-run spread 1.66–4.90%; the backends' ranges are disjoint in both cells, and the
  ordering reverses between batch 1 and batch 8. Qualified by a binding 72 W power cap
  in three of four cells and by session-level warming (55 → 80 °C).

### Added — Phase 5A preparation: GPU validation runbook

- `docs/runbooks/gpu-validation.md`: the procedure for validating NVML, the PyTorch CUDA
  path, `CudaEventTimer`, warmup, CPU-vs-CUDA and PyTorch-vs-ORT correctness, the ORT
  CUDA EP, telemetry and a small controlled benchmark on the first NVIDIA machine, with
  pre-registered success criteria. **Nothing in it has run on NVIDIA hardware.**
- `analysis/gpu_validation/`: the scripts and configs the runbook uses. They are
  validation procedures, not benchmark features; on the development machine only their
  refusal paths (and the analysis scripts, on CPU data) have been executed.

### Fixed

- **`GPUInfo.multiprocessor_count` held CUDA cores, not SMs** (environment schema
  1.1 → **1.2**, field renamed to `cuda_core_count`). Found on an NVIDIA L4, where NVML
  reported 7424 against the CUDA runtime's 58 SMs (7424 = 58 × 128).
  `nvmlDeviceGetNumGpuCores` documents itself as "the device's core count"; NVML exposes
  no SM count. Older results still load, as the field is simply absent from them.
- `docs/environment.md` suggested the `cu128` PyTorch index, which stops at torch 2.11.0;
  the documented commands now pin 2.14.0 on indexes that carry it.

### Added — Phase 4: ONNX export and ONNX Runtime backend

- **Reproducible ONNX export** (`gpu-bench onnx export`): dynamo exporter, opset 20
  pinned, dynamic batch, single self-contained file, with a provenance manifest (model,
  weights SHA-256, tool versions, exporter settings, I/O spec, artifact SHA-256, git
  state). Artifacts are validated before use (hash, no external data, `onnx.checker`
  full check, opset, I/O) and never committed. Re-export is byte-identical.
- **PyTorch-vs-ONNX Runtime correctness check** (`gpu-bench onnx verify`): identical
  inputs and weights, pre-registered scale-aware tolerance plus top-1, and an FP16
  negative control that must fail. Stores `report.json` + raw outputs;
  `analysis/rederive_correctness.py` re-derives the verdict without importing the tool.
- **ONNX Runtime backend** (`backend: onnxruntime`) through the existing lifecycle:
  load = read + hash, build = session creation, prepare = inputs + sanity run + measured
  node placement. CPU EP executed on real hardware; CUDA EP implemented (IOBinding,
  `use_tf32=0` for FP32) and **unverified on NVIDIA hardware**.
- **No silent CUDA→CPU fallback:** availability pre-check, `session.get_providers()`
  post-check and measured placement. Verified against the real onnxruntime-gpu 1.30,
  which silently built a CPU session when asked for CUDA without CUDA libraries.
- `core/inputs.py` (backend-independent synthetic inputs), `core/correctness.py`,
  `models/torch_loader.py`, ADR 0007, CI job running the headline check end to end.
- Extras: `[onnx-cpu]`, `[onnx]`, `[onnx-export]`, floored at tested versions. The ORT
  backend needs Python ≥ 3.11 (onnxruntime 1.30).

### Fixed

- CLI crashed on a Windows cp1252 console on a `Δ` column header (after saving its
  evidence, so a passing check exited 1). Guarded by a test over all CLI string literals.
- Rich markup swallowed bracketed text in error messages (`[torch]`, `[onnx-export]`
  extras disappeared from the console). Error text is now escaped.
- CPU result note and banner now state the measurements are not suitable for backend
  performance ranking.

### Added — Phase 3: PyTorch backend (ResNet-50)

- **PyTorch backend** through the existing lifecycle. CPU path executed on real
  hardware; CUDA path implemented and structurally tested, **not validated on NVIDIA
  hardware** (the development machine has no NVIDIA GPU).
- **No implicit device, no fallback.** `cuda:0` without CUDA is `status: unavailable`.
- **Explicit precision.** Casting (not autocast); IEEE FP32 enforced via the
  `fp32_precision` API because PyTorch defaults cuDNN convolutions to TF32; effective
  settings recorded; output dtype verified by a sanity forward pass (ADR 0005).
- **`CudaEventTimer`**: CUDA-event device time as the primary sample, synchronized
  host time as a secondary series.
- **Model registry** (ResNet-50: class `ResNet`, 25,557,032 parameters asserted) and
  **pinned-weights verification** with the SHA-256 recorded on every result (ADR 0006).
- `gpu-bench models list|fetch`; example configs for CPU FP32 and CUDA FP16.
- **Contract changes:** `status: unavailable` is now producible; errors tagged with
  the exact lifecycle phase; `validate(config, environment)`; `execution_context()`;
  backend `settings` / `device_kind`; `model_info`; optional secondary timing;
  validated `backend_options`. Result schema **1.1** (1.0 still loads).
- **CPU labelling.** CPU results carry a note, a `cpu-` storage prefix and
  `is_gpu_measurement == False`.
- **Gross-anomaly flag.** A sample > 10× the run's median adds an `ANOMALY` note,
  shown in the CLI; samples are never dropped. Added after a real run recorded a
  627 s sample spanning a system suspend.
- **Power source recorded** in the environment (`power_plugged`, `battery_percent`);
  environment schema **1.1**.
- `analysis/ab_weights_cpu.py`: stdlib-only re-derivation and comparison script.
- Published evidence: `results/published/2026-09-18-phase3-cpu-resnet50/`.

### Fixed

- `.gitignore` rule `models/` also matched `src/gpu_benchlab/models/`; now anchored.
- A run that never created a timer no longer claims `timing_mechanism: wall_clock`.
- An OOM raised during `prepare()` was tagged phase `execute`.
- The effective input shape was not recorded when `model.input_shape` was omitted.

### Findings (documented, not fixed)

- Within-run non-stationarity on the laptop CPU; cause unconfirmed.
- The pinned-vs-random weights A/B experiment was inconclusive (confounded).

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
