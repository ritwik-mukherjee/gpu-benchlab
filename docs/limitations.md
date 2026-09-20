# Limitations

This document exists so that no reader has to guess what has actually been verified.
It is updated whenever something is implemented but not executed on real hardware.

Last updated: 2026-09-19 (end of Phase 4).

---

## 0. The four kinds of evidence in this repository

Every claim about this tool falls into exactly one of these. They must never be
blurred.

| | Kind | Exists? | Where |
|---|---|---|---|
| **A** | **Real CPU measurements** | Yes | PyTorch and ONNX Runtime (CPU EP) ResNet-50 on the development laptop's CPU, plus the PyTorch-vs-ORT correctness reports. Evidence: `results/published/2026-09-18-phase3-cpu-resnet50/`, `results/published/2026-09-19-phase4-onnx/`. Stamped `device_kind: cpu`, stored under `cpu-` prefixes, carry a CPU note. **Not GPU performance.** |
| **B** | **Simulated measurements** | Yes | The `fake` backend. Stamped `is_simulated: true`, `timing_mechanism: scripted`, `sim-` prefixes. **Not measurements of anything.** |
| **C** | **CUDA functionality: implemented, only fake/structurally tested** | Yes | NVML success path, PyTorch CUDA validation, `CudaEventTimer`, ONNX Runtime CUDA EP (provider options, IOBinding, placement). Tested against patched CUDA queries and fake event/NVML APIs. **Never executed on NVIDIA hardware.** |
| **D** | **NVIDIA hardware measurements** | **No** | None exist. No GPU performance claim of any kind can be made from this repository today. |

The development machine (Intel Core i7-8565U, Intel UHD 620 only) has **no NVIDIA GPU**.

---

## 1. What has been executed and verified

| Item | Verified how |
|---|---|
| Package installs on Python 3.10 and 3.12 (Windows) | `uv pip install -e ".[dev]"` |
| Install and test suite **without torch** | Separate venv: all non-torch tests pass, torch tests skip, a `pytorch` config yields `status: unavailable` |
| `gpu-bench hardware` / `doctor` | Executed; output in [environment-report.md](environment-report.md) |
| Host / CPU / RAM / Python / **power source** detection | Executed; power reading matched `Win32_Battery` |
| NVML *absence* handling | Executed on a machine with genuinely no NVIDIA driver |
| Compute-capability → precision matrix | Unit tests against documented capabilities of real GPUs |
| Timing abstraction (`WallClockTimer`) | Real sleeps; sync-hook call order asserted |
| Latency statistics | Hand-computed known answers; stored values re-derived from `raw.json` with the stdlib, for both simulated and real CPU runs |
| Phase separation | Backend with a real 60 ms engine build: latency statistics byte-identical to the no-build run |
| Error status / phase mapping | `unsupported`, `unavailable`, `failed` produced and tagged with the exact lifecycle phase |
| **PyTorch CPU path, ResNet-50** | Real inference executed: class name `ResNet`, 25,557,032 parameters, pinned weights SHA-256 `11ad3fa6…` verified, IEEE FP32 applied and recorded, sanity forward pass, `inference_mode` active for every iteration |
| **`device: cuda:0` on a machine without CUDA** | Genuine `status: unavailable` in phase `validate`; model never built; no CPU fallback |
| TF32 / FP32-precision flag behaviour on torch 2.14 | Observed on the installed build (flags are settable without a GPU): default `cudnn.conv.fp32_precision == "tf32"`; reading legacy `allow_tf32` after using the new API raises; snapshot-and-restore returns the exact original state |
| Process-global torch state restoration | Tested after success and after failure |
| Result schema 1.1, 1.0 compatibility | A committed Phase 2 (schema 1.0) result still loads |
| Gross-anomaly flag | Regression test, and applied to the real stored runs: flags exactly the run containing a system suspend |
| **ONNX export of pinned ResNet-50** | Executed: opset 20, dynamic batch, single self-contained file, `onnx.checker` full check, byte-identical re-export |
| **PyTorch vs ONNX Runtime correctness (CPU EP)** | 9/9 cases pass (max \|Δ\| ≤ 3.1e-6, ≤ 0.44 % of tolerance, top-1/top-5 100 %); FP16 negative control rejected (14.1× the bound); verdict re-derived by a script sharing no code with the tool |
| **ONNX Runtime CPU EP benchmark path** | Real ResNet-50 runs; active EP and per-node placement (58/58 on CPU EP) measured and recorded |
| **No silent CUDA→CPU fallback in ONNX Runtime** | Against the **real onnxruntime-gpu 1.30** on this machine (CUDA libraries absent), where ORT itself silently built a CPU session: the backend returned `unavailable` in phase `build`, 0 samples |
| CLI on a Windows cp1252 console | Regression tests for non-cp1252 characters and for Rich markup swallowing `[extra]` names |
| **NVML success path on a real GPU** | Executed on an NVIDIA L4 (2026-09-20): detection OK, and 19 fields compared with `nvidia-smi` taken either side of the reading. All static fields matched exactly (name, UUID, PCI bus ID, serial, capability 8.9, total memory, 72 W limit, max clocks, persistence, compute mode); dynamic fields fell inside the bracket. Two findings, both now documented: `nvmlDeviceGetNumGpuCores` returns CUDA cores (7424), not SMs (58); and NVML's `used` memory includes driver-reserved memory (493,748,224 B where `nvidia-smi` shows 0 MiB) |

## 2. What has NOT been executed on real NVIDIA hardware (category C)

Implemented and structurally tested; **unverified**. The procedure and pre-registered
success criteria for validating each item are in
[runbooks/gpu-validation.md](runbooks/gpu-validation.md) (not yet executed).

| Item | How it was tested | What is unknown |
|---|---|---|
| ~~NVML success path and GPU field extraction~~ | **Validated on an L4 — moved to §1** | — |
| Multi-GPU enumeration | Fake | Everything real |
| PyTorch CUDA validation (build / availability / index / capability → precision) | torch's CUDA queries patched | Behaviour against a real driver; NVML-vs-CUDA device ordering |
| **`CudaEventTimer`** | Fake event/stream API: call order, primary = event time, host time secondary | **Whether event placement is correct for real asynchronous CUDA work.** Whether the captured stream is the one kernels actually run on. Event resolution and overhead on real hardware. |
| CUDA FP16 / BF16 / TF32 execution | Not executed | The sanity check confirms the output dtype, not which kernels ran. Whether tensor-core kernels were selected (and whether `"ieee"` really disables TF32 on a real GPU) is unverified. |
| `cudnn.benchmark` autotuning during warmup | Not executed | How many warmup iterations autotuning actually needs |
| `torch.cuda.synchronize`, `empty_cache` calls | Not executed (the only unexercised lines in the backend apart from the legacy-TF32 branch) | — |
| GPU memory, OOM on a real device | torch OOM mapping tested with a raised `torch.OutOfMemoryError` | Real OOM behaviour and recovery |
| Legacy `allow_tf32` branch (torch < 2.9) | **Not tested at all** — the installed torch has the new API | Whether the fallback works |
| **ONNX Runtime CUDA EP** | Real-fallback test (real ORT, CUDA absent) + fake ORT module for provider options, IOBinding and placement | Whether the EP loads and runs; whether ResNet-50 places fully on it; whether ORT synchronizes the stream at the end of `Run` (the timing relies on it); IOBinding behaviour; `use_tf32` effect |
| PyTorch-vs-ORT correctness **on CUDA** | Not executed | Whether the same tolerance holds with cuDNN kernels (`use_tf32 = 0`) |

**Semantics of CUDA-event time, stated so it is not misread later.** The primary
CUDA sample is device time between two events on the stream. If the host launches
kernels more slowly than the GPU executes them, idle gaps are *inside* that
interval. It is therefore not "sum of kernel durations". Launch-bound workloads
(small models, small batches) will show this, and the secondary host-time series
exists so the gap can be quantified rather than assumed.

## 3. CPU measurements on this machine — what they can and cannot support (category A)

- **They are real** and correctly labelled. They measure this laptop's CPU.
- **They are never GPU data** and must not be compared with GPU results as a
  speedup. `BenchmarkResult.is_gpu_measurement` is `False` for all of them.
- **Observed non-stationarity within a single run.** In the clean AC-powered run
  (P0), iterations 0–39 averaged ≈ 88 ms and iterations 40–99 averaged ≈ 111–118 ms.
  The 10 warmup samples (80–99 ms) sit in the faster regime, so a fixed
  10-iteration warmup did **not** reach a stationary state, and the run's p50
  blends two regimes. *Possible explanation:* a turbo / package-power-limit
  transition on this 15 W mobile CPU. **Cause unconfirmed** — no clock, power or
  thermal telemetry was recorded.
- **Power source is a plausible confounder and was not recorded before Phase 3's
  close.** The three runs made entirely on battery (R1, P2, R2) were slower than the
  AC run (per-run p50 ≈ 189–224 ms vs ≈ 102 ms), but all three also ran after a
  system resume, so battery power and post-resume state **cannot be separated**
  from this data. From now on the environment records `power_plugged` and
  `battery_percent`; runs before that field existed do not carry it.
- **System suspend is counted as latency.** `time.perf_counter_ns` kept counting
  through an S3 sleep, producing a single 627,219.9 ms sample. The engine now
  attaches an `ANOMALY` note to any run with a sample > 10× its median and the CLI
  shows it; the sample is kept, not dropped. The 10× threshold is a loose
  heuristic: it will not catch moderate disturbances (e.g. the 5–8× outliers seen
  after resume).

## 4. Not yet implemented

Phases 5–12: TensorRT, telemetry sampling during runs, the experiment runner
(matrices, repeats), the comparison engine, dashboard, reports, and the LLM generation
path. Reduced-precision (FP16/INT8) ONNX artifacts are out of Phase 4 scope. See
[roadmap.md](roadmap.md).

## 5. Platform limitations

- **TensorRT-LLM is Linux-first.** Not expected to work on native Windows.
- **Windows GPU telemetry is narrower than Linux.** Unavailable NVML fields degrade
  to `null`, never `0`.
- **WSL2 GPU passthrough** requires a WSL-capable Windows NVIDIA driver; NVML inside
  WSL2 has historically had reduced functionality.

## 6. Methodological caveats

- **The 10-warmup / 100-iteration defaults remain an unvalidated hypothesis.** The
  only real workload measured so far (CPU, above) shows that an iteration-count
  warmup does not guarantee stationarity. A stationarity / drift check is needed
  before any default can be called validated.
- A p99 from 100 samples is not a stable statistic (methodology §6).
- Whether random-init and pinned weights cost the same on CPU is **still unknown**:
  the Phase 3 A/B experiment was inconclusive (see ENGINEERING_LOG 2026-09-18).
  Until answered, only pinned-weight runs are reportable.
- Single runs only. Repeats and between-run variance are not implemented yet
  (Phase 7), and the A/B experiment showed run-to-run variation on this machine
  larger than the effect being tested.
- **No cross-backend latency ranking is possible yet.** Beyond CPU non-stationarity,
  PyTorch and ORT benchmark inputs come from different generators, thread counts differ
  unless set, and runs are not interleaved. The ORT CPU run of 2026-09-19 also drifted
  (20-iteration block means 46–57 ms).
- The FP16 negative control is a proxy for TF32, not a TF32 measurement.
- The ORT backend requires Python ≥ 3.11 (onnxruntime 1.30); the core supports 3.10+.
- Node placement is measured on a separate profiling session with identical options,
  assuming identical partitioning (deterministic given model, options and EPs).

## 7. Known open questions

- Whether each experiment should run in an isolated process, to prevent allocator
  and autotune state leaking between configurations in a matrix run. (Torch global
  flags are restored; allocator/autotune caches are not.)
- How to detect non-stationarity robustly (block means? change-point test?) without
  over-fitting to one machine.
- Whether `ddof=1` is the right default at thousands of iterations.
- Whether percentile confidence thresholds should scale with observed variance.
- ~~Whether `nvmlDeviceGetNumGpuCores` or the CUDA runtime should supply SM count.~~
  **Answered on an L4 (2026-09-20):** NVML returns CUDA cores and has no SM-count
  call, so the field is now `cuda_core_count` and SM count must come from the CUDA
  runtime (environment schema 1.2).
- How to handle GPUs shared with a desktop compositor.
- Do transformers and TensorRT-LLM both deduplicate Qwen3's separately stored `lm_head`?
