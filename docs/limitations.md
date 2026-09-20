# Limitations

This document exists so that no reader has to guess what has actually been verified.
It is updated whenever something is implemented but not executed on real hardware.

Last updated: 2026-09-20 (Phase 5A: first validation on NVIDIA hardware).

---

## 0. The four kinds of evidence in this repository

Every claim about this tool falls into exactly one of these. They must never be
blurred.

| | Kind | Exists? | Where |
|---|---|---|---|
| **A** | **Real CPU measurements** | Yes | PyTorch and ONNX Runtime (CPU EP) ResNet-50 on the development laptop's CPU, plus the PyTorch-vs-ORT correctness reports. Evidence: `results/published/2026-09-18-phase3-cpu-resnet50/`, `results/published/2026-09-19-phase4-onnx/`. Stamped `device_kind: cpu`, stored under `cpu-` prefixes, carry a CPU note. **Not GPU performance.** |
| **B** | **Simulated measurements** | Yes | The `fake` backend. Stamped `is_simulated: true`, `timing_mechanism: scripted`, `sim-` prefixes. **Not measurements of anything.** |
| **C** | **CUDA functionality: implemented, only fake/structurally tested** | Yes | What remains in §2: multi-GPU enumeration, OOM handling, FP16/BF16 execution, the legacy TF32 branch, TensorRT and the LLM path. Tested against patched CUDA queries and fake APIs. |
| **D** | **NVIDIA hardware measurements** | **Yes, since 2026-09-20** | One NVIDIA L4 (g2-standard-4, driver 580.159.04, CUDA 13.0, torch 2.14.0+cu130, onnxruntime-gpu 1.30.0). ResNet-50 FP32 at batch 1 and 8, 20 controlled runs plus 9 warmup-trajectory runs, with NVML telemetry and full provenance. Evidence: `results/published/2026-09-20-phase5a-l4/`. Stamped `device_kind: cuda`. **This is one GPU, one model, two batch sizes, FP32 only.** |

The development machine (Intel Core i7-8565U, Intel UHD 620 only) has **no NVIDIA GPU**;
GPU work runs on a cloud L4 instance.

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
| **PyTorch CUDA path, ResNet-50, on an L4** | Real GPU inference: weights, input and output all on `cuda:0`, IEEE FP32 applied and recorded, sanity pass, device identity matched to NVML by UUID, allocated memory covering the weights |
| **`CudaEventTimer` on an L4** | Event time scales 2.0006× / 4.0001× with GPU work (CV 0.019%); launch is 0.156% of execution; the synchronized host interval contains the device interval in every trial (+24–30 µs); the class agrees with an independent event pair (19.67 vs 19.68 ms); a 20 ms host gap between events is counted, as documented. **Control:** an unsynchronized host timer read 0.0161 ms where the truth was 19.68 ms |
| **No silent CPU fallback, on real hardware** | With `CUDA_VISIBLE_DEVICES=""`, both backends stored `unavailable` in `validate` with zero samples |
| **ONNX Runtime CUDA EP executes on the GPU** | 122/122 nodes on the CUDA EP (profiled), IOBinding input and output device-resident, `libcudnn.so.9`/`libcublas.so.13`/`libcudart.so.13` confirmed loaded in-process, PID listed by `nvidia-smi`, `use_tf32=0` effective |
| **ORT's end-of-`Run` stream synchronization** | Adding a device-wide sync after `run_with_iobinding` changed the median by +0.58% (12.20 → 12.27 ms). **Control:** with `disable_synchronize_execution_providers=1` the call returned in 13.6% of that time, so the check could have detected a missing sync |
| **GPU correctness, CPU vs CUDA and PyTorch vs ORT** | 9/9 in each comparison under the unchanged Phase 4 tolerance (max \|Δ\| 4.65e-6 CPU-vs-CUDA, 4.83e-4 PyTorch-vs-ORT), top-1/top-5 100%, FP16 control rejected. **TF32 measured:** 11.8–14.7× the tolerance while top-1 stayed 100%, so the tolerance detects TF32 and top-1 alone does not |
| **First GPU benchmark, ResNet-50 FP32 on an L4** | 20 controlled runs (2 backends × batch 1 and 8 × 5 repeats, alternating order), all `ok`, no anomalies, run-to-run spread 1.66–4.90%, with 100 ms NVML telemetry. Evidence: `results/published/2026-09-20-phase5a-l4/` |
| **NVML success path on a real GPU** | Executed on an NVIDIA L4 (2026-09-20): detection OK, and 19 fields compared with `nvidia-smi` taken either side of the reading. All static fields matched exactly (name, UUID, PCI bus ID, serial, capability 8.9, total memory, 72 W limit, max clocks, persistence, compute mode); dynamic fields fell inside the bracket. Two findings, both now documented: `nvmlDeviceGetNumGpuCores` returns CUDA cores (7424), not SMs (58); and NVML's `used` memory includes driver-reserved memory (493,748,224 B where `nvidia-smi` shows 0 MiB) |

## 2. What has NOT been executed on real NVIDIA hardware (category C)

Implemented and structurally tested; **unverified**. The procedure and pre-registered
success criteria are in [runbooks/gpu-validation.md](runbooks/gpu-validation.md);
§1–§11 of it were executed on an L4 on 2026-09-20, and what follows is what that run
did **not** cover.

| Item | How it was tested | What is unknown |
|---|---|---|
| ~~NVML success path and GPU field extraction~~ | **Validated on an L4 — moved to §1** | — |
| Multi-GPU enumeration | Fake | Everything real. The L4 instance has exactly one GPU, so NVML-vs-CUDA device ordering is still unverified |
| ~~PyTorch CUDA validation~~ | **Validated on an L4 — see §1** | Index selection beyond device 0 |
| ~~**`CudaEventTimer`**~~ | **Validated on an L4 — see §1** | Behaviour with multiple streams or CUDA graphs |
| CUDA FP16 / BF16 execution | Not executed | Which kernels run, and whether tensor cores are selected. **TF32 was executed and measured** (§1): it changes results by 11.8–14.7× the FP32 tolerance, so `"ieee"` and `use_tf32=0` demonstrably differ from TF32 mode |
| `cudnn.benchmark` autotuning | Executed on an L4, ResNet-50 batch 1, 3 repeats each: **on** → medians 5.488–5.571 ms with `prepare_inputs_ms` 585–595 ms; **off** → medians 5.616–5.646 ms with `prepare_inputs_ms` 363–371 ms. Autotuning costs ≈220 ms once and buys ≈1.5–2.5% steady-state latency (ranges disjoint) | Its effect at other batch sizes and shapes. It runs in the untimed sanity pass, **not** during warmup as `PyTorchOptions.cudnn_benchmark`'s description claims |
| `torch.cuda.synchronize`, `empty_cache` calls | Not executed (the only unexercised lines in the backend apart from the legacy-TF32 branch) | — |
| GPU memory, OOM on a real device | torch OOM mapping tested with a raised `torch.OutOfMemoryError` | Real OOM behaviour and recovery |
| Legacy `allow_tf32` branch (torch < 2.9) | **Not tested at all** — the installed torch has the new API | Whether the fallback works |
| ~~**ONNX Runtime CUDA EP**~~ | **Validated on an L4 — see §1** | TensorRT EP; multi-stream; CUDA graphs |
| ~~PyTorch-vs-ORT correctness **on CUDA**~~ | **Validated on an L4 — see §1** | Other models and precisions |

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

## 3b. GPU measurements on the L4 — what qualifies them (category D)

- **They are real** and stamped `device_kind: cuda`. They measure one NVIDIA L4 on a
  `g2-standard-4` instance, for ResNet-50 FP32 at batch 1 and 8 only.
- **The 72 W power cap binds in three of four cells.** The driver flagged `SwPowerCap`
  on essentially every busy sample for ORT batch 1 (SM clock 1665–1755 MHz), ORT batch 8
  and PyTorch batch 8 (~1230–1260 MHz). PyTorch batch 1 never reached the cap (57–65 W)
  and held 2040 MHz. The backends are therefore not running at equal clocks, and the
  cell where one is capped and the other is not is the cell with the largest gap.
- **The GPU warmed from 55 °C to 80 °C across the session** and later repeats are
  slightly slower (ORT batch 1: 3.134 → 3.294 ms). Alternating the backend order spreads
  this across both rather than removing it. No thermal-slowdown flag was ever raised.
- **Warmup adequacy is backend-specific.** PyTorch's first 10 iterations are within
  0.6–2.0% of steady state; ORT's first 10 are 7.1–8.6% *faster*, because it starts at
  the boost clock and is then power-capped down. The 10-iteration default is not safe
  for ORT on this GPU; the controlled runs used 100.
- **Input generators still differ between backends** (`torch.randn` vs numpy PCG64):
  same shape, dtype and distribution, different values. Not expected to matter for a
  dense CNN, but uncontrolled and not yet fixed.
- **Peak VRAM is not measured in-process.** `telemetry/` holds 100 ms `nvidia-smi`
  samples that also cover load and export phases, so they are not a peak for the
  measured loop. In-process peak memory is Phase 6 work.
- **Graphs are not identical:** ORT executes 122 nodes on CUDA where the CPU EP fused
  the same model to 58.

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

- **The 10-warmup default is now known to be backend-dependent.** On the L4 it is
  adequate for PyTorch (first 10 iterations within 0.6–2.0% of steady state) and
  inadequate for ONNX Runtime (first 10 are 7.1–8.6% faster, before the power cap pulls
  clocks down). On CPU it does not reach stationarity at all. A stationarity rule must
  be registered before the data it judges: the ±2% block-median rule used in Phase 5A
  proved jitter-dominated and could not separate warmup from noise (see
  `results/published/2026-09-20-phase5a-l4/NOTES.md`).
- A p99 from 100 samples is not a stable statistic (methodology §6).
- Whether random-init and pinned weights cost the same on CPU is **still unknown**:
  the Phase 3 A/B experiment was inconclusive (see ENGINEERING_LOG 2026-09-18).
  Until answered, only pinned-weight runs are reportable.
- Single runs only. Repeats and between-run variance are not implemented yet
  (Phase 7), and the A/B experiment showed run-to-run variation on this machine
  larger than the effect being tested.
- **Cross-backend comparison is possible on the GPU under stated conditions, and still
  not on CPU.** The Phase 5A runs are interleaved, repeated five times, use host-side
  time for both backends and report their spread; their ranges are disjoint in both
  cells. They remain qualified by §3b (power cap, warming, input generators). No CPU
  result may be compared with any GPU result.
- ~~The FP16 negative control is a proxy for TF32, not a TF32 measurement.~~ **TF32 has
  now been measured** on the L4: 11.8–14.7× the tolerance, confirming the proxy's
  premise — though top-1 agreement stayed 100%, so top-1 alone detects neither.
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
