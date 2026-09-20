# 2026-09-20 — Phase 5A: first NVIDIA GPU evidence (NVIDIA L4)

**This is the first GPU data in this repository.** Everything here was produced on one
Google Cloud `g2-standard-4` instance with a single NVIDIA L4, copied exactly as the
tools wrote it. Nothing was edited, and no number here came from anywhere else.

Hardware and software, as recorded in every result's `environment` block:

| | |
|---|---|
| GPU | NVIDIA L4, compute capability 8.9, 24,152,899,584 B VRAM, 72 W enforced power limit, max SM clock 2040 MHz |
| Driver / CUDA (driver max) | 580.159.04 / 13.0 |
| PyTorch | 2.14.0+cu130 (CUDA runtime 13.0, cuDNN 9.24.0.43) |
| ONNX Runtime | onnxruntime-gpu 1.30.0, CUDA EP |
| OS / Python | Ubuntu 24.04.5 LTS, kernel 7.0.0-1011-gcp, Python 3.12.3 |
| Model | ResNet-50, pinned `IMAGENET1K_V2`, weights SHA-256 `11ad3fa6…`, 25,557,032 parameters |
| ONNX artifact | opset 20, dynamic batch, SHA-256 `7084761c…` (exported on this machine) |
| Commit | every run carries `git_commit` with `git_dirty: false` |

The CUDA libraries used were the wheel-provided ones (`libcudart.so.13`,
`libcublas.so.13`, `libcudnn.so.9`), recorded per process in `ort/*/ort_cuda_check.json`.
`LD_LIBRARY_PATH` was unset, so the separately installed CUDA 13.4 toolkit was **not**
used — relevant because that toolkit's default PTX is rejected by this R580 driver.

## What is here

| Path | Contents |
|---|---|
| `env-before/`, `env-after/` | `nvidia-smi`, `nvidia-smi -q`, `pip freeze`, git state, `gpu-bench hardware --json`, framework probe |
| `nvml-idle/` | NVML vs `nvidia-smi`, 19 fields compared, both readings kept |
| `pytorch/` | PyTorch CUDA backend checks and the full CUDA-event experiment, with every raw trial |
| `correctness/` | GPU correctness report + `outputs.npz` (all raw outputs, for independent re-derivation) |
| `ort/` | ORT CUDA proof-of-execution in three library conditions, ORT profiles, sync experiment |
| `runs/` | 29 benchmark runs: `result.json`, `raw.json` (every warmup and measured sample), `metadata.json`, `summary.json` |
| `telemetry/` | `nvidia-smi` samples at 100 ms for each run |
| `analysis/`, `logs/`, `NOTES.md` | trajectory analysis, console logs, and the operator log including a recorded deviation |

## Validation results — all checks passed

| Area | Evidence |
|---|---|
| NVML success path | All static fields match `nvidia-smi` exactly. Two findings recorded: `nvmlDeviceGetNumGpuCores` returns **CUDA cores (7424), not SMs (58)**, and NVML's `used` memory includes driver-reserved memory (493,748,224 B where `nvidia-smi` shows 0 MiB) |
| PyTorch CUDA path | Weights, input and output all on `cuda:0`; IEEE FP32 applied and recorded; device identity matched to NVML by UUID; sanity pass recorded |
| CUDA-event timing | Event time scales 2.0006× / 4.0001× with GPU work, CV 0.019%, launch is 0.156% of execution, host interval contains the device interval in every trial (+24–30 µs). The real `CudaEventTimer` read 19.67 ms against a 19.68 ms reference |
| Timer negative control | An **unsynchronized** host timer read **0.0161 ms** where the true time was **19.68 ms** — the experiment can detect the classic async-timing bug |
| No silent CPU fallback | With CUDA hidden, every stored run is `unavailable` in `validate` with zero samples, for both backends |
| ORT really executes on the GPU | 122/122 nodes on the CUDA EP, IOBinding input and output device-resident, CUDA/cuDNN/cuBLAS libraries confirmed loaded, process listed by `nvidia-smi` |
| ORT timing validity | `run_with_iobinding` p50 12.20 ms; adding a device-wide sync adds **+0.58%**. Control: with ORT's sync disabled the call returns in 13.6% of that time, so the check had the power to detect a missing sync |

## Numerical correctness (Phase 4 criterion, unchanged)

`|cand − ref| ≤ 1e-4·max|ref| + 1e-4·|ref|` elementwise plus identical top-1, over
batches {1,4,8} × seeds {0,1,2}.

| Comparison | Result | max abs error | % of tolerance | top-1 |
|---|---|---|---|---|
| PyTorch CPU vs PyTorch CUDA (IEEE FP32) | **9/9 pass** | 4.65e-06 | 0.7% | 100% |
| PyTorch CUDA vs ORT CUDA | **9/9 pass** | 4.83e-04 | 69.7% | 100% |
| PyTorch CPU vs ORT CUDA | **9/9 pass** | 4.85e-04 | 69.6% | 100% |
| PyTorch CPU vs PyTorch **TF32** | 0/9 (observed) | 7.08e-03 | 1183% | 100% |
| PyTorch CPU vs ORT **TF32** | 0/9 (observed) | 9.53e-03 | 1473% | 100% |
| FP16 negative control | **rejected, as required** | — | 1186% | 100% |

Two things this settles. **TF32 is detectable by this tolerance on real hardware**
(11.8–14.7× over the bound), which the Phase 4 FP16 proxy could only argue by analogy.
And **top-1 agreement stayed 100% in every degraded case**, so a top-1-only check would
have passed TF32 and FP16 alike. Repeated runs in one process were bitwise identical.

## Controlled benchmark — PyTorch CUDA vs ONNX Runtime CUDA

One configuration: ResNet-50, pinned weights, IEEE FP32 (`fp32_precision="ieee"` /
`use_tf32=0`), NCHW, device-resident input and output, 100 warmup + 1000 measured
iterations, 5 repeats per cell in separate processes, backend order alternating between
repeats. **Comparable series: host-side time for both** — PyTorch's secondary
synchronized host series against ORT's host clock, because their primary mechanisms
differ (CUDA events vs host wall clock).

| Backend | Batch | Host p50 (median of 5 runs) | Run-to-run min–max | Spread | p90 | p99 | Throughput |
|---|---|---|---|---|---|---|---|
| PyTorch | 1 | **5.655 ms** | 5.517 – 5.760 | 4.29% | 5.740 | 5.932 | 176.8 samples/s |
| ONNX Runtime | 1 | **3.260 ms** | 3.134 – 3.294 | 4.90% | 3.322 | 3.349 | 306.2 samples/s |
| PyTorch | 8 | **11.864 ms** | 11.518 – 11.955 | 3.68% | 11.969 | 12.104 | 676.9 samples/s |
| ONNX Runtime | 8 | **12.641 ms** | 12.466 – 12.675 | 1.66% | 12.795 | 12.927 | 632.4 samples/s |

All 20 runs finished `ok`, none carried an anomaly note, within-run drift was ≤1.2%, and
PyTorch's host time exceeded its event time in every iteration (median +0.012 ms at
batch 1, +0.016 ms at batch 8).

**What can be said.** On this L4, with this software stack and configuration, the two
backends' run-to-run ranges do not overlap in either cell: ONNX Runtime is faster at
batch 1 (about 1.73× on median host latency) and PyTorch is faster at batch 8 (about
1.07×). The direction **reverses with batch size**, which is itself the main result.

**What cannot be said.** Nothing here generalises to another GPU, another model, other
batch sizes, other precisions, concurrent load, or production serving; and nothing here
says anything about TensorRT. None of it may be compared with the CPU numbers from
Phases 3–4.

## Conditions that qualify these numbers

- **Power cap.** The L4's 72 W limit was reached in three of the four cells, flagged by
  the driver as `SwPowerCap` on essentially every busy sample: ORT at batch 1
  (SM clock 1665–1755 MHz), ORT at batch 8 and PyTorch at batch 8 (both ~1230–1260 MHz).
  **PyTorch at batch 1 never hit the cap** (57–65 W) and held 2040 MHz throughout. The
  backends are therefore not competing under identical clocks — ORT at batch 1 is
  power-limited precisely because it completes more work per second.
- **Session-level warming.** GPU temperature rose from 55 °C to 80 °C over the session
  and later repeats are slightly slower (e.g. ORT batch 1: 3.134 → 3.294 ms). The
  alternating order spreads this across both backends rather than removing it. No
  thermal-slowdown flag was ever raised.
- **Warmup differs by backend.** PyTorch's first 10 iterations are within 0.6–2.0% of
  steady state; **ORT's first 10 are 7.1–8.6% faster**, because it starts at the 2040 MHz
  boost clock and is then power-capped down. 10 warmup iterations would have been
  insufficient for ORT; these runs used 100.
- **Input generators still differ** (`torch.randn` for PyTorch, numpy PCG64 for ORT).
  Values differ while shape, dtype and distribution match. For a dense CNN with no
  data-dependent control flow this is not expected to affect latency, but it is an
  uncontrolled difference and is not yet fixed.
- **Peak VRAM is not measured in-process.** The `memory.used` figures in `telemetry/`
  come from 100 ms `nvidia-smi` sampling, which also catches load and export phases;
  they are not a reliable peak for the measured loop.
- ORT executes 122 nodes on CUDA, where on the CPU EP it fused the same graph to 58.
  The graphs being executed are not identical in shape.

## Re-deriving the verdicts

```
python analysis/gpu_validation/trajectory.py results/published/2026-09-20-phase5a-l4/runs/ctl-*/*/
python analysis/gpu_validation/telemetry_summary.py results/published/2026-09-20-phase5a-l4/telemetry/ctl-pytorch-b8-r1.csv 100
```

`correctness/outputs.npz` holds every reference and candidate output, so the correctness
verdicts can be recomputed without this repository's code.
