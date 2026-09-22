# 2026-09-20 — Phase 5B: controlled PyTorch vs ONNX Runtime on an NVIDIA L4

The Phase 5A matrix, rerun with **one thing changed**: both backends now take their
benchmark input from `core.inputs.synthetic_input`, so the tensors are bit-identical
before any runtime-specific conversion. Phase 5A's input mismatch was the last
uncontrolled difference in that comparison; this run removes it.

**The data files here are never edited.** This README is the only human-written file
besides `NOTES.md` (the operator log, written during the session). Every JSON, CSV and
log is exactly what the tools produced. Phase 5A's evidence is untouched and still
stands on its own at `results/published/2026-09-20-phase5a-l4/`.

| | |
|---|---|
| GPU | NVIDIA L4, compute capability 8.9, 72 W enforced power limit, max SM clock 2040 MHz |
| Driver / CUDA (driver max) | 580.159.04 / 13.0 |
| PyTorch / ONNX Runtime | 2.14.0+cu130 (cuDNN 9.24.0.43) / onnxruntime-gpu 1.30.0, CUDA EP |
| OS / Python | Ubuntu 24.04.5 LTS, kernel 7.0.0-1011-gcp, Python 3.12.3 |
| Model | ResNet-50, pinned `IMAGENET1K_V2`, weights SHA-256 `11ad3fa6…`, 25,557,032 parameters |
| Commit under test | `a02c94f`, `git_dirty: false` on every result |
| Input | `numpy.random.default_rng(seed).standard_normal(shape, dtype=float32)`, seed 0, recorded on every result as `input_generator` / `input_seed` |

## Experiment configuration

Identical to Phase 5A apart from the input path. Configs are
`analysis/gpu_validation/configs/controlled-{pytorch,onnxruntime}-cuda-fp32-b{1,8}.yaml`
at the commit above, and each run records its own full configuration.

| Dimension | Value |
|---|---|
| Precision | IEEE FP32: PyTorch `fp32_precision="ieee"`, ORT `use_tf32=0` |
| Batch / shape | 1 and 8 × 3 × 224 × 224, NCHW |
| Data movement | device-resident input and output both sides (PyTorch cuda tensors, ORT IOBinding); no host↔device copy is timed |
| Autotuning | PyTorch `cudnn_benchmark: true`; ORT `cudnn_conv_algo_search: EXHAUSTIVE` — both in the untimed sanity pass |
| Warmup / measured | 100 / 1000 iterations |
| Repeats | 5 per cell, separate processes, backend order alternating between repeats |
| Timing | PyTorch: CUDA events (primary) + synchronized host time (secondary). ORT: host clock around `run_with_iobinding`, whose end-of-Run sync was verified here |
| Telemetry | `nvidia-smi` at 100 ms in a separate process, same fields as Phase 5A |

## Pre-flight verification (before any measurement)

| Check | Evidence | Result |
|---|---|---|
| Input tensors bit-identical on the GPU, batch 1 and 8 | `input-identity/input_identity_gpu.json` | **PASS** — PyTorch's device tensor and ORT's device-resident `OrtValue` both equal the canonical array exactly, and each other |
| Provenance recorded by both backends | same | **PASS** — `input_generator`, `input_seed` |
| ORT executes on the CUDA EP | `ort-verify/` | **PASS** — 122/122 nodes on the CUDA EP, IOBinding device-resident, CUDA/cuDNN/cuBLAS loaded, PID visible to `nvidia-smi` |
| ORT waits for the GPU before returning (and the check could tell if it did not) | `ort-verify/` S1, S2 | **PASS** |
| No silent CPU session when CUDA is hidden | `ort-verify/` N1 | **PASS** |
| PyTorch CUDA synchronization | `analysis/controlled.json` | **PASS** — synchronized host time ≥ CUDA-event time in every measured iteration of all 10 runs (median +0.012 ms at batch 1, +0.016 ms at batch 8) |

## Results

Host-side series for both backends, because their primary mechanisms differ. Each cell is
5 independent runs of 1000 measured iterations.

| Backend | Batch | p50 (median of 5 runs) | Run min–max | Spread | p90 | p99 | Throughput |
|---|---|---|---|---|---|---|---|
| PyTorch | 1 | **5.650 ms** | 5.532 – 5.864 | **5.87%** | 5.754 | 5.975 | 176.8 samples/s |
| ONNX Runtime | 1 | **3.246 ms** | 3.123 – 3.277 | 4.75% | 3.305 | 3.338 | 307.9 samples/s |
| PyTorch | 8 | **11.719 ms** | 11.499 – 11.919 | 3.58% | 11.848 | 11.978 | 685.8 samples/s |
| ONNX Runtime | 8 | **12.676 ms** | 12.359 – 12.729 | 2.93% | 12.849 | 12.983 | 632.5 samples/s |

p90/p99 are the medians of the per-run values; percentiles are never pooled across runs.

**Backend comparison.** Run-to-run ranges do not overlap in either cell:

- **Batch 1:** ONNX Runtime is faster, median ratio **1.741×** (ORT max 3.277 ms < PyTorch min 5.532 ms).
- **Batch 8:** PyTorch is faster, median ratio **1.082×** (PyTorch max 11.919 ms < ORT min 12.359 ms).

The direction reverses between batch 1 and batch 8, as it did in Phase 5A.

## Stability gate — one cell fails, and is reported anyway

The gate was pre-registered in Phase 5A and applied unchanged.

| Condition | Result |
|---|---|
| all runs `ok`, no anomaly notes | PASS — 20/20, none flagged, none discarded |
| within-run drift ≤ 2% | PASS — max 1.53% |
| PyTorch containment violations = 0 | PASS |
| no thermal slowdown | PASS — no `HwThermal`/`SwThermal` flag in any run |
| run-to-run spread ≤ 5% | **FAIL for PyTorch batch 1: 5.87%** (others 2.93–4.75%) |

**PyTorch batch 1 does not meet the stability criterion.** It is published as measured.

What the telemetry shows for that cell: the SM clock was **2040 MHz in every sample of
every run** and **no `SwPowerCap` flag ever appeared** (58.7–64.3 W against the 72 W
limit), so neither clock throttling nor the power cap accounts for the spread. GPU
temperature rose 52 → 78 °C across the repeats, but the per-run medians are not monotonic
in temperature (r3 = 5.851 ms at 67–71 °C; r4 = 5.566 ms at 71–75 °C).

The cause is therefore **not established**. One untested hypothesis is recorded in
`NOTES.md`: at batch 1 with the GPU below its power limit this configuration may be bound
by host-side kernel launch rather than device execution, which would expose it to
scheduling jitter on a 4 vCPU instance that is also running the telemetry sampler. No
experiment here distinguishes that from the alternatives.

Note this cell was also the least stable in Phase 5A (4.29%, just inside the gate).

## Warmup sufficiency — a measured property of these runs, recorded after publication

This section was added after the matrix ran, from values already stored in
`analysis/controlled.json`. **No measurement, raw sample or result file was changed.**

**MEASURED OBSERVATION.** `analysis/controlled.json` carries a `warmup_sufficient` flag
per run, defined in `analysis/gpu_validation/trajectory.py` as: the trajectory has a
settle index *k* (the first iteration of the concatenated warmup+measured series from
which every later block median stays within ±2% of *m\**), **and** *k* ≤ the configured
warmup count. With warmup configured at 100 and 1000 measured iterations, **19 of the 20
runs report `warmup_sufficient: false`.** Per-cell settle indices:

| Cell | settle indices (5 repeats) |
|---|---|
| PyTorch b1 | 860, 330, 320, 0, 1010 |
| ONNX Runtime b1 | 980, 810, 490, 520, 750 |
| PyTorch b8 | 480, 470, 300, 380, 530 |
| ONNX Runtime b8 | 140, 460, 430, 440, 210 |

Only PyTorch b1 repeat 4 (*k* = 0) satisfies the criterion. A *k* of 1010 means the
series first met the ±2% condition 910 iterations into the measured window.

**WHAT THIS DOES NOT SAY.** This is **not** an explanation of the PyTorch batch-1
stability failure. No experiment here tests whether warmup length affects the run-to-run
spread, and the cell containing the largest settle index also contains the only run that
met the criterion — so the two do not even order consistently within the cell.

**A SECOND, COMPOUNDING CAVEAT.** The ±2% block-median rule is *itself* known to be
unreliable at this jitter level: Phase 5A found it "jitter-dominated and could not
separate warmup from noise" (`../2026-09-20-phase5a-l4/NOTES.md`, and `limitations.md`
§6). A `false` verdict may therefore reflect the sensitivity of the criterion rather
than genuine non-stationarity. Which of the two it is has not been established.

**UNTESTED HYPOTHESIS.** A longer warmup might reduce the observed spread, or might
change nothing. Testing it needs an experiment this run did not perform — for example
the same matrix at several warmup counts, with a stationarity criterion registered in
advance and validated against a synthetic trajectory of known settling behaviour.

**STATUS OF THE PUBLISHED NUMBERS.** The results table above stands exactly as measured.
This is a qualification on how confidently a steady state can be claimed, not a
correction, and nothing above it has been recomputed or restated.


## Telemetry

| Cell | SM clock | Power (median) | Event flags |
|---|---|---|---|
| PyTorch b1 | 2040 MHz throughout | 58.7–64.3 W | none |
| ONNX Runtime b1 | 1650–1785 MHz | ≈71.5–72.2 W | `SwPowerCap` on every busy sample |
| PyTorch b8 | 1215–1275 MHz | ≈71.9–72.1 W | `SwPowerCap` on every busy sample |
| ONNX Runtime b8 | 1230–1275 MHz | ≈71.9–72.1 W | `SwPowerCap` on every busy sample |

*Measured:* the flags, clocks and watts above. *Observed association:* the three capped
cells ran at lower SM clocks than the uncapped one. *Interpretation:* the cells did not
execute at equal clocks, so part of any latency difference may reflect that rather than
the runtimes. **No causal direction was tested** — establishing one needs locked clocks or
a swept power limit, which this run did not do.

## Phase 5B vs Phase 5A — descriptive only

Same hardware, same configuration, same day; the only deliberate change is the unified
input path (Phase 5A's PyTorch runs used `torch.randn`, and its PyTorch results carry no
`input_generator` field at all).

| Cell | 5A p50 | 5B p50 | Δ | 5A spread | 5B spread |
|---|---|---|---|---|---|
| PyTorch b1 | 5.655 ms | 5.650 ms | **−0.09%** | 4.29% | 5.87% |
| ONNX Runtime b1 | 3.260 ms | 3.246 ms | **−0.44%** | 4.90% | 4.75% |
| PyTorch b8 | 11.864 ms | 11.719 ms | **−1.22%** | 3.68% | 3.58% |
| ONNX Runtime b8 | 12.641 ms | 12.676 ms | **+0.27%** | 1.66% | 2.93% |

Every shift is smaller than the corresponding cell's run-to-run spread, and the backend
ordering and its reversal with batch size are unchanged (batch 1: 1.735× → 1.741×;
batch 8: 1.065× → 1.082×). Throughput tracks latency: PyTorch b8 685.8 vs 676.9
samples/s, ORT b1 307.9 vs 306.2 samples/s.

Telemetry is likewise consistent between the phases: the same three cells were power
capped, the same cell was not, and clocks sit within a few tens of MHz of the 5A values.

**What this does not establish.** It does not prove the input generator is irrelevant to
latency in general; it shows that on this GPU, for this model and these batch sizes, the
difference between the two input streams did not move the medians beyond run-to-run
spread. Nothing here generalises to other GPUs, models, batch sizes, precisions or
concurrent load, and none of it may be compared with CPU results from Phases 3–4.

## What is here

| Path | Contents |
|---|---|
| `runs/ctl-<backend>-b<N>-r<rep>/<id>/` | `result.json`, `raw.json` (every warmup and measured sample), `metadata.json`, `summary.json`, `logs/` |
| `telemetry/` | `nvidia-smi` samples at 100 ms, one CSV per run |
| `analysis/` | `controlled.json` (per-run trajectory, drift, containment, spread) and one telemetry summary per run |
| `input-identity/` | GPU-side proof that both backends received the same bytes |
| `ort-verify/` | ORT CUDA execution, placement, sync experiment, no-fallback test, ORT profile |
| `env/`, `env-after/` | `nvidia-smi`, `nvidia-smi -q`, `pip freeze`, git state, `gpu-bench hardware --json` |
| `logs/`, `NOTES.md` | console logs; operator notes including the stability-gate failure |

## Reproducing and sharing

```
python analysis/gpu_validation/trajectory.py results/published/2026-09-20-phase5b-l4-controlled-inputs/runs/ctl-*/*/
python analysis/gpu_validation/telemetry_summary.py results/published/2026-09-20-phase5b-l4-controlled-inputs/telemetry/ctl-pytorch-b1-r3.csv 100
```

These files carry infrastructure identifiers (VM hostname, GPU UUID and serial,
`/home/<user>` paths) because that is what the tools recorded. To share a copy outside
the project, generate a redacted one rather than editing these:

```
python analysis/sanitize_evidence.py results/published/2026-09-20-phase5b-l4-controlled-inputs /tmp/phase5b-public --host gpu-benchlab-l4
```

There are no credentials, tokens, keys or external IP addresses in this directory.
