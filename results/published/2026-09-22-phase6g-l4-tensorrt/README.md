# 2026-09-22 — Phase 6G: TensorRT FP32 controlled benchmark on an NVIDIA L4

**The data files here are never edited.** This README is the only human-written file.
Every JSON, CSV and log under `runs/`, `telemetry/`, `logs/`, `analysis/`,
`engine-provenance/` and `correctness/` is exactly what the tools produced — copied
verbatim from `results/phase6/controlled/` and `results/phase6/correctness/`.

| | |
|---|---|
| GPU | NVIDIA L4, compute capability 8.9 (Ada Lovelace), 72 W enforced power limit, max SM clock 2040 MHz |
| Driver / CUDA (driver max) | 580.159.04 / 13.0 |
| TensorRT | 11.3.0.99 |
| OS / Python | Ubuntu 24.04 (kernel `7.0.0-1011-gcp`), Python 3.12.3 |
| Model | ResNet-50, pinned `IMAGENET1K_V2`, weights SHA-256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`, 25,557,032 parameters (matches expected) |
| Repository commit under test | `5679b984a7f9a88ac1ce44e870ed97153adcbb37` |

**A note on `git_commit`/`git_dirty` inside the evidence files.** Every `metadata.json`
in `runs/` records `git_commit: 24afe32c5ba4db9c6503a3eb2bc49556ef66bb86` and
`git_dirty: true` — the exact code state at the moment each run executed
(2026-09-22 10:09 UTC), which predates the documentation commit above. `git_dirty: true`
was caused solely by the then-untracked TensorRT backend smoke config sitting in the
working tree at benchmark time; no tracked source, config, or engine-build file differed
from commit `24afe32`. This is expected and does not affect the measurements.

## Engine provenance

- **The FP32 TensorRT engine used for every Phase 6G controlled run:**
  - engine SHA-256: `20670508aebc2464c3633aa46f937520432d7fb7d7198a007f1c7844ebecb9bc`
  - build_identity: `2f1c5499720c388ade4c0fd7ec535134c20febb9aadb50d38de66ab3ee40f7aa`
  - optimization profile: min=`[1,3,224,224]` / opt=`[8,3,224,224]` / max=`[8,3,224,224]`
  - **zero engine builds occurred during Phase 6G** — `engine_built_this_run: false` recorded in all 10 runs
  - **batch 1 and batch 8 intentionally share this one engine and profile.** Batch 1 is not measured against a batch-1-optimized engine; both cells exercise the same compiled kernels and the same profile chosen with opt batch = 8.
  - the manifest (JSON metadata only, no compiled binary) is published at `engine-provenance/resnet50-IMAGENET1K_V2-opset20-dynbatch-trt11.3.0.99-sm89-b1_8_8-ieee_fp32-2f1c5499720c.manifest.json`

- **Disclosure: a separate TF32-precision TensorRT engine also exists in the engine cache** (build_identity beginning `e2d3f91d053d`). It was generated as the TF32 negative-control engine during the Phase 6E correctness gate (see `correctness/correctness_tensorrt.json`, checks `E1` and `NC2`, which confirm the TF32 engine measurably differs from the IEEE FP32 engine). **It was not referenced or used by any Phase 6G controlled benchmark run** and is not published here.

- The historical schema-1.0 engine manifest (predating the `6ddd555` engine-identity fix) is likewise not published; it was never used by Phase 6G.

- No `.plan` engine binary is published in this directory. A compiled TensorRT engine is functionally equivalent to serialized model weights; only its SHA-256 and build metadata are recorded.

## Experimental methodology

| Dimension | Value |
|---|---|
| Batch sizes | 1 and 8 |
| Warmup / measured iterations | 100 / 1000 |
| Repeats | 5 independent process repeats per batch |
| Order | Batch order (B1/B8) alternates across repeats. TensorRT is the only backend under test in this experiment, so "alternating order" here means batch order, not backend order — pre-registered before any run (`docs/plans/phase-6-tensorrt.md` §16a) |
| Timing | CUDA events (primary) + synchronized host time (secondary), via `TrtCudaEventTimer` |
| Data movement | Device-resident input and output (`io_device_resident: true`); no host↔device copy inside the measured region |
| Excluded from measured latency | Engine build, engine deserialization, execution-context creation, and device buffer allocation — all performed before the warmup/measured loop begins |
| Measured region | `execute_async_v3` plus its stream synchronization, used for timing |
| Telemetry | `nvidia-smi` sampled at ~100 ms in a separate process — run-level context only |

## Headline results

Host-side series. Each cell is 5 independent runs of 1000 measured iterations.

| Batch | p50 (median of 5 runs) | p90 | p99 | Throughput |
|---|---|---|---|---|
| 1 | **3.240 ms** | 3.244 ms | 3.254 ms | 308.6 samples/s |
| 8 | **7.495 ms** | 7.582 ms | 7.654 ms | 1066.9 samples/s |

p90/p99 are the medians of the per-run values; percentiles are never pooled across runs.

**Stability:**

| Condition | Result |
|---|---|
| run-to-run spread | Batch 1: **0.034%** · Batch 8: **2.12%** (both ≤ 5% gate) |
| within-run drift | ≤ 2% in every run |
| host/device containment violations | 0 (across all measured samples, both cells) |
| thermal slowdown flags | none (no `HwThermal`/`SwThermal` in any run) |

## Batch-8 settling observation

**MEASURED OBSERVATION**, recorded as a diagnostic property, not a publication-blocking
failure.

Settle indices across the 5 batch-8 repeats: r1=160, r2=10, r3=730, r4=10, r5=130.
**3 of 5 repeats report `warmup_sufficient: false`** against the configured 100-iteration
warmup. Batch 1 settles at k=0 in all 5 repeats.

Full trajectory inspection (not the capped view in `analysis/controlled.json`) shows real,
measured oscillation/settling behavior in the affected repeats. **No causal explanation
was established.** GPU telemetry lacks per-iteration timestamps, so the stored evidence
cannot establish a fine-grained causal relationship between individual latency samples and
clock/power changes — nothing here identifies a cause, and none should be inferred from
this document.

**This does not invalidate the benchmark.** The behavior is already reflected in, not
hidden from, the reported spread (2.12%, well under the 5% gate); no run's drift exceeds
2%, and containment holds with 0 violations. Consistent with Phase 5B precedent (19 of 20
runs there also failed an equivalent criterion and were published as measured), this
project treats settle-index ≤ warmup as a diagnostic property to report in full, not a
publication-blocking gate.

## Telemetry limitations

Telemetry (`telemetry/*.csv`) is **run-level context**, sampled at approximately 100 ms by
`nvidia-smi` in a separate process. It is **not per-iteration aligned** — individual
measured samples cannot be matched to individual telemetry rows. Consequently, this
evidence **cannot establish precise causal relationships** between specific latency
samples and specific clock or power-state changes. Any association drawn from telemetry
in this document (or in `docs/plans/phase-6-tensorrt.md`) is descriptive only.

## Comparison limitation

**Phase 6G TensorRT numbers must not be treated as a perfectly apples-to-apples comparison
with the Phase 5A/5B PyTorch/ONNX Runtime results.** The methodologies differ in more than
backend choice: TensorRT here uses a single engine built with an opt-batch=8 optimization
profile shared across both measured batch sizes, which has no equivalent in the PyTorch/ORT
methodology. **No cross-phase speedup is calculated or advertised in this document.**

## What is here

| Path | Contents |
|---|---|
| `analysis/controlled.json` | Per-run trajectory, drift, containment and spread statistics — the source for every number quoted above |
| `runs/ctl-tensorrt-b<N>-r<rep>/<id>/` | `result.json`, `raw.json` (every warmup and measured sample), `metadata.json` (full environment, engine and backend settings), `summary.json` |
| `telemetry/` | `nvidia-smi` samples at ~100 ms, one CSV per run. **Phase 6G recorded one CSV per run and no separate `.err` file** (unlike Phase 5A/5B's telemetry captures, which paired each CSV with a `.err`) |
| `logs/` | Console log, one per run |
| `engine-provenance/` | The one FP32 engine manifest actually used by Phase 6G (JSON only — no `.plan` binary) |
| `correctness/` | Phase 6E numerical correctness gate evidence, copied verbatim (see below) |

## Correctness evidence

`correctness/correctness_tensorrt.json` and `correctness/outputs.npz` are the Phase 6E
correctness-gate evidence, included here **verbatim, unmodified**, generated independently
of the `TensorRtBackend` production code path (`analysis/gpu_validation/correctness_tensorrt.py`
drives the build layer and its own CUDA plumbing directly).

As recorded in that file: criterion is the unchanged Phase 4 tolerance
(`rtol=1e-4, atol_scale=1e-4`). All checks **PASS**, zero `failed_checks`:

| Check | Result |
|---|---|
| W1 — reference model and ONNX artifact share the same weights | PASS |
| E1 — IEEE FP32 engine cleared `BuilderFlag.TF32`; TF32 control engine did not | PASS |
| C1 — TensorRT FP32 vs CPU reference, 9 batch×seed cases | PASS |
| C2 — TensorRT FP32 vs PyTorch CUDA, 9 cases | PASS |
| C3 — TensorRT FP32 vs ONNX Runtime CUDA, 9 cases | PASS |
| NC1 — FP16 negative control correctly REJECTED by the tolerance | PASS |
| NC2 — TF32 engine measurably differs from the FP32 engine (TF32 effect is observable, not silently inert) | PASS |

Software versions recorded in that gate: TensorRT 11.3.0.99, PyTorch 2.14.0+cu130,
ONNX Runtime 1.30.0.

## Reproducing


## Integrity and security — what was actually checked

- All 64 evidence files in this directory were verified byte-identical (SHA-256) to their
  source in `results/phase6/controlled/`, `results/phase6/correctness/`, and the engine
  cache, both before and after this copy.
- No `.plan` engine binary is present anywhere in this directory (verified by search).
- A credential/API-key/private-key/token-oriented pattern scan found no matches in this
  directory.
- A dry run of `analysis/sanitize_evidence.py` against a throwaway copy of this directory
  redacted only the expected GPU UUID, hostname, and POSIX home-path patterns, and verified
  none of those patterns remained afterward. This directory itself was not rewritten by
  that tool — the published files are the raw, unredacted originals, matching Phase 5A/5B
  convention.
- The string `11.3.0.99` (the TensorRT version) was flagged by the sanitizer as
  IPv4-shaped; this is a known false-positive class documented in the tool itself and is
  not an address.

These checks establish what they check — no broader security guarantee is claimed.
