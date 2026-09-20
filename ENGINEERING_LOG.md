# Engineering log

A running record of what was built, what was measured, what was learned and what
remains unknown. Appended to at the end of every phase.

---

## 2026-09-18 — Phase 0 + Phase 1

### Environment

Inspected before writing any code, as the PRD requires. Full report:
[`docs/environment-report.md`](docs/environment-report.md).

```
OS        Windows 11 Pro 10.0.26200
CPU       Intel Core i7-8565U, 4C/8T
RAM       15.81 GiB
GPU       Intel UHD Graphics 620 (integrated) — no NVIDIA device
Driver    none
CUDA      none
Python    none installed at start; Python 3.12.10 installed during setup
Docker    CLI present, daemon not running
WSL2      enabled, no distributions
```

### Finding that shaped everything else

**The development machine has no NVIDIA GPU.** Confirmed three independent ways:
NVML fails to load, `nvidia-smi` absent from all standard locations, and Windows PnP
enumeration returns no `VEN_10DE` device.

This is a hardware-requirements decision, so it was escalated rather than assumed.
Decision taken: build the framework here, run real benchmarks on a cloud NVIDIA GPU
later. Consequence: **no GPU performance number can be produced on this machine**,
and the framework is built to refuse rather than estimate.

### Research (PRD §48 — do not trust model memory on NVIDIA packages)

Checked against PyPI on 2026-09-18. Three findings that contradict what would
otherwise have been assumed:

| Assumption | Reality |
|---|---|
| `pynvml` is the NVML binding | **Deprecated.** `nvidia-ml-py` (NVIDIA-authored, v13.610.43) is correct, and it installs the `pynvml` *module*. |
| `tensorrt` is a single wheel | Metapackage (v11.3.0.99) delegating to `tensorrt_cu12` / `tensorrt_cu13`. |
| PyTorch PyPI wheels are CUDA 12 | CUDA 13.0 is the PyPI default from the 2.11 series. |
| ONNX Runtime GPU targets CUDA 12 | v1.30.0 targets CUDA 13, requires Python ≥3.11. |

The first one would have been a silent adoption of a deprecated package. Recorded in
CLAUDE.md §12 so the next contributor does not repeat it.

### What was built

**Phase 0** — repo structure, `CLAUDE.md`, README, `CONTRIBUTING`, Apache-2.0,
architecture/methodology/roadmap/limitations/environment docs, three ADRs, CI
workflow, dependency strategy.

**Phase 1** — hardware detection:

- `capability.py` — pure compute-capability → architecture + precision matrix.
- `nvml.py` — NVML access that cannot raise; every field individually guarded.
- `host.py`, `frameworks.py`, `detect.py`, versioned pydantic schema in `types.py`.
- `gpu-bench hardware` and `gpu-bench doctor`, with CI-usable exit codes.

### Results

Real output on this machine:

```
Detection status   driver_unavailable
NVML error         NVML Shared Library Not Found
NVIDIA GPUs        0
Exit code          1
```

Verified:

| Check | Result |
|---|---|
| `pytest` | **67 passed** |
| `ruff check` | clean |
| `ruff format --check` | clean |
| `mypy --strict` | clean, 11 source files |
| `gpu-bench hardware` | executed; output above |
| Python 3.10.21 | 67 passed (separate venv, then removed) |
| Python 3.12.10 | 67 passed |

No performance measurements were produced, because no benchmark engine exists yet
and no GPU is present.

### Observations

1. **The "no GPU" machine is a useful asset, not only an obstacle.** Testing that a
   tool correctly reports a missing driver is awkward on a machine that has one.
   Here, the `driver_unavailable` path was exercised against reality rather than a mock.

2. **Distinguishing four failure modes was worth the effort.** `LIBRARY_UNAVAILABLE`,
   `DRIVER_UNAVAILABLE`, `NO_NVIDIA_DEVICE` and `ERROR` need completely different
   user responses. Most tools collapse these into a boolean and then give unhelpful
   advice. This machine hits `DRIVER_UNAVAILABLE`; a cloud VM with a GPU not yet
   attached would hit `NO_NVIDIA_DEVICE`.

3. **`None` vs `0` is a correctness issue, not style.** A GPU that does not report
   power draw and a GPU drawing 0 W must not be the same value, or a missing reading
   silently becomes a data point in an average.

4. **Separating "supported" from "tensor-core accelerated" per precision matters.**
   FP16 on Pascal is supported but has no tensor cores. A benchmark reporting "FP16
   gave no speedup" without that context invites a wrong conclusion about the format
   rather than the hardware.

### Hypothesis to test on real hardware

The NVML success path is the largest unverified surface in the codebase. It has been
exercised only against an injected fake module, so the real call signatures, the
bytes-vs-str return behaviour of the current binding, and the availability of
`nvmlDeviceGetNumGpuCores` are all unconfirmed. Expectation is that it works; that is
an expectation, not a result.

### Known limitations

Tracked in [`docs/limitations.md`](docs/limitations.md). The headline: everything
requiring an actual NVIDIA GPU is implemented-and-unit-tested but **not executed**.

### Next

1. **Phase 2 — benchmark core.** Timing engine, warmup, measurement loop, statistics,
   result schema. Fully buildable and testable without a GPU using a fake backend.
2. **Validate Phase 1 on real hardware** at the first opportunity — highest-priority
   unknown.
3. **Model selection** research for Phase 3, documented with reasoning.

---

## 2026-09-18 — Phase 2: benchmark core

### Goal

Build the measurement core — timing, warmup, statistics, result schema, config
validation and storage — plus a deterministic simulated backend so all of it can
be tested on a machine with no GPU.

### Environment

Unchanged from Phase 1: no NVIDIA GPU. See
[`docs/environment-report.md`](docs/environment-report.md).

### What was built

| Module | Responsibility |
|---|---|
| `core/timing.py` | `Timer` contract, `WallClockTimer`, `ScriptedTimer`, `TimingMechanism` |
| `core/statistics.py` | Percentiles, spread, sample validation, confidence flags |
| `core/backend.py` | The contract every backend implements |
| `core/config.py` | Validated YAML experiment configuration |
| `core/schema.py` | Versioned `BenchmarkResult` |
| `core/engine.py` | Phase structure and the measurement loop |
| `core/storage.py` | JSON result persistence |
| `core/provenance.py` | Git commit + dirty state |
| `core/errors.py` | Exception taxonomy mapped to result status |
| `backends/fake.py` | Deterministic simulated backend |
| `cli/run_cmd.py` | `gpu-bench run` |

### Results

| Check | Result |
|---|---|
| `pytest` | **229 passed** (was 67) |
| Coverage | 90% overall; Phase 2 core modules 93-100% |
| `ruff check` / `format` | clean (now including bandit `S` rules) |
| `mypy --strict` | clean, 23 source files |
| Python 3.10.21 | 229 passed |
| Python 3.12.10 | 229 passed |
| `gpu-bench run` end to end | Executed; sample committed to `examples/sample-output/` |

**Independent verification of every metric.** Rather than asserting that numpy
agrees with numpy, the stored statistics were re-derived from `raw.json` using the
stdlib `statistics` module and a hand-written percentile function. All agreed to
within 1e-9: mean, median, stdev, min, max, p50, p90, p95, p99. The unit tests
additionally use an analytically tractable dataset (1..100) where every expected
value is computed by hand from the definition.

### Observations

1. **Simulation nearly produced a fabricated number, and the design caught it.**
   The first end-to-end run reported observed throughput of 3,838,771 samples/sec.
   The cause: that metric divides *real* wall-clock loop time by the work done, but
   under the scripted timer the per-iteration durations are fabricated while the
   loop's wall time is real. The two are not commensurable. The fix was to emit
   nothing rather than emit a number nothing supports — which is the project's core
   rule applied to its own output. Worth noting that this class of bug (mixing two
   incommensurable time bases) is exactly what will bite on real hardware too.

2. **Making the backend own its timer paid off immediately.** It was adopted so the
   engine would not hard-code CUDA synchronization (ADR 0004), but the first
   consumer was the simulated backend returning a `ScriptedTimer` — which is what
   makes the entire test suite fast and deterministic. A design chosen for
   correctness turned out to be the one that made testing tractable.

3. **Rejecting NaN rather than dropping it is a correctness decision, not strictness.**
   Dropping one bad sample from 100 silently changes every percentile and hides the
   fact that the timing mechanism misbehaved. The engine records such a run as
   `failed` with no statistics.

4. **A single outlier moves p99 far less than intuition suggests.** With 99 samples
   at 10ms and one at 1000ms, the linear-interpolated p99 is 19.9ms — it nearly
   doubles, but lands nowhere near the outlier. So "p99 ≈ the slow case" is wrong in
   both directions at this sample count. This is now a documented test case and the
   concrete justification for the low-confidence flags.

5. **`ddof` and percentile method had to become part of the schema.** Sample vs
   population standard deviation differs by ~0.5% at n=100, and linear vs
   nearest-rank percentiles differ visibly at small n. Recording the convention costs
   two fields and removes a whole class of "why doesn't this match my other tool?".

### Hypotheses to test on real hardware

- **The synchronization hook placement.** `WallClockTimer` calls `synchronize()` at
  both `start()` and `stop()`. Against `time.sleep` this is trivially correct; against
  real asynchronous CUDA work it is an assumption. Expect it to hold; it is not a result.
- **Measurement-loop overhead.** The loop is written bare (locals pre-bound, list
  pre-sized, nothing else inside), but its cost relative to a real kernel is unknown.
  An empty-loop baseline should quantify it before any speedup is claimed.
- **Whether 10 warmup iterations is enough.** Warmup samples are now retained, so this
  is answerable the moment a real workload exists — plot latency against iteration
  index and find where it stabilises.

### Known limitations

The core has only ever driven the simulated backend. `TimingMechanism.CUDA_EVENT` is
declared but not implemented. Full list in
[`docs/limitations.md`](docs/limitations.md) §2b.

**No performance data exists in this repository.** The only results produced are
simulated and marked as such in five independent places.

### Next

1. **Model selection** — research and document the first vision model and first small
   decoder-only LLM, with reasoning, in `docs/models.md`.
2. **Phase 3 — PyTorch backend.** Much of it is developable on CPU (`device=cpu`
   exercises load/prepare/execute and the whole engine path); the CUDA-event timer and
   any CUDA measurement are not.
3. **Validate Phases 1-2 on real hardware** when a GPU is available.

---

## 2026-09-18 — Phase 3 preparation: model selection and backend design

No code changed. Research and design only; see [`docs/models.md`](docs/models.md)
and [`docs/plans/phase-3-pytorch-backend.md`](docs/plans/phase-3-pytorch-backend.md).

### Selected

- **Vision:** ResNet-50 (torchvision, v1.5), weights pinned by explicit name
  `IMAGENET1K_V2`, plus a download-free random-init mode for tests.
- **LLM:** Qwen3-1.7B @ `70d244cc`, with Qwen3-0.6B @ `c1899de2` as a secondary.

### Findings that would have produced wrong numbers

1. **PyTorch "FP32" is not IEEE FP32 by default.** `torch.backends.cudnn.allow_tf32`
   defaults to True (PyTorch 2.14 docs), so FP32 convolutions on Ampere+ run as TF32.
   For a convolution-dominated model like ResNet-50, a naive "FP32 vs FP16"
   comparison is really "TF32 vs FP16". The plan sets FP32 precision explicitly and
   records it.
2. **transformers v5 loads the saved dtype by default** (`dtype="auto"`), so a config
   saying FP32 would silently get BF16.
3. **Qwen3's checkpoint stores `lm_head` separately despite tied embeddings.**
   Verified from the safetensors index: shard 2 contains only `lm_head.weight`. Disk
   size (4.06 GB) overstates tied memory (~3.44 GB), and runtimes may disagree on
   deduplication, which would contaminate cross-backend VRAM comparisons.

All added to CLAUDE.md §12.

### Why Qwen3 over Qwen2.5, arithmetically

The per-token KV cache follows from published configs: 112 KiB (Qwen3-1.7B, 8 KV
heads) vs 28 KiB (Qwen2.5-1.5B, 2 KV heads). Only the former produces meaningful
memory pressure in sequence-length experiments on consumer cards. Qwen3-0.6B shares
Qwen3-1.7B's KV geometry exactly, making the pair a controlled comparison of
weight-driven vs KV-driven decode cost.

### Gaps found in the Phase 2 contract while designing

- Nothing can produce `status: unavailable`, so a missing device would be
  mis-recorded as `failed`.
- Error phase attribution is coarse: OOM is always tagged `execute`, even when raised
  in `prepare()`.

Both are scheduled as step 1 of Phase 3.

---

## 2026-09-18 — Phase 3: PyTorch backend (ResNet-50)

### Environment

Unchanged hardware: Intel Core i7-8565U, no NVIDIA GPU. Installed CPU builds
`torch 2.14.0+cpu` and `torchvision 0.29.0+cpu` (≈ 136 MiB download, 511.9 MB on disk).

**Everything below is one of: real CPU measurement, simulated, or CUDA code tested
only against fakes. No NVIDIA hardware measurement exists.**

### What was built

- Contract changes applied before the backend (schema 1.1): `unavailable` status,
  exact error phases, environment passed to `validate`, `execution_context()` hook,
  effective backend settings, `model_info` with weights SHA-256, optional secondary
  timing, validated `backend_options`, model registry.
- `PyTorchBackend` (CPU + CUDA), `CudaEventTimer`, `gpu-bench models list|fetch`.
- Closing fixes from review: gross-anomaly flag, power-source recording, effective
  input-shape recording, anomaly shown in the CLI.

### Findings from the installed torch (flag behaviour, verified without a GPU)

1. `torch.backends.cudnn.conv.fp32_precision` defaults to `"tf32"`, confirming the
   documented TF32 trap on this build.
2. After using the new `fp32_precision` API, reading legacy `cudnn.allow_tf32`
   raises `RuntimeError` — even with conv and RNN both `"ieee"`. The backend uses
   only the new API; restoring the snapshot makes legacy reads work again.

### Real CPU measurement (P0)

ResNet-50, IEEE FP32, batch 1, pinned `IMAGENET1K_V2` (SHA-256 `11ad3fa6…`), 10 warmup +
100 measured, on AC power, clean git tree at `1f153b9`:

```
p50 101.6 ms   mean 104.1 ms   stdev 18.8 ms   min 68.1   max 150.6   (p95/p99 low confidence)
throughput 9.61 samples/sec (both formulas agree)
```

All nine statistics re-derived from `raw.json` with the stdlib + a hand-written
percentile; all matched to 1e-9. **This is CPU inference on a laptop. It is not GPU
performance and is published only as methodology evidence.**

**Harness overhead (first real measurement):** loop wall time minus sum of samples =
0.375 ms / 100 iterations = 3.7 µs per iteration *between* timed windows.

**Observed non-stationarity:** block means (20 iterations) ≈ 89, 88, 118, 114, 111 ms.
The 10 warmup samples (80–99 ms) were all in the faster regime. *Possible
explanation:* turbo / power-limit transition. **Cause unconfirmed** — no clock or
power telemetry was recorded.

### A/B experiment: pinned vs random weights — INCONCLUSIVE

Design: alternating P1, R1, P2, R2 (10 + 100 iterations each).

| Run | p50 | mean | stdev | Notes |
|---|---|---|---|---|
| P1 | 119.1 | 6514.5 | 62698.1 | AC → battery mid-run; **iteration 79 = 627,219.9 ms** |
| R1 | 188.7 | 197.9 | 46.9 | battery, after resume |
| P2 | 223.8 | 310.4 | 268.0 | battery, after resume; 1.2–1.9 s outliers |
| R2 | 216.4 | 235.7 | 89.0 | battery, after resume |

Windows System log: power source change (`AcOnline=false`) at 05:00:20, S3 sleep
at 05:00:25, resume with the clock corrected from 05:00:27 to 05:10:54 — a 627 s gap
equal to P1's iteration 79. `perf_counter_ns` counted through the suspend.

Difference in mean per-run p50 (random − pinned) = +31.1 ms; the pinned arm's own
run-to-run range is 104.7 ms. With two runs per arm, one ordering, a power change and
a suspend, **no conclusion is justified** — not that weights matter, not that they
don't. No significance test was computed because the design cannot support one.
Evidence preserved in `results/published/2026-09-18-phase3-cpu-resnet50/`.

### Observations

1. **A clean-looking `ok` result can contain a 10-minute suspend.** Nothing in the
   Phase 2 schema flagged it; the mean became 6.5 s. Fixed with a loose anomaly flag
   (samples kept, never dropped). The flag deliberately does not catch P2's 5–8×
   outliers — it is a tripwire, not a stationarity test.
2. **The biggest confounder was invisible in the results.** Power source was only
   found via the OS event log. Now recorded in the environment.
3. **Iteration-count warmup is not evidence of steady state** on this machine. The
   same class of problem (boost/power/thermal limits) exists on GPUs.
4. Reviewing the implementation line by line found a real provenance gap: when the
   config omitted `input_shape`, the executed shape was recorded nowhere. Fixed.

### Deviations from the Phase 3 plan

- `torch.OutOfMemoryError` needed no special mapping: its class name is already
  `OutOfMemoryError`, and with exact phase tagging the generic path records it
  correctly. `execute()` stays a single line.
- Weights are fetched in `validate()` (untimed), so a first-run download cannot
  inflate `model_load_ms`.
- Two additions not in the plan, both driven by real data: the anomaly flag and
  power-source recording.

### What remains unverified (needs NVIDIA hardware)

NVML success path; PyTorch CUDA validation against a real driver; `CudaEventTimer`
placement for real asynchronous work; whether `"ieee"` really prevents TF32 kernel
selection; `cudnn.benchmark` warmup needs; real OOM; the legacy-TF32 branch
(torch < 2.9, untested anywhere).

### Next

Awaiting approval for Phase 4. See the Phase 3 report for the recommendation.

---

## 2026-09-19 — Phase 4: ONNX export and ONNX Runtime backend

### Environment

Same laptop, no NVIDIA GPU, on AC power for the evidence runs. Installed onnx 1.23.0,
onnxscript 0.7.2, onnxruntime 1.30.0 (CPU); onnxruntime-gpu 1.30.0 in a separate,
disposable venv for fallback testing.

### Findings that would have produced wrong results

1. **ORT silently substitutes CPU for a requested CUDA EP.** With the real
   onnxruntime-gpu 1.30 and no CUDA libraries, `get_available_providers()` *listed*
   the CUDA EP, and a session requested with only that EP was created and ran on CPU.
   Only a stderr warning. The CPU package does the same with a `UserWarning`. A naive
   backend would have produced a "CUDA" result from the CPU.
2. **`disable_cpu_ep_fallback` cannot tell load failure from partial placement** (same
   error), so the backend checks `session.get_providers()` and measures placement.
3. **ORT's CUDA EP enables TF32 by default** (`use_tf32=1`), mirroring PyTorch.
4. **The exporter writes weights to a separate `.onnx.data` file by default**, so a hash
   of the `.onnx` alone would not identify the model. Fixed with `external_data=False`.
5. **The exporter's default is a static batch**; batch 4 was rejected by ORT.
6. **Top-1 agreement alone would have passed an FP16 execution** (the negative control
   kept top-1 = 1.0 while failing elementwise by 14×).

### Correctness (the headline)

Pinned ResNet-50, PyTorch FP32 vs ORT CPU EP, identical inputs, 3 batches × 3 seeds:
**9/9 pass**, max |Δ| 1.7e-6 – 3.1e-6 on logits of magnitude ≈ 5, at most 0.44 % of the
pre-registered allowance, top-1 and top-5 100 %. **FP16 negative control rejected**
(687/1000 out of tolerance, 14.1× the bound). Two separate runs gave identical errors.
The stored verdict was re-derived from `outputs.npz` by a script that imports nothing
from the tool: 0 disagreements.

### CPU benchmark observation (not a ranking)

ORT CPU EP ResNet-50, batch 1, all 58 nodes on the CPU EP: the run completed and was
labelled. Its 20-iteration block means (46.4 → 57.1 ms) drifted like the Phase 3 runs.
**Local CPU measurements; the environment is known to be non-stationary and these
measurements are not suitable for backend performance ranking.** It is also not
comparable with the PyTorch run: different input generators, thread counts, days, no
interleaving, and a graph fused from 122 to 58 nodes.

### Bugs found in my own code during Phase 4

- `Δ` in a CLI table header crashed a cp1252 console after the evidence was saved, so a
  passing check exited 1. CliRunner (UTF-8) could not see it. A test now scans all CLI
  string literals, and it fails on the original header.
- Rich markup deleted `[torch]` / `[onnx-export]` from error messages on screen. Now escaped,
  with a test that fails without the fix.

### Deviations from the plan

- `disable_cpu_ep_fallback` is not used (finding 2).
- Node placement is measured on a separate profiling session with identical options:
  the benchmark session is never profiled.
- The PyTorch backend's input generator was **not** unified with ORT's (numpy), so that
  Phase 3 stays untouched. It must be unified before any cross-backend latency comparison.

### Unverified until NVIDIA hardware

The ORT CUDA EP itself, full placement of ResNet-50 on it, ORT's end-of-Run stream
synchronization (the CUDA timing relies on it), IOBinding, the effect of `use_tf32`,
and correctness on CUDA.

### Next

Awaiting approval for Phase 5.

---

## 2026-09-20 — Phase 5A: first validation on NVIDIA hardware (NVIDIA L4)

Ran the GPU validation runbook end to end on a Google Cloud `g2-standard-4` with one
NVIDIA L4 (driver 580.159.04, CUDA driver max 13.0, torch 2.14.0+cu130,
onnxruntime-gpu 1.30.0, Ubuntu 24.04.5, Python 3.12.3). Evidence, unedited, in
`results/published/2026-09-20-phase5a-l4/`.

### What the hardware confirmed

Every pre-registered check passed: NVML against `nvidia-smi` (19 fields), the PyTorch
CUDA path, the CUDA-event timer with its negative control, ORT's CUDA EP with proof of
execution and of its end-of-`Run` synchronization, and correctness in both directions.
The 20-run controlled benchmark produced the project's first GPU latency numbers.

### Bugs the hardware exposed (none of which fakes could have caught)

1. **`multiprocessor_count` held CUDA cores, not SMs** — NVML reported 7424 where the
   CUDA runtime reports 58 SMs (7424 = 58 × 128). `nvmlDeviceGetNumGpuCores` is "the
   device's core count"; NVML has no SM-count call. Renamed `cuda_core_count`;
   environment schema 1.1 → 1.2.
2. **The ORT CUDA EP depended on torch's import side effects.** Without torch imported
   first, ORT built a session that reported the CUDA EP active and then failed at the
   first Conv: `dlopen failed for libcudnn.so`. `gpu-bench` only worked because it
   imports torch while detecting the environment. `create_session` now calls
   `onnxruntime.preload_dlls()` and records the outcome. Proven three ways on the L4:
   torch-first passes, ORT-alone fails, ORT-alone with preload passes.
3. **A test encoded "ORT cannot load CUDA here."** It now skips where the EP genuinely
   loads, decided by creating a session and asking it — never by the provider list.

### Findings that change how results must be read

- **The L4's 72 W cap binds.** ORT at batch 1 runs power-capped at 1665–1755 MHz while
  PyTorch at batch 1 never reaches the cap and holds 2040 MHz. The driver flagged
  `SwPowerCap` on essentially every busy sample in three of four cells.
- **Warmup adequacy is backend-specific.** PyTorch's first 10 iterations sit within
  0.6–2.0% of steady state; ORT's first 10 are 7.1–8.6% *faster*, as it starts at the
  boost clock and is then capped down. The 10-iteration default is unsafe for ORT here.
- **cuDNN autotuning happens in the untimed sanity pass, not during warmup** as
  `PyTorchOptions.cudnn_benchmark`'s description claims: `prepare_inputs_ms` rises
  363–371 → 585–595 ms with it on, buying ≈1.5–2.5% steady-state latency.
- **TF32 is detectable by the Phase 4 tolerance** (11.8–14.7× over) — but **top-1
  agreement stayed 100%** for TF32 and FP16 alike, so top-1 alone detects neither.
- The GPU warmed 55 → 80 °C across the session; later repeats are slightly slower.

### Deviations from the plan

- The pre-registered warmup rule ("if settle k > 50, use 2k") was inapplicable: the
  ±2% block-median statistic proved jitter-dominated and returned `None` for some runs.
  The deviation and the decision to keep `warmup_iterations = 100` were written to
  `NOTES.md` **before** the controlled runs. No config was edited after seeing data.
- The input-generator mismatch between backends is still not fixed, so it qualifies the
  comparison rather than being eliminated.

### Next

TensorRT is still untouched, as is the LLM path: `gpu-bench models list` has only
ResNet-50, so no Qwen3 metric (TTFT, inter-token latency, tokens/sec) can exist yet.
