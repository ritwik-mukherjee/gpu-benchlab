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
