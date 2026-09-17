# Benchmark methodology

> **Status:** this document specifies the methodology the benchmark engine (Phase 2)
> will implement. Phase 1 — environment detection — is implemented and tested.
> Nothing here has produced a measurement yet, because no benchmark engine exists yet.

The purpose of this document is to let a sceptical reader decide whether to believe
any number this project eventually publishes.

---

## 1. Why `time.time()` is wrong

CUDA kernel launches are asynchronous. This is the canonical mistake:

```python
start = time.time()
output = model(input_tensor)  # returns almost immediately; the GPU has not finished
elapsed = time.time() - start  # measures launch overhead, not execution
```

Two separate corrections are required:

1. **Use a monotonic clock.** `time.time()` is a wall clock subject to NTP
   adjustment and, on some platforms, coarse resolution. Use
   `time.perf_counter_ns()`.
2. **Synchronize.** The host must wait for the device before stopping the clock.

Even done correctly, host-side timing includes launch overhead and synchronization
cost. Where a backend exposes device-side timers (CUDA events), we record both and
report them separately rather than picking a favourite.

## 2. Timing mechanism per backend

| Backend | Primary | Secondary | Notes |
|---|---|---|---|
| PyTorch | `torch.cuda.Event` elapsed time | `perf_counter_ns` + `torch.cuda.synchronize()` | Events measure device time; the wall clock captures end-to-end cost including Python overhead. |
| ONNX Runtime | `perf_counter_ns` around `run()` | ORT profiling where enabled | ORT synchronizes internally before returning. The **execution provider is recorded** — a CUDA-EP result and a TensorRT-EP result are not interchangeable. |
| TensorRT | CUDA events on the execution stream | `perf_counter_ns` + stream sync | Engine build and engine load are measured separately and never included. |
| TensorRT-LLM | per-token timestamps | — | TTFT and inter-token latency require timestamps inside generation, not around it. |

Whichever mechanism is used, the result records **which one**, so two results are
never silently compared across different timing bases.

## 3. Phase separation

These are measured as distinct quantities and stored in separate fields:

| Phase | What it covers |
|---|---|
| Cold start | Process start, CUDA context creation, library load. |
| Model load | Reading weights and constructing the runtime object. |
| Engine build | TensorRT engine compilation. Can take minutes. **Never** counted as inference. |
| Engine load | Deserializing a prebuilt engine. |
| Warmup | Iterations discarded before measurement (see §4). |
| Steady state | The reported inference latency. |
| End-to-end | Steady state plus pre/post-processing, where relevant. |

The single most important rule: **a TensorRT engine build must never appear inside
inference latency.** A tool that gets this wrong makes TensorRT look catastrophically
slow on the first run and impossibly fast afterwards.

## 4. Warmup

The first iterations of any GPU workload are not representative. They include
CUDA context initialization, kernel autotuning and algorithm selection (notably
cuDNN benchmark mode), lazy memory allocation and allocator cache population,
JIT compilation, and clock ramp from idle.

Defaults will be **10 warmup iterations** and **100 measured iterations**, both
configurable. The rationale:

- 10 is generally enough to get past allocator and autotune effects for small
  vision models at small batch sizes.
- 100 gives a usable p95 and a meaningful p99 is *not* claimed from it — see §6.
- Both numbers are recorded in the result, so a reader can judge them.

**These defaults are a starting hypothesis, not a validated constant.** Once the
engine exists, the correct warmup count will be determined empirically per backend
by plotting latency against iteration index and identifying where it stabilises.
That analysis will live in `analysis/` and this section will be updated with the
measured answer. Until then, treat the defaults as unvalidated.

The runner is designed so that reporting cold-start numbers as steady-state
requires deliberately setting `warmup_iterations: 0`, which is recorded in the
result and surfaced in the report.

## 5. Statistics

Raw per-iteration samples are **always** stored. Aggregates are derived from them,
never stored in place of them. This means percentiles can be recomputed, outliers
inspected, distributions plotted and methodology errors corrected retroactively.

Reported per run: min, max, mean, median, p50, p90, p95, p99, standard deviation,
sample count.

Percentiles use linear interpolation between order statistics (`numpy.percentile`
default). The method is stated because percentile definitions differ between tools
and the difference is visible at small sample counts.

## 6. What a percentile from N samples can honestly support

With 100 samples, the p99 is essentially the maximum — a single outlier defines it.
It is reported for completeness but should not be treated as a stable statistic.
Guidance the tooling will enforce in reports:

| Statistic | Minimum samples for a stable estimate |
|---|---|
| p50 | ~30 |
| p95 | ~200 |
| p99 | ~1000 |

Where sample count is insufficient, the report will mark the statistic as
low-confidence rather than printing it as though it were solid.

## 7. Variability and repeats

A single run on a thermally throttling laptop GPU is not the same measurement as a
single run on a datacenter card with a fixed clock. Results therefore record GPU
clocks, temperature and power alongside latency, and the report presents spread,
not just a central value.

Where a configuration is repeated, each repeat is stored as its own result with a
shared group id. Repeats are never averaged into a single stored number, because
that would destroy the between-run variance that is the entire point of repeating.

## 8. Telemetry without distorting the measurement

NVML queries are not free and must not run inside the timed loop. Telemetry is
sampled from a **separate thread** at a configurable interval, with before/after
snapshots for VRAM. The report distinguishes:

- **instantaneous** — a single reading (e.g. clocks at detection time),
- **mean** — time-averaged across the run,
- **peak** — maximum observed.

Sampling interval is recorded, because a 100 ms sampler will under-report the peak
utilisation of a 5 ms kernel, and a reader needs to know that.

## 9. Framework overhead

The harness itself must not distort what it measures. Inside the measurement loop
there is no logging, no allocation, no telemetry polling and no avoidable Python
branching. Input tensors are allocated and moved to the device **before** the loop,
so host-to-device transfer is not silently counted as inference — unless the
experiment explicitly asks for end-to-end measurement, in which case it is recorded
as such.

Python interpreter overhead is itself measurable and will be characterised (empty-loop
baseline) once the engine exists, so it can be reported rather than assumed negligible.

## 10. Comparisons

A speedup is only meaningful between configurations that differ in exactly the
dimension under test. The comparison engine requires an explicit baseline and
refuses to compare results whose model, batch size or input shape differ from the
baseline unless that dimension is the one being compared.

All derived metrics are computed from stored raw samples at read time:

```
speedup = baseline_latency_p50 / candidate_latency_p50
```

No derived metric is ever hard-coded, cached into documentation, or written by hand.

## 11. Known threats to validity

Tracked openly, and to be quantified rather than hand-waved once measurement is possible:

- **Thermal throttling** on laptop and small-form-factor GPUs across a long suite.
- **Clock boost variability** — results are clock-dependent; clocks are recorded.
- **Other GPU consumers** — a desktop compositor shares the device. VRAM and
  utilisation are captured before the run so contamination is visible.
- **Driver/framework version drift** between machines — hence full version capture.
- **Allocator caching** — a later experiment can inherit a warmer allocator than an
  earlier one; process isolation per experiment is the mitigation under consideration.
- **First-run file cache** effects on model load time.
