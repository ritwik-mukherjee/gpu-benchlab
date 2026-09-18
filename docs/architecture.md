# Architecture

## 1. Design goals, in priority order

1. **Measurement validity.** A number that is wrong is worse than no number.
2. **Reproducibility.** A result nobody can reproduce is an anecdote.
3. **Honest failure.** Unsupported and failed configurations are data.
4. **Extensibility without ceremony.** New backends should be easy; the framework
   should not be a framework for its own sake.

Explicit non-goals for v0: distributed execution, a serving layer, a database, a
plugin system, cloud infrastructure. Each of these can be added later without a
rewrite; none of them is needed to produce a trustworthy latency number today.

## 2. Layers

```
                   Experiment configuration (YAML, validated)
                                    |
                                    v
                            Compatibility matrix
                    (GPU x CUDA x driver x backend x model x precision)
                                    |
                     valid? ------------------ invalid -> result(status=unsupported)
                        |
                        v
                          Benchmark engine
        cold start -> load -> [engine build] -> warmup -> timed loop
                                    |
        +---------------+-----------+-----------+----------------+
        v               v                       v                v
    PyTorch        ONNX Runtime            TensorRT        TensorRT-LLM
   (CUDA events)  (EP recorded)        (build/load/infer  (TTFT, ITL,
                                        measured apart)    tokens/sec)
                                    |
                            GPU telemetry (NVML)
                     sampled in a thread, never in the loop
                                    |
                              Results store
                     results/<experiment-id>/{metadata,raw,summary}.json
                                    |
                    +---------------+---------------+
                    v                               v
              Comparison engine                Report / dashboard
       (speedups derived at read time)      (renders stored data only)
```

## 3. Package layout

```
src/gpu_benchlab/
    hardware/      Phase 1 — GPU, driver, CUDA, framework and capability detection
        capability.py   pure compute-capability -> architecture + precision matrix
        nvml.py         NVML access, degrades honestly when absent
        host.py         OS / CPU / RAM / Python
        frameworks.py   which runtimes are installed and CUDA-capable
        types.py        versioned pydantic schema
        detect.py       orchestrator -> EnvironmentReport
    core/          Phase 2 — the benchmark core
        timing.py       Timer contract; WallClockTimer + ScriptedTimer
        statistics.py   percentiles, spread, confidence flags
        backend.py      the contract every backend implements
        config.py       validated experiment configuration (YAML)
        schema.py       versioned BenchmarkResult
        engine.py       phase structure and the measurement loop
        storage.py      JSON result persistence
        provenance.py   git commit + dirty state
        errors.py       exception taxonomy -> result status
    backends/      one module per inference runtime
        fake.py         simulated backend (framework testing only)
        pytorch.py      eager PyTorch, CPU + CUDA; CudaEventTimer (Phase 3)
        (onnxruntime, tensorrt, tensorrt_llm: Phases 4-5, 10)
    models/        Phase 3 — model registry and pinned-weights verification
        registry.py     name -> architecture, expected params, shapes, pinned weights
        weights.py      stdlib download + SHA-256 verification
    telemetry/     Phase 6 — NVML sampling during a run
    experiments/   Phase 7 — config parsing, matrix expansion, runner
    compare/       Phase 8 — baselines and derived metrics
    report/        Phase 11 — HTML / Markdown / JSON output
    cli/           the `gpu-bench` command
```

## 4. Key decisions

### 4.1 The environment report is part of the result, not a side file

`EnvironmentReport` is embedded in every stored result. A latency figure without
the driver version, GPU, precision and framework build that produced it cannot be
compared to anything. Making it a separate file makes it optional; making it a
field makes it mandatory.

### 4.2 Absence is a state, never an exception

`NVMLProbe.probe()` cannot raise. Every NVML field is individually guarded and
degrades to `None`. The distinction between:

- `LIBRARY_UNAVAILABLE` — the Python binding is missing,
- `DRIVER_UNAVAILABLE` — no NVIDIA driver on the machine,
- `NO_NVIDIA_DEVICE` — driver present, zero GPUs,
- `ERROR` — something else,

is preserved because each one requires different user action. Collapsing them into
a single "no GPU" boolean is the most common defect in tools of this kind.

### 4.3 `None` never means zero

A GPU that does not report power draw and a GPU drawing 0 W are different facts.
Every optional metric is nullable rather than defaulting to `0`, so that a missing
reading can never be averaged into a statistic as if it were a measurement.

### 4.4 Precision support is a matrix, not a boolean

`capability.py` reports, per precision: *supported* and *tensor-core accelerated*,
separately, with a human-readable reason. FP16 on Pascal is supported but has no
tensor cores, and a benchmark that reports "FP16 gave no speedup" without that
context is misleading. The module is pure, so the whole matrix is unit-tested on a
machine with no GPU.

### 4.5 Core install is CPU-only

Every inference runtime is an optional extra. This is not just hygiene: it means
the framework, schema, comparison maths and test suite can be developed and
verified on hardware that cannot run a single benchmark, and it lets CI run the
full non-GPU suite on a standard runner.

### 4.6 Storage starts as JSON on disk

One directory per experiment, containing `metadata.json`, `raw.json` (every
individual latency sample) and `summary.json` (derived statistics). Reasons:
inspectable without tooling, diffable, trivially portable, and no schema migration
story needed on day one. A SQLite index over these files can be added for query
performance when the result count justifies it — the JSON stays canonical.
Parquet becomes interesting only once raw sample volume is large.

### 4.7 Derived metrics are computed at read time

Speedups and percentage deltas are never stored. They are recomputed from raw
samples whenever they are displayed, so a stored result can never disagree with
the numbers shown next to it, and a methodology fix retroactively corrects every
comparison.

### 4.8 The backend owns its timer; the engine never synchronizes

The engine drives the phase structure but calls no CUDA API. Each backend returns
its own `Timer` from `make_timer()`, because only the backend knows whether its
work needs a device synchronization, a stream synchronization, CUDA events, or
nothing at all. Hard-coding `torch.cuda.synchronize()` into the engine would be
wrong for ONNX Runtime (which synchronizes internally), wrong for TensorRT (which
wants a stream sync), wrong for CPU backends, and would put a CUDA dependency in
code that must import on a GPU-less machine.

The mechanism used is recorded on every result, so results measured on different
bases are never silently compared.

### 4.9 Simulation is contagious and multiply marked

The simulated backend exists to test the framework, and the one thing this project
must never do is let its output pass as a measurement. So `is_simulated` propagates
from the backend descriptor to the top level of the result, the timing mechanism
is `scripted`, a warning note travels with the result, the storage directory is
`sim-` prefixed, and all four stored JSON files carry the flag independently.

`BenchmarkResult.is_real_measurement` is the single check a consumer should use.

## 5. Result schema (implemented — see `core/schema.py`)

```jsonc
{
  "schema_version": "1.0",
  "experiment_id": "...",
  "status": "ok | unsupported | failed | unavailable | skipped",
  "environment": { /* EnvironmentReport, see hardware/types.py */ },
  "model":       { "name": ..., "revision": ..., "format": ... },
  "backend":     { "name": ..., "version": ..., "execution_provider": ... },
  "configuration": {
    "precision": ..., "batch_size": ..., "input_shape": ...,
    "warmup_iterations": ..., "measurement_iterations": ...
  },
  "phases": {
    "model_load_ms": ..., "engine_build_ms": ..., "warmup_ms": ...
  },
  "metrics": { "latency_ms": { "p50": ..., "p95": ... }, "throughput": {...} },
  "raw_samples": { "latency_ms": [ ... ] },
  "telemetry": { "samples": [...], "peak": {...}, "mean": {...} },
  "errors": [ { "type": ..., "message": ..., "traceback": ... } ]
}
```

`throughput` always carries its unit. `phases` keeps engine build time structurally
separate from inference latency so it cannot leak in by accident.

## 6. Extension points

A new backend implements load / prepare / execute / synchronize / teardown and
declares its own capability constraints. It is *not* required to fit the vision
model shape: TensorRT-LLM reports TTFT and inter-token latency, which have no
meaning for a ResNet forward pass, and forcing both through one interface would
damage both.
