# Roadmap

Phases are built in order. Each one ends with working, tested code and an entry in
[`ENGINEERING_LOG.md`](../ENGINEERING_LOG.md). A phase is not "done" because code
exists — it is done when it has been executed and its output verified.

| Phase | Scope | Status |
|---|---|---|
| **0** | Repo, architecture, CLAUDE.md, README, dependency strategy | **Complete** |
| **1** | Hardware / driver / CUDA / framework detection | **Complete** (GPU-present path unverified — see [limitations](limitations.md)) |
| **2** | Benchmark core: timing engine, warmup, measurement loop, statistics, result schema | **Complete** (validated against a simulated backend; no real runtime yet) |
| 3 | PyTorch backend: model load, CUDA inference, result persistence | Next |
| 4 | ONNX Runtime backend: CUDA EP, explicit provider recording | Planned |
| 5 | TensorRT backend: ONNX→engine build, serialization, inference | Planned |
| 6 | GPU telemetry: sampled utilisation, VRAM, power, temperature | Planned |
| 7 | Experiment runner: YAML configs, batch/precision matrices | Planned |
| 8 | Comparison engine: baselines, speedups, derived metrics | Planned |
| 9 | Dashboard: experiment browser, filtering, graphs | Planned |
| 10 | LLM benchmarking: TensorRT-LLM, TTFT, tokens/sec, inter-token latency | Planned |
| 11 | Reports: HTML, Markdown, JSON, CSV export | Planned |
| 12 | Polish: CI, examples, reproducibility guide, analysis notebooks | Planned |

## Immediate next steps

1. **Model selection (Phase 3 prerequisite).** Research and document the first
   vision model and first small decoder-only LLM, with reasoning, in `docs/models.md`.
2. **Phase 3 — PyTorch backend.** The first real runtime. Most of it is
   developable on CPU (`device=cpu` exercises load/prepare/execute and the whole
   engine path); the CUDA-event timer and any CUDA measurement are not.
3. **Validate Phases 1-2 on real hardware.** Two things need a GPU to confirm: the
   NVML success path, and the CUDA-event timer against a synchronized host clock.

## Deliberately deferred

Not to be started until the measurement core is trustworthy: Triton Inference
Server, multi-GPU, distributed and concurrent-request benchmarking, serving
benchmarks, cost efficiency, Jetson/edge, Nsight integration, a public result
database or leaderboard, and the "Optimization Advisor".

A recommendation engine built on measurements nobody trusts is worse than no
recommendation engine.
