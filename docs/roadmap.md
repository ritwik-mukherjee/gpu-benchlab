# Roadmap

Phases are built in order. Each one ends with working, tested code and an entry in
[`ENGINEERING_LOG.md`](../ENGINEERING_LOG.md). A phase is not "done" because code
exists — it is done when it has been executed and its output verified.

| Phase | Scope | Status |
|---|---|---|
| **0** | Repo, architecture, CLAUDE.md, README, dependency strategy | **Complete** |
| **1** | Hardware / driver / CUDA / framework detection | **Complete** — GPU-present path validated on an NVIDIA L4 (2026-09-20), NVML checked field by field against `nvidia-smi` |
| **2** | Benchmark core: timing engine, warmup, measurement loop, statistics, result schema | **Complete** (validated against a simulated backend; no real runtime yet) |
| **3** | PyTorch backend: model load, CUDA inference, result persistence | **Complete** — CPU and CUDA paths both executed on real hardware; CUDA-event timing validated on an L4 |
| **4** | ONNX Runtime backend: CUDA EP, explicit provider recording | **Complete** — export reproducible; CPU EP and CUDA EP both executed and verified against PyTorch on real hardware |
| **5A** | First NVIDIA hardware validation (NVML, CUDA events, correctness, ORT CUDA EP, telemetry, controlled benchmark) | **Complete** — NVIDIA L4, 2026-09-20; evidence in `results/published/2026-09-20-phase5a-l4/` |
| **5B** | Controlled PyTorch vs ONNX Runtime rerun under unified inputs | **Complete** — NVIDIA L4, 2026-09-20; evidence in `results/published/2026-09-20-phase5b-l4-controlled-inputs/` |
| **6** | TensorRT backend: ONNX→engine build, serialization, inference | **Implemented, not validated** — backend, engine-build layer, correctness gate, controlled configs and 37 tests are committed (`3ad58be`…`077e036`). Every test runs against a mocked TensorRT; **no TensorRT library has ever been executed by this project**. Phase 6A (real-environment probe) and Phase 6G (controlled benchmark) are outstanding |
| 7 | Experiment runner: YAML configs, batch/precision matrices; in-process GPU telemetry sampling during a run (utilisation, VRAM, power, temperature) | Planned |
| 8 | Comparison engine: baselines, speedups, derived metrics | Planned |
| 9 | Dashboard: experiment browser, filtering, graphs | Planned |
| 10 | LLM benchmarking: TensorRT-LLM, TTFT, tokens/sec, inter-token latency | Planned |
| 11 | Reports: HTML, Markdown, JSON, CSV export | Planned |
| 12 | Polish: CI, examples, reproducibility guide, analysis notebooks | Planned |

## Immediate next steps

1. **Phase 6A — TensorRT environment probe (read-only).** On the L4 VM, establish
   which TensorRT release installs against driver 580.159.04 / CUDA 13.0, which shared
   objects it actually loads, and whether `trtexec` exists on the pip route. Nothing is
   installed into the benchmark venv, and no CUDA/driver change is made, until this is
   reported. The dependency pins in `pyproject.toml` are **declared, not verified**.
2. **Phase 6B–6F — execute the committed TensorRT path.** Parse, build, deserialize,
   run once, then the correctness gate against PyTorch and ONNX Runtime. The code exists;
   none of it has met a real TensorRT library. A failure at any gate stops the phase.
3. **Phase 6G — controlled benchmark.** Only after every gate passes, and only with the
   Phase 5B methodology unchanged, into a new evidence directory. Phase 5A and 5B
   evidence is never touched.

**On phase numbering.** TensorRT is **Phase 6**: that is what the commit messages, the
plan (`docs/plans/phase-6-tensorrt.md`), the planned evidence directory name and the
`p6-` experiment names in `analysis/gpu_validation/configs/` already say. In-process
telemetry sampling, which this table previously listed as Phase 6, moves into the
Phase 7 row rather than renumbering Phases 7–12 — those numbers are referenced from
ADRs and earlier plans, which are historical records and are not rewritten.

## Deliberately deferred

Not to be started until the measurement core is trustworthy: Triton Inference
Server, multi-GPU, distributed and concurrent-request benchmarking, serving
benchmarks, cost efficiency, Jetson/edge, Nsight integration, a public result
database or leaderboard, and the "Optimization Advisor".

A recommendation engine built on measurements nobody trusts is worse than no
recommendation engine.
