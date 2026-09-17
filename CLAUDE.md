# GPU Inference Benchmarking Lab — Engineering Rules

This file is the contract for anyone (human or AI) writing code in this repository.

---

## 0. Core principle

**REAL MEASUREMENTS > CLAIMS.**

Never fabricate benchmark results. Every performance number that appears in this
repository — in code, docs, README, dashboard, report or commit message — must be
traceable to an actual benchmark execution whose raw samples are stored on disk.

Specifically, **never**:

- hard-code a speedup, latency or throughput value;
- write an illustrative number that could be mistaken for a measurement;
- present theoretical or vendor-published performance as measured performance;
- fill a chart, table or README with plausible-looking placeholder data;
- claim a feature works when it has not been executed on real hardware.

If a benchmark cannot run, it must be **recorded** with an explicit terminal status
(`unsupported`, `unavailable`, `failed`, `skipped`) and a machine-readable reason.
Silence and omission are both bugs.

Placeholders in documentation must be *obviously* non-numeric — write `<not yet measured>`
or `TBD`, never `~5x` or `12.4 ms`.

---

## 1. Measurement validity

- **Never** wrap `time.time()` around a GPU call. GPU work is asynchronous.
  Use `time.perf_counter_ns()` with explicit device synchronization, and CUDA events
  where the backend exposes them. Document which mechanism each backend uses.
- Always separate these phases and never let one leak into another:
  `cold start` → `model load` → `engine build` → `warmup` → `steady-state` → `end-to-end`.
  A TensorRT engine build in particular must **never** be counted as inference latency.
- Always store **raw per-iteration samples**, not just aggregates. Aggregates are derived.
- Never report "throughput" without naming its unit (`samples/sec`, `tokens/sec`, `req/sec`).
- Keep the measurement loop clean: no logging, no allocation, no telemetry polling,
  no Python-level branching that can be hoisted out of it.
- Prefer reporting variability (std dev, percentile spread, repeat count) over a single
  number presented as universal truth.

## 2. Comparisons

- Every optimization experiment needs an explicit, recorded **baseline**.
- Derived metrics (`speedup`, `% latency reduction`, …) are computed at read time from
  stored raw measurements. They are never authored by hand and never cached into docs.
- Only compare configurations that differ in the dimension under test. Comparing
  unrelated configurations and calling the ratio a speedup is a correctness bug.

## 3. Hardware and environment

- Never hard-code a GPU, compute capability, VRAM size, CUDA version or driver version.
- Detect capability; do not assume it. Every GPU does not support every precision.
- Every result must record enough hardware + software + config + git state to reproduce it.
- Code must run — and degrade honestly — on a machine with **no NVIDIA GPU at all**.
  That is the current primary development environment; see `docs/environment-report.md`.

## 4. Errors are first-class outcomes

- An OOM, an unsupported precision or a failed engine build is a **result**, not a crash.
- Record error type, message, stack trace where useful, config and timestamp.
- One failed experiment must not abort an independent experiment in the same suite.
- Distinguish clearly, in the schema and in the UI:
  - `unsupported` — the configuration is invalid for this hardware/backend/model;
  - `failed` — the configuration is valid but execution errored;
  - `unavailable` — a required dependency or device is not present;
  - `skipped` — deliberately not run.

## 5. Backends

- Keep backend implementations modular and behind a common benchmark interface, but
  **do not** force incompatible APIs into an artificial abstraction. LLM generation and
  vision forward-passes are different shapes; let them be.
- Be precise about what is actually executing. Never label something "ONNX Runtime GPU"
  when the active execution provider is TensorRT. Record the execution provider.
- Record backend version and, where relevant, build flags in every result.

## 6. Code quality

- Type hints everywhere; `mypy --strict` must pass.
- `ruff check` and `ruff format` must pass.
- No unnecessary dependencies. Core install must stay CPU-installable and lightweight;
  every GPU runtime is an optional extra.
- Prefer simple architecture over premature abstraction. No microservices, no cloud
  infrastructure, no database until a database is genuinely required.
- Public functions get docstrings that say what is measured and how.

## 7. Testing

- Unit tests must not require a GPU. Use fixtures and fakes for hardware.
- Hardware-dependent tests are marked `@pytest.mark.gpu` and skip cleanly when absent.
- Statistics, schema, config parsing and comparison maths must have direct unit tests
  with known-answer inputs.
- **Never say something works without testing it.** If it cannot be tested in the current
  environment, say so explicitly and record the limitation in `docs/limitations.md`.

## 8. Development workflow

Before implementing a major feature:

1. Explain the proposed approach.
2. Identify the relevant files.
3. Identify the risks.
4. Implement.
5. Run tests.
6. Report what changed.
7. **Report what could not be tested, and why.**

After each phase, append to `ENGINEERING_LOG.md`.

## 9. Git

Small, logical, conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`,
`chore:`). One enormous commit is a smell.

## 10. Documentation

- Do not write a claim the repository cannot substantiate.
- Document assumptions, defaults and the reasoning behind them — especially warmup counts,
  iteration counts and timing mechanism choices.
- Record notable technical decisions as ADRs in `docs/decisions/`.

## 11. Security

- Never execute downloaded model code without an explicit warning and opt-in.
- No shell interpolation of untrusted strings.
- Never require or store secrets; never write credentials into benchmark results.

## 12. NVIDIA-specific research rule

NVIDIA's stack moves fast and package names change. Before implementing any NVIDIA
integration, **check current official documentation rather than relying on memory.**

Already-verified examples of things memory gets wrong (verified 2026-09-18):

| Trap | Reality |
|---|---|
| `pynvml` | **Deprecated.** Use `nvidia-ml-py` (NVIDIA-authored). It still provides the `pynvml` *module* namespace. |
| `tensorrt` is one wheel | Split into `tensorrt-cu12` / `tensorrt-cu13` variants behind a metapackage. |
| PyTorch CUDA wheels on PyPI | CUDA 13.0 is the PyPI default from the 2.11 series; other CUDA builds live on `download.pytorch.org/whl/cuXXX`. |

Add to this table whenever you catch another one.
