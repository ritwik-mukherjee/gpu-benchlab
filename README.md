# GPU Inference Benchmarking Lab

**Open-source framework for measuring and comparing AI inference performance across NVIDIA
acceleration stacks — PyTorch, ONNX Runtime, TensorRT and TensorRT-LLM.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

> **Status: early development — Phase 6 of 12, partially validated.**
> Environment detection, the benchmark core, a PyTorch backend and an ONNX Runtime backend
> (ResNet-50) are implemented and tested; ONNX Runtime's outputs are verified against
> PyTorch. The development machine has **no NVIDIA GPU**: the PyTorch CPU
> path has run on real hardware, but the CUDA path is implemented and only structurally
> tested. **This repository contains no GPU performance measurements.** The only real
> measurements are CPU runs, published as methodology evidence and labelled as CPU results.

---

## 1. What this is

A benchmarking harness that answers questions like *"how much faster is TensorRT than PyTorch
for **this** model on **my** GPU?"* with a measurement rather than a vendor claim.

It is designed to produce results that survive technical scrutiny:

- **Raw per-iteration latency samples are stored**, not just aggregates, so anyone can
  recompute the statistics or run their own analysis.
- **Warmup, model load, engine build and steady-state inference are measured separately.**
  A TensorRT engine build is never allowed to leak into inference latency.
- **Every result records the full environment** — GPU, driver, CUDA, framework versions,
  config and git commit — so it can be reproduced or invalidated.
- **Failures and unsupported configurations are recorded as results**, with a reason, rather
  than being silently dropped or crashing a suite.

## 2. Why it exists

Inference performance claims are usually published without the configuration that produced
them. A "5x speedup" is meaningless without the model, batch size, precision, GPU, driver,
sequence length and timing methodology. This project treats that metadata as part of the
measurement, not as a footnote.

The design rules are written down in [CLAUDE.md](CLAUDE.md) and apply to every contributor,
human or AI.

## 3. Architecture

```
                   Experiment configuration (YAML)
                                |
                                v
                        Benchmark engine
             (warmup -> timed loop -> statistics)
                                |
          +---------------+-----+---------+----------------+
          v               v               v                v
       PyTorch      ONNX Runtime      TensorRT       TensorRT-LLM
          |               |               |                |
          +---------------+-----+---------+----------------+
                                |
                        GPU telemetry (NVML)
                                |
                          Results store
                                |
                    +-----------+-----------+
                    v                       v
                Dashboard                Reports
```

Full detail: [docs/architecture.md](docs/architecture.md).
Timing and validity methodology: [docs/methodology.md](docs/methodology.md).

## 4. Supported backends

| Backend | Status | Notes |
|---|---|---|
| Simulated (`fake`) | Implemented | **Produces no measurements.** Seeded random latency model for testing the framework on a GPU-less machine. |
| PyTorch | Implemented (Phase 3) | CPU and **CUDA both executed on real hardware** (NVIDIA L4, 2026-09-20). CUDA-event timing validated, explicit precision, IEEE FP32 enforced. |
| ONNX Runtime | Implemented (Phase 4) | CPU EP and **CUDA EP both executed on real hardware** (NVIDIA L4, 2026-09-20): 122/122 nodes on the CUDA EP, outputs verified against PyTorch. Active EP verified — no silent CPU fallback. |
| TensorRT | **Implemented (Phase 6), not validated** | Backend, engine-build layer and correctness gate are committed and unit-tested **against a mocked TensorRT**. No TensorRT library has been installed or executed; there are no TensorRT measurements. Engine build, deserialization and inference are separate lifecycle phases, so a build can never leak into inference latency |
| TensorRT-LLM | Planned — Phase 10 | TTFT, inter-token latency, tokens/sec |

## 5. Supported models

| Model | Role | Status |
|---|---|---|
| ResNet-50 (torchvision v1.5, weights `IMAGENET1K_V2` pinned and SHA-256 verified) | Primary vision | Implemented |
| Qwen3-1.7B / Qwen3-0.6B | Primary / secondary LLM | Selected for Phase 10 |

Reasoning, rejected candidates and sources: [docs/models.md](docs/models.md). Weights are never
committed; `gpu-bench models fetch resnet50` downloads and verifies them.

## 6. Hardware requirements

- An NVIDIA GPU with a working driver, for benchmark execution.
- Linux or Windows. TensorRT-LLM is Linux-first; see [docs/limitations.md](docs/limitations.md).
- **No GPU is required** to install the package, run the test suite, or use `gpu-bench hardware`.
  On a machine without an NVIDIA GPU the tool reports that fact explicitly and refuses to
  produce benchmark numbers.

Check your machine:

```bash
gpu-bench hardware
```

## 7. Installation

```bash
git clone <repository-url>
cd gpu-inference-benchmarking-lab
```

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Or with stock tooling:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

The core install is CPU-only and lightweight. Inference runtimes are optional extras, because
the correct PyTorch/TensorRT wheel depends on your driver's CUDA version — see
[docs/environment.md](docs/environment.md) for the per-CUDA install matrix.

## 8. Running your first check

```bash
gpu-bench hardware
```

This prints the host, NVIDIA driver, every detected GPU (VRAM, compute capability, clocks,
power, instantaneous telemetry), a per-precision support matrix derived from the GPU's compute
capability, and which inference frameworks are installed and CUDA-capable.

Machine-readable form, for recording alongside results:

```bash
gpu-bench hardware --json -o results/environment.json
```

A one-line-per-check summary:

```bash
gpu-bench doctor
```

Exit codes: `0` GPU present, `1` no usable GPU, `2` detection failed. This makes the command
usable as a CI gate.

## 9. Running a benchmark

Experiments are YAML files, validated before anything executes:

```yaml
name: simulated-smoke-test
backend: fake          # the only backend that exists today
precision: fp32
batch_size: 4
model:
  name: simulated-vision-model
  input_shape: [3, 224, 224]
benchmark:
  warmup_iterations: 10
  measurement_iterations: 200
```

```bash
gpu-bench run --config examples/simulated-smoke-test.yaml
```

This prints the phase breakdown (model load / engine build / prepare / warmup /
measurement kept separate), latency statistics with low-confidence percentiles
flagged, and throughput with its unit and formula named. The result is written to
`results/<experiment-id>/`.

Because `backend: fake` is simulated, the run prints a prominent warning, the
result is stamped `is_simulated: true`, and its directory is `sim-` prefixed.

A committed example of the output — schema and all — is in
[`examples/sample-output/`](examples/sample-output/).

Automated experiment matrices (sweeping batch sizes and precisions) arrive in Phase 7.

## 10. Viewing results

Each experiment writes four files:

| File | Contents |
|---|---|
| `result.json` | The complete result. Canonical — the others are views of it. |
| `metadata.json` | Config, full environment, backend, provenance, status. |
| `raw.json` | Every individual sample, warmup included, with units named. |
| `summary.json` | Phases, latency statistics, throughput, errors. |

Raw samples are always retained, so every statistic can be independently recomputed.

The web dashboard arrives in Phase 9.

## 11. Reproducibility

Every result embeds the full `EnvironmentReport`, the verbatim experiment configuration,
and the git commit of the benchmark code — including whether the working tree was dirty,
since a result produced from uncommitted changes is not reproducible from the commit alone.
The schema is versioned (`schema_version`) and evolves additively, so stored results stay
readable.

## 12. Benchmark methodology

The short version:

- `time.time()` is never used around GPU work; GPU execution is asynchronous.
- Timing uses `time.perf_counter_ns()`. **The engine never synchronizes on its own** — each
  backend supplies its own timer, because only it knows whether a device sync, a stream sync
  or CUDA events is correct. The mechanism used is recorded on every result.
- Model load, engine build, warmup and steady-state inference are timed separately. A
  TensorRT engine build can never leak into inference latency.
- Raw samples are always retained so percentiles can be recomputed and outliers inspected.
- Percentiles computed from too few samples to be stable are **flagged**, not presented as
  solid. Percentile method and standard-deviation convention are recorded, since tools differ.

The long version, including why each decision was made:
[docs/methodology.md](docs/methodology.md).

## 13. Limitations

Tracked honestly in [docs/limitations.md](docs/limitations.md), including which parts of this
repository have **not** been executed on real hardware.

## 14. Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The one non-negotiable rule is in
[CLAUDE.md](CLAUDE.md): never fabricate a measurement.

## 15. Roadmap

Phased plan in [docs/roadmap.md](docs/roadmap.md). Current position: **Phases 0–5B
complete and validated on an NVIDIA L4; Phase 6 (TensorRT) implemented but not yet
validated** — Phase 6A (real-environment probe) and Phase 6G (controlled benchmark)
are outstanding.

## 16. License

[Apache-2.0](LICENSE).
