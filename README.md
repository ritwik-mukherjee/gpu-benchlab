# GPU Inference Benchmarking Lab

**Open-source framework for measuring and comparing AI inference performance across NVIDIA
acceleration stacks — PyTorch, ONNX Runtime, TensorRT and TensorRT-LLM.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

> **Status: early development — Phase 1 of 12.**
> Hardware/environment detection is implemented and tested. Benchmark execution is not yet
> implemented. This repository currently contains **no performance measurements**, because
> none have been produced yet. When it does, every number in it will be traceable to a
> stored raw result.

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
| PyTorch (CUDA) | Planned — Phase 3 | CUDA-event timing |
| ONNX Runtime | Planned — Phase 4 | Execution provider recorded explicitly |
| TensorRT | Planned — Phase 5 | Engine build time measured separately |
| TensorRT-LLM | Planned — Phase 10 | TTFT, inter-token latency, tokens/sec |

## 5. Supported models

Not yet selected. Model selection is a Phase 3 deliverable and will be documented — with the
reasoning for each choice — in `docs/models.md`. The repository will not commit model weights;
fetch scripts will be provided instead.

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

## 9. Creating experiments

Not yet implemented (Phase 7).

## 10. Viewing results

Not yet implemented (Phase 9).

## 11. Reproducibility

Every benchmark result will embed the `EnvironmentReport` produced by `gpu-bench hardware`,
plus the experiment configuration and the git commit of the benchmark code itself. The schema
is versioned (`schema_version`) so stored results stay readable as the format evolves.

## 12. Benchmark methodology

The short version:

- `time.time()` is never used around GPU work; GPU execution is asynchronous.
- Timing uses `time.perf_counter_ns()` with explicit synchronization, plus CUDA events where
  the backend exposes them.
- Warmup iterations are configurable and their defaults are justified, not arbitrary.
- Raw samples are always retained so percentiles can be recomputed and outliers inspected.

The long version, including why each decision was made:
[docs/methodology.md](docs/methodology.md).

## 13. Limitations

Tracked honestly in [docs/limitations.md](docs/limitations.md), including which parts of this
repository have **not** been executed on real hardware.

## 14. Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The one non-negotiable rule is in
[CLAUDE.md](CLAUDE.md): never fabricate a measurement.

## 15. Roadmap

Phased plan in [docs/roadmap.md](docs/roadmap.md). Current position: **Phase 1 complete.**

## 16. License

[Apache-2.0](LICENSE).
