# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Phase 1: hardware detection

- NVIDIA GPU detection via NVML (`nvidia-ml-py`), reporting name, UUID, PCI bus,
  architecture, compute capability, VRAM, clocks, power, temperature and compute mode.
- Four distinct detection states — `ok`, `no_nvidia_device`, `driver_unavailable`,
  `library_unavailable` — plus `error`, because each needs different user action.
- Per-precision support matrix (FP32/TF32/FP16/BF16/INT8/FP8/INT4/FP4) derived from
  compute capability, distinguishing *supported* from *tensor-core accelerated*.
- Host detection: OS, CPU model, core counts, RAM, Python.
- Framework detection for PyTorch, ONNX Runtime, TensorRT and TensorRT-LLM,
  separating "installed" from "has a usable CUDA device", and recording ONNX Runtime
  execution providers.
- Versioned, immutable `EnvironmentReport` schema (`schema_version` 1.0) that
  round-trips through JSON.
- `gpu-bench hardware` (human and `--json` output, `-o` to write a file) and
  `gpu-bench doctor`, with meaningful exit codes: `0` GPU present, `1` no usable GPU,
  `2` detection failed.

### Added — Phase 0: project foundation

- Repository structure, Apache-2.0 license, README, CONTRIBUTING, CLAUDE.md.
- Architecture, methodology, roadmap, limitations and environment documentation.
- ADRs 0001 (JSON storage), 0002 (NVML binding), 0003 (CPU-installable core).
- CI on Ubuntu and Windows across Python 3.10 and 3.12, including a check that
  detection degrades honestly on GPU-less runners.
- Tooling: ruff (lint + format), mypy strict, pytest with GPU/backend markers.

### Notes

This release contains **no benchmark measurements**. The benchmark engine is not yet
implemented, and the development machine has no NVIDIA GPU. See
[docs/limitations.md](docs/limitations.md) for exactly what has and has not been
executed on real hardware.
