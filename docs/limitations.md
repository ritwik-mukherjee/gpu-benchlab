# Limitations

This document exists so that no reader has to guess what has actually been verified.
It is updated whenever something is implemented but not executed on real hardware.

Last updated: 2026-09-18 (end of Phase 1).

---

## 1. What has been executed and verified

| Item | Verified how |
|---|---|
| Package installs on Python 3.12 (Windows) | `uv pip install -e ".[dev]"` |
| `gpu-bench hardware` runs end to end | Executed; output in [environment-report.md](environment-report.md) |
| `gpu-bench doctor` runs | Executed |
| Host/CPU/RAM/Python detection | Executed; values cross-checked against Windows CIM |
| NVML *absence* handling | Executed on a machine with genuinely no NVIDIA driver |
| `driver_unavailable` vs `no_nvidia_device` distinction | Unit-tested with fakes; the `driver_unavailable` branch also confirmed on real hardware |
| Compute-capability → architecture/precision matrix | 55 unit tests against documented capabilities of real GPUs |
| Schema JSON round-trip | Unit-tested |
| CLI exit codes | Integration-tested |
| `ruff check`, `ruff format`, `mypy --strict` | All clean |
| Test suite | 67 tests, all passing |

## 2. What has NOT been executed on real hardware

**The development machine has no NVIDIA GPU** (see
[environment-report.md](environment-report.md)). The following are therefore
implemented-and-unit-tested but **never run against a real driver**:

| Item | Status |
|---|---|
| NVML *success* path (`DetectionStatus.OK`) | Tested only against an injected fake NVML module. The real NVML call signatures have not been exercised. |
| GPU field extraction (UUID, VRAM, clocks, power, temperature) | Same — fake-tested only. |
| `nvmlDeviceGetNumGpuCores` availability | Not confirmed against a real driver; this call is newer than most of the NVML surface and may be unsupported on some drivers. It is guarded, so a failure degrades to `None`. |
| Multi-GPU enumeration | Fake-tested only. |
| PyTorch / ONNX Runtime / TensorRT probes returning *positive* results | Only the "not installed" branch has run. |

**This is the highest-priority thing to validate** the first time the tool runs on a
real NVIDIA machine. Until then, treat the GPU-present path as plausible but unproven.

## 3. Not yet implemented

Phases 2–12. Specifically: the benchmark engine, every backend, telemetry sampling,
the experiment runner, the comparison engine, the dashboard and report generation.
See [roadmap.md](roadmap.md).

## 4. Platform limitations

- **TensorRT-LLM is Linux-first.** It is not expected to work on native Windows.
  The intended path is a Linux cloud GPU instance or WSL2 with GPU passthrough.
- **Windows GPU telemetry is narrower than Linux.** Some NVML fields (notably power
  limits and certain clock queries) are unavailable or restricted on Windows,
  particularly on consumer cards. Every such field degrades to `null` rather than `0`.
- **WSL2 GPU passthrough** requires a Windows NVIDIA driver with WSL support; NVML
  inside WSL2 has historically had reduced functionality compared to native Linux.

## 5. Methodological caveats

- The default warmup (10) and measurement (100) iteration counts are a **starting
  hypothesis, not an empirically validated constant**. They will be determined
  properly once the engine exists. See [methodology.md](methodology.md) §4.
- A p99 computed from 100 samples is not a stable statistic. See methodology §6.
- No model has been selected yet, so nothing is known about model-specific behaviour.

## 6. Known open questions

- Whether each experiment should run in an isolated process, to prevent allocator
  and autotune state leaking between configurations in a matrix run.
- Whether `nvmlDeviceGetNumGpuCores` is the right source for SM count, or whether
  it should come from the CUDA runtime instead.
- How to handle GPUs shared with a desktop compositor, where the baseline is not idle.
