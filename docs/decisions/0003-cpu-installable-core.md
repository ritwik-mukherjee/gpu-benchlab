# ADR 0003 — Keep the core install CPU-only and GPU-free

- **Status:** Accepted
- **Date:** 2026-09-18
- **Phase:** 0

## Context

This is a GPU benchmarking tool. The obvious design makes PyTorch, ONNX Runtime and
TensorRT hard dependencies.

Two facts argue against that. First, the primary development machine has **no NVIDIA
GPU at all** (see `docs/environment-report.md`), so a GPU-required install would make
the project undevelopable on the machine it is being written on. Second, the correct
CUDA wheel differs per machine, so any pin in `pyproject.toml` is wrong somewhere.

## Decision

Core dependencies are limited to `typer`, `rich`, `pydantic`, `pyyaml`, `numpy`,
`psutil` and `nvidia-ml-py` — all CPU-installable, none requiring a GPU or a driver.

Every inference runtime is an optional extra (`[torch]`, `[onnx]`, `[tensorrt]`).
PyTorch is not pinned to a CUDA variant; `docs/environment.md` carries the per-CUDA
install matrix instead.

## Rationale

- **The framework is developable and testable without a GPU.** Schema, statistics,
  comparison maths, config validation, capability logic and CLI can all be verified
  on any machine. 67 tests currently pass on a GPU-less laptop.
- **CI runs the full non-GPU suite on a standard runner** — no GPU runners needed,
  no cost, no queue.
- **No wrong pin.** The user installs the wheel that matches their driver, guided by
  `gpu-bench hardware` output.
- **The degradation paths get exercised for free.** A machine with no driver is the
  ideal fixture for testing that the tool reports "no driver" correctly — a condition
  that is otherwise awkward to simulate on a GPU machine.

## Consequences

- Installation is a two-step process on GPU machines: base install, then the runtime
  extras. This is documented as the first thing in `docs/environment.md`.
- `gpu-bench hardware` must clearly distinguish "framework not installed" from
  "framework installed but has no CUDA device", because the two-step install makes
  the CPU-wheel mistake more likely. It does.
- The test suite must be meaningful in both worlds: GPU-dependent tests are marked
  `@pytest.mark.gpu` and skip cleanly.
