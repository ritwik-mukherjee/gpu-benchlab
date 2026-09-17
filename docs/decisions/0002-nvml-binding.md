# ADR 0002 — Use `nvidia-ml-py` as a core dependency for GPU telemetry

- **Status:** Accepted
- **Date:** 2026-09-18
- **Phase:** 1

## Context

The project needs GPU utilisation, VRAM, power, temperature and clocks. NVIDIA
exposes these through NVML. Two Python packages have historically provided bindings,
and the distinction is not obvious from either name.

Verified on PyPI, 2026-09-18:

| Distribution | Reality |
|---|---|
| `nvidia-ml-py` | Authored by NVIDIA Corporation (`nvml-bindings@nvidia.com`). Latest 13.610.43, June 2026. **Installs the `pynvml` module.** |
| `pynvml` | **Deprecated.** Its own description states the `pynvml` module is not developed or maintained in that project. Since 12.0.0 it removed its implementation and simply depends on `nvidia-ml-py>=12.0.0`. |

This is precisely the trap PRD §48 warns about: the obvious-looking package name is
the wrong one, and model memory or an old blog post would pick it.

Alternatives considered: shelling out to `nvidia-smi` and parsing its output, or
using the CUDA runtime API via PyTorch.

## Decision

Depend on `nvidia-ml-py`, and import it as `import pynvml`. It is a **core**
dependency, not an optional extra.

## Rationale

- **Official and maintained** by NVIDIA, tracking driver branches.
- **Pure Python ctypes wrapper** — it installs successfully on a machine with no
  GPU and no driver, and fails only at `nvmlInit()`. This is a feature: it makes
  "no driver present" a clean, testable, reportable state rather than an install
  failure, which is why it can be a core dependency rather than an extra.
- **Structured return values** rather than screen-scraping `nvidia-smi`, which has
  no output stability guarantee and costs a process spawn per query — unacceptable
  for telemetry sampling during a run.
- **Independent of any inference framework**, so telemetry works identically whether
  the benchmark is PyTorch, ONNX Runtime or TensorRT.

## Consequences

- The package/module name mismatch is confusing and must be documented wherever it
  appears. It is recorded in CLAUDE.md §12 and `docs/environment.md`.
- Users who have `pynvml` installed from an old tutorial may get a conflicting
  install. `docs/environment.md` warns about this explicitly.
- NVML queries can fail per-field on consumer and virtualised GPUs. Every call is
  individually guarded and degrades to `None` — never `0`.
- **Unverified:** the NVML success path has only been exercised against an injected
  fake. See `docs/limitations.md`.
