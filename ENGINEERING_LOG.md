# Engineering log

A running record of what was built, what was measured, what was learned and what
remains unknown. Appended to at the end of every phase.

---

## 2026-09-18 — Phase 0 + Phase 1

### Environment

Inspected before writing any code, as the PRD requires. Full report:
[`docs/environment-report.md`](docs/environment-report.md).

```
OS        Windows 11 Pro 10.0.26200
CPU       Intel Core i7-8565U, 4C/8T
RAM       15.81 GiB
GPU       Intel UHD Graphics 620 (integrated) — no NVIDIA device
Driver    none
CUDA      none
Python    none installed at start; Python 3.12.10 installed during setup
Docker    CLI present, daemon not running
WSL2      enabled, no distributions
```

### Finding that shaped everything else

**The development machine has no NVIDIA GPU.** Confirmed three independent ways:
NVML fails to load, `nvidia-smi` absent from all standard locations, and Windows PnP
enumeration returns no `VEN_10DE` device.

This is a hardware-requirements decision, so it was escalated rather than assumed.
Decision taken: build the framework here, run real benchmarks on a cloud NVIDIA GPU
later. Consequence: **no GPU performance number can be produced on this machine**,
and the framework is built to refuse rather than estimate.

### Research (PRD §48 — do not trust model memory on NVIDIA packages)

Checked against PyPI on 2026-09-18. Three findings that contradict what would
otherwise have been assumed:

| Assumption | Reality |
|---|---|
| `pynvml` is the NVML binding | **Deprecated.** `nvidia-ml-py` (NVIDIA-authored, v13.610.43) is correct, and it installs the `pynvml` *module*. |
| `tensorrt` is a single wheel | Metapackage (v11.3.0.99) delegating to `tensorrt_cu12` / `tensorrt_cu13`. |
| PyTorch PyPI wheels are CUDA 12 | CUDA 13.0 is the PyPI default from the 2.11 series. |
| ONNX Runtime GPU targets CUDA 12 | v1.30.0 targets CUDA 13, requires Python ≥3.11. |

The first one would have been a silent adoption of a deprecated package. Recorded in
CLAUDE.md §12 so the next contributor does not repeat it.

### What was built

**Phase 0** — repo structure, `CLAUDE.md`, README, `CONTRIBUTING`, Apache-2.0,
architecture/methodology/roadmap/limitations/environment docs, three ADRs, CI
workflow, dependency strategy.

**Phase 1** — hardware detection:

- `capability.py` — pure compute-capability → architecture + precision matrix.
- `nvml.py` — NVML access that cannot raise; every field individually guarded.
- `host.py`, `frameworks.py`, `detect.py`, versioned pydantic schema in `types.py`.
- `gpu-bench hardware` and `gpu-bench doctor`, with CI-usable exit codes.

### Results

Real output on this machine:

```
Detection status   driver_unavailable
NVML error         NVML Shared Library Not Found
NVIDIA GPUs        0
Exit code          1
```

Verified:

| Check | Result |
|---|---|
| `pytest` | **67 passed** |
| `ruff check` | clean |
| `ruff format --check` | clean |
| `mypy --strict` | clean, 11 source files |
| `gpu-bench hardware` | executed; output above |
| Python 3.10.21 | 67 passed (separate venv, then removed) |
| Python 3.12.10 | 67 passed |

No performance measurements were produced, because no benchmark engine exists yet
and no GPU is present.

### Observations

1. **The "no GPU" machine is a useful asset, not only an obstacle.** Testing that a
   tool correctly reports a missing driver is awkward on a machine that has one.
   Here, the `driver_unavailable` path was exercised against reality rather than a mock.

2. **Distinguishing four failure modes was worth the effort.** `LIBRARY_UNAVAILABLE`,
   `DRIVER_UNAVAILABLE`, `NO_NVIDIA_DEVICE` and `ERROR` need completely different
   user responses. Most tools collapse these into a boolean and then give unhelpful
   advice. This machine hits `DRIVER_UNAVAILABLE`; a cloud VM with a GPU not yet
   attached would hit `NO_NVIDIA_DEVICE`.

3. **`None` vs `0` is a correctness issue, not style.** A GPU that does not report
   power draw and a GPU drawing 0 W must not be the same value, or a missing reading
   silently becomes a data point in an average.

4. **Separating "supported" from "tensor-core accelerated" per precision matters.**
   FP16 on Pascal is supported but has no tensor cores. A benchmark reporting "FP16
   gave no speedup" without that context invites a wrong conclusion about the format
   rather than the hardware.

### Hypothesis to test on real hardware

The NVML success path is the largest unverified surface in the codebase. It has been
exercised only against an injected fake module, so the real call signatures, the
bytes-vs-str return behaviour of the current binding, and the availability of
`nvmlDeviceGetNumGpuCores` are all unconfirmed. Expectation is that it works; that is
an expectation, not a result.

### Known limitations

Tracked in [`docs/limitations.md`](docs/limitations.md). The headline: everything
requiring an actual NVIDIA GPU is implemented-and-unit-tested but **not executed**.

### Next

1. **Phase 2 — benchmark core.** Timing engine, warmup, measurement loop, statistics,
   result schema. Fully buildable and testable without a GPU using a fake backend.
2. **Validate Phase 1 on real hardware** at the first opportunity — highest-priority
   unknown.
3. **Model selection** research for Phase 3, documented with reasoning.
