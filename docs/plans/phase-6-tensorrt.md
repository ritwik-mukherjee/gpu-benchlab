# Phase 6 — TensorRT backend (plan)

**Status: plan only. No TensorRT code exists yet, and nothing here is a result.**
Written 2026-09-20 against TensorRT 11.3.0 documentation and the actual L4 VM.

Goal: a third backend that consumes the *same* canonical ONNX artifact as ONNX Runtime,
proves correctness against PyTorch before any performance claim, and plugs into the
existing lifecycle without new benchmark semantics.

Out of scope, deliberately: FP16, INT8, TF32-enabled cells, TensorRT-LLM, LLM metrics,
multi-GPU, dashboards.

---

## 0. Findings that drive the design (checked, not remembered)

| Finding | Source | Consequence |
|---|---|---|
| TensorRT **11.3.0.99** is current; packages are built against **CUDA 13.4** | release notes 11.3.0 | The Debian/tar route would want the 13.4 toolkit |
| Driver requirement for CUDA 13.x builds: **"r580 or later (both platforms)"** | installing-tensorrt/prerequisites | Our driver 580.159.04 satisfies it |
| pip wheels accept **"CUDA 12.x or 13.x"** | prerequisites | pip route does not require the 13.4 toolkit |
| pip install has **no `trtexec`**, no C++ headers | install-pip | `trtexec` is unavailable on the pip route; record it rather than pretend |
| cp312 `manylinux_2_28_x86_64` wheels exist for `tensorrt_cu13_libs` and `_bindings` | `pypi.nvidia.com` index | Python 3.12 venv is supported |
| `pip install tensorrt` resolves to `tensorrt_cu13` → `_libs` + `_bindings`, **with no CUDA runtime wheel dependencies** | `pip install --dry-run --report` on the VM | Which CUDA libraries TensorRT loads must be *measured*, not assumed |
| **Strongly typed networks are the default since 11.0**; weak-typing APIs removed | release notes 11.x | No `BuilderFlag.FP16`/`INT8`; precision comes from the ONNX graph's types |
| **`BuilderFlag.TF32` still exists and is "Enabled by default"** | BuilderConfig API docs | **An FP32 engine is TF32 unless TF32 is explicitly cleared.** This is the Phase 6 analogue of the PyTorch and ORT TF32 traps |
| `DISABLE_TIMING_CACHE`, `ERROR_ON_TIMING_CACHE_MISS`, `DISABLE_COMPILATION_CACHE`, `STRICT_NANS` exist | BuilderConfig API docs | Reproducibility knobs to record |

The earlier PTX incident on this VM (CUDA 13.4 `nvcc` output rejected by the r580 driver)
is **not** evidence that TensorRT 11.3 will fail: that was offline PTX from `nvcc`.
It is, however, a reason to treat "the package imports" as meaningless and to probe the
runtime stack explicitly (§3).

---

## 1. Installation strategy

**Chosen: pip wheels into the existing project venv**, from NVIDIA's index:

```
.venv/bin/pip install "tensorrt==11.3.0.99" --extra-index-url https://pypi.nvidia.com
```

Why not the alternatives:

- **Debian package** — pulls CUDA 13.4 toolkit components as dependencies and can move the
  system CUDA stack. The driver caps at CUDA 13.0, and this VM already carries a 13.4
  toolkit whose PTX the driver rejects. Changing the system stack is the risky option the
  phase brief tells me to avoid.
- **tar package** — no dependency resolution, manual `LD_LIBRARY_PATH` surgery, which is
  precisely the library-resolution failure mode ORT already demonstrated here.
- **Container** — would replace the whole validated environment for one backend.

Cost of the pip route: **no `trtexec`**. Acceptable — every check this phase needs is
available through the Python API, and the absence is recorded in the evidence.

## 2. Version pinning

`pyproject.toml` extra becomes:

```
tensorrt = ["tensorrt==11.3.0.99"]
```

Exact pin, not a floor: an engine is a compiled artifact whose bytes depend on the
builder version, so "whatever pip resolves" would make engine provenance meaningless.
The NVIDIA index is documented in `docs/environment.md`, not hidden in code.

## 3. Library loading and provenance

ORT taught us that a package importing proves nothing. Before any integration, a probe
records:

- `tensorrt.__version__`, `tensorrt.__file__`, the bindings/libs wheel versions;
- **every CUDA/TensorRT shared object mapped into the process** (`/proc/self/maps`),
  with full paths, so we can tell wheel libraries from `/usr/local/cuda-13.4/` ones;
- `trt.Builder(logger)` construction and `builder.platform_has_fast_fp16`-style
  capability queries (read-only, recorded, not acted on);
- driver / runtime versions as seen by CUDA, and the GPU identity;
- whether `trtexec` exists (expected: no, on the pip route).

Reuses `analysis/gpu_validation/_common.py:loaded_libraries`, the same mechanism that
caught ORT's cuDNN problem.

## 4. ONNX parser integration

Source of truth is the **existing canonical artifact** from `export/onnx_export.py`
(`ensure_artifact`), the same bytes ORT consumes, with its SHA-256 recorded. TensorRT
never gets its own export path.

```
logger  = trt.Logger(trt.Logger.WARNING)          # captured, not printed
builder = trt.Builder(logger)
network = builder.create_network(<flags from introspection>)
parser  = trt.OnnxParser(network, logger)
ok = parser.parse(model_bytes)
```

**Parser errors are first-class**: on failure, every `parser.get_error(i)` is collected
into the raised `BackendError` and recorded with phase `build` — never a bare "parse
failed". Network creation flags are chosen by *introspecting*
`trt.NetworkDefinitionCreationFlag` on the installed version rather than copying a 10.x
tutorial, because strong typing changed which flags exist and which are defaults.

## 5. Optimization profile for dynamic batch

The artifact has a dynamic batch dimension, so TensorRT requires at least one profile.

**One engine covering the whole matrix**, not one engine per batch:

| | shape |
|---|---|
| min | `(1, 3, 224, 224)` |
| opt | `(8, 3, 224, 224)` |
| max | `(8, 3, 224, 224)` |

Rationale: the Phase 5B matrix is batch 1 and 8. A single profile spanning 1–8 keeps one
compiled artifact for both cells, so batch is the only thing that varies between them —
the same reason the other backends share one session. `opt` is set to 8 because a tactic
chosen for the larger shape is the honest default when one engine serves both; **this
choice is recorded in the result and stated wherever the batch-1 number appears**, since
it can disadvantage batch 1.

Configurable via backend options (`profile_min_batch`, `profile_opt_batch`,
`profile_max_batch`), recorded in settings, and the runtime shape is validated against
the profile before execution — a shape outside it is `unsupported`, never silently
clamped. A later phase may compare per-batch engines; that is an experiment, not a
default.

## 6. Engine serialization / deserialization

- Build once, serialize to a cache file beside the ONNX artifact:
  `<stem>-trt<version>-sm<capability>-<profile>-<precision>.plan`, with a JSON manifest
  mirroring `OnnxManifest` (source ONNX SHA-256, engine SHA-256, builder settings, TRT
  and driver versions, GPU name and compute capability, timestamp, git provenance).
- **Engines are never committed** (`.gitignore` gains `*.plan`), like ONNX artifacts.
- A cached engine is only reused when the manifest matches the current TRT version, GPU
  compute capability, source artifact hash, profile and precision policy; otherwise it is
  rebuilt. A mismatch is a rebuild, never a silent reuse.
- `engine_build_ms` and `engine_deserialization_ms` are recorded separately, and the
  serialized size in bytes.

## 7. CUDA execution model

`cuda-python` (`cuda.bindings.runtime`) provides allocation, stream, event and copy
primitives. It is an official NVIDIA package and is added explicitly to the `[tensorrt]`
extra rather than being borrowed from torch's dependency tree, so the TensorRT backend
does not require torch.

Per run: one non-default stream; device buffers allocated once in `prepare`; input filled
once from the canonical host array; `context.set_input_shape`, `set_tensor_address` for
every I/O tensor; steady state is `context.execute_async_v3(stream)`.

## 8. Buffer ownership

The backend owns every device allocation and frees it in `close()`, including on the
error path. Input and output stay device-resident for the whole measured loop — no
host↔device copy inside timing, matching ORT's IOBinding and PyTorch's device tensors.
Output is copied to the host exactly once, in the untimed sanity pass.

## 9. Synchronization

`execute_async_v3` is asynchronous. The timed region ends with an explicit stream
synchronization, so no timing is read before the GPU finishes. This mirrors the PyTorch
contract and is validated experimentally (§12), not assumed — the ORT sync check is the
template.

## 10. Timing boundary

Backend-owned timer, per ADR 0004, with the **same two series as PyTorch**:

- **primary**: CUDA events recorded on the execution stream → device time
  (`cuda_event`);
- **secondary**: synchronized host wall time (`wall_clock_synchronized`).

This makes TensorRT comparable with PyTorch on either basis and with ORT on the host
basis. Excluded from the measured loop, and recorded separately: engine build, engine
deserialization, context creation, buffer allocation, the sanity pass, file I/O and all
telemetry.

## 11. Correctness

Reuses `core/correctness.py` unchanged — same pre-registered tolerance
(`rtol = 1e-4`, `atol = 1e-4·max|ref|`) and top-1/top-5 agreement, at batches {1,4,8} ×
seeds {0,1,2}, on `core.inputs.synthetic_input`.

Comparisons: PyTorch CPU FP32 (reference) vs TensorRT; PyTorch CUDA IEEE vs TensorRT;
ORT CUDA vs TensorRT. Raw outputs are stored for independent re-derivation.

**Negative control:** the existing FP16 control, plus a TensorRT-specific one — an engine
built with TF32 *enabled* must be measurably different from the IEEE FP32 engine. If it
is not, the correctness check cannot detect a precision change on this model and that is
reported rather than glossed over.

## 12. No-fallback and real-execution proof

TensorRT has no CPU execution provider, so "fallback" means something different here.
What is proven instead:

- the engine reports the device it was created on, and the CUDA device matches the
  benchmark GPU by UUID;
- device buffers are GPU-resident (pointer attributes queried via the CUDA runtime);
- the process appears in `nvidia-smi --query-compute-apps` during execution;
- the loaded TensorRT/CUDA libraries are recorded (§3);
- a run with `CUDA_VISIBLE_DEVICES=""` ends `unavailable`/`failed` with **zero samples**,
  exactly like the other two backends;
- every lifecycle failure is tagged with its exact phase.

## 13. Engine build reproducibility

Recorded, never assumed: TRT version, CUDA runtime and driver versions, GPU name and
compute capability, the full profile, precision policy, builder optimization level,
workspace pool limit, timing-cache state, `STRICT_NANS`, engine SHA-256 and size, build
duration, timestamp, git commit and dirty flag, and relevant environment variables.

**Two engines are built back to back and their SHA-256 compared.** Whatever the answer,
it is recorded as an observation. No claim of byte-identical engines is made unless that
comparison shows it — TensorRT tactic selection is timing-dependent, so identical bytes
are not expected by default.

## 14. Schema / metadata changes

Preferred: **no result-schema change.** Engine facts live in
`BackendDescriptor.settings` (strings/numbers/bools), which is exactly what it is for,
and a separate `TensorRtManifest` JSON sits beside the `.plan`. If something cannot be
expressed that way, the schema version is bumped deliberately and the compatibility test
extended — not silently.

## 15. Tests (must pass with TensorRT absent)

A `FakeTrt` module mirroring the `FakeOrt` pattern, plus REAL tests that skip only when
TensorRT genuinely is not installed:

unavailable/import failure; malformed ONNX and parser errors surfaced with all messages;
build failure; serialize/deserialize failure; profile construction (min/opt/max);
shape outside the profile → `unsupported`; precision policy (TF32 cleared) asserted on
the builder config; device selection and index parsing; execution and synchronization
call order; cleanup frees buffers including after failure; provenance fields present;
engine manifest round-trip; no-silent-fallback; and the input-identity test extended to
TensorRT automatically via `EXECUTING_BACKENDS`.

## 16. Evidence artifacts

New directory `results/published/<date>-phase6-tensorrt-l4/` containing: the install and
library probe, engine build manifest and reproducibility comparison, correctness report
with raw outputs, GPU execution proof, and — only if every gate passes — a controlled
benchmark matching Phase 5B's methodology exactly (batch 1 and 8, 5 repeats, 100 warmup,
1000 measured, alternating order, 100 ms telemetry). Phase 5A/5B evidence is not touched.

## 17. Explicitly NOT implemented in this phase

FP16, INT8, TF32-enabled benchmark cells, mixed precision, DLA, refit, weight streaming,
CUDA graphs, multi-stream or multi-profile execution, TensorRT-LLM, LLM metrics,
`trtexec`-based flows, engine version-compatible/lean runtimes, and any cross-backend
speedup claim that is not computed from stored raw samples.

---

## Gate order

Install probe → parse → build → deserialize → single inference → correctness → execution
proof → integration → tests → controlled benchmark. **A failure at any gate stops the
phase and is reported**; no later stage is attempted on an unproven earlier one.
