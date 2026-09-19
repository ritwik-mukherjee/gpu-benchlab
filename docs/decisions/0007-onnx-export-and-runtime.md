# ADR 0007 — ONNX export and the ONNX Runtime backend

- **Status:** Accepted
- **Date:** 2026-09-19
- **Phase:** 4
- **Plan:** [phase-4-onnxruntime.md](../plans/phase-4-onnxruntime.md)

## Context

A second runtime is only meaningful if it runs *the same model on the same weights*
and produces *the same numbers*, and if the harness is certain which execution
provider actually ran. Every item below was established on the development machine
(torch 2.14.0, onnx 1.23.0, onnxscript 0.7.2, onnxruntime 1.30.0) rather than assumed:

1. `torch.onnx.export` defaults to the dynamo exporter, **opset 20**, **static shapes**,
   and writes weights to a **separate `.onnx.data` file** by default.
2. Its progress output **crashes on a Windows cp1252 console** (emoji →
   `UnicodeEncodeError`).
3. Two identical exports were **byte-identical**.
4. The `onnx` library supports opset 28; ORT 1.30 (built against ONNX 1.22) **rejects** it.
5. The CPU package registers `AzureExecutionProvider` as well as the CPU EP.
6. With the **real onnxruntime-gpu 1.30** on a machine without CUDA libraries,
   `get_available_providers()` **listed `CUDAExecutionProvider`** (and TensorRT's EP),
   and a session requested with **only** the CUDA EP was **created and ran on CPU**,
   with only a stderr warning. The CPU package behaves the same way (with a Python
   `UserWarning`).
7. `session.disable_cpu_ep_fallback = "1"` makes creation fail, but with the **same
   error** whether the CUDA EP failed to *load* or merely could not run some *nodes*.
8. ORT's CUDA EP enables **TF32 by default** (`use_tf32 = 1`).
9. ORT profiling events carry a per-node `provider`; on CPU the 58 kernel events equal
   the 58 nodes of ORT's own optimised graph.

## Decision

**Export.** `torch.onnx.export(dynamo=True, verbose=False, opset_version=20,
external_data=False, optimize=True)`, dynamic batch (`"batch"`), static C/H/W. Opset 20 is
pinned explicitly (the exporter's native default, so no version conversion; supported
by ORT 1.30). A manifest beside each artifact records model, weights id and SHA-256,
tool versions, exporter settings, I/O names/dtypes/shapes, artifact SHA-256 and git
provenance. Artifacts are cached, never committed, and re-validated (hash, no external
data, `onnx.checker` full check, opset, I/O) before use.

**Correctness is a gate, not a benchmark.** PyTorch FP32 (reference) vs ORT on
identical float32 inputs, with a pre-registered, scale-aware tolerance
(`rtol = 1e-4`, `atol = 1e-4·max|ref|`) plus identical top-1, and an **FP16 negative
control that must fail**. Raw outputs are stored so the verdict can be re-derived by a
script that shares no code with the tool.

**Provider selection.** Exactly one EP, derived from `device`; never "whatever is
available". Three independent checks, because each catches a different failure:

| Check | When | Catches | Result |
|---|---|---|---|
| `get_available_providers()` | validate | package without the EP (CPU-only wheel) | `unavailable` |
| `session.get_providers()` | build | EP listed but failed to load → ORT substituted CPU (finding 6) | `unavailable` |
| measured node placement (profiling a separate, identical session) | prepare | EP loaded but some nodes assigned to the CPU EP | `unsupported` (or recorded, if explicitly allowed) |

`disable_cpu_ep_fallback` is not used: its error cannot distinguish load failure from
partial placement (finding 7), and the two checks above separate them.

**Lifecycle.** validate (EP, precision, artifact — exported if missing, untimed) →
load (read bytes, re-verify SHA-256) → **build = `InferenceSession` creation** (graph
optimisation and EP partitioning: ORT's engine build, never inference latency) →
prepare (inputs, sanity run, placement probe) → execute (`run`, or
`run_with_iobinding` on CUDA).

**Precision.** FP32 only (the artifact is an FP32 graph); on CUDA, `use_tf32 = "0"` is
set explicitly for FP32 and `"1"` only when TF32 is requested on a GPU whose NVML
compute capability is ≥ 8.0.

**CUDA data movement.** IOBinding with device-resident input and output, so the timed
`run` does not include host↔device copies the PyTorch path does not perform.

**Python.** ORT 1.30 requires Python ≥ 3.11. Extras floors are the tested versions, so
the ORT backend is supported on 3.11+ only; the core stays 3.10+.

## Consequences

- An ORT result is traceable to exact artifact bytes, which are traceable to exact
  weight bytes.
- A CUDA request can never silently become a CPU benchmark — verified against the real
  onnxruntime-gpu package, which *would* have done exactly that.
- The placement probe costs a second session creation during prepare (untimed).
- **Unverified on NVIDIA hardware:** the CUDA EP itself; that ORT synchronizes the CUDA
  stream at the end of `Run` (the timing relies on it); IOBinding behaviour; whether
  ResNet-50 places fully on the CUDA EP; and TF32 selection via `use_tf32`.
- The FP16 negative control is a proxy for TF32 (same mantissa width, but TF32
  accumulates in FP32): evidence the tolerance can catch TF32, not proof.
- ORT fuses the graph (122 exported nodes → 58 on CPU, including a hardware-specific
  NCHWc layout). That is a real part of any runtime comparison and must be stated
  whenever one is made.
