# Phase 4 plan — ONNX export and ONNX Runtime backend

- **Status:** Implemented as described below, except where ENGINEERING_LOG records a deviation.
- **Headline deliverable:** cross-runtime **numerical correctness** (PyTorch vs ONNX Runtime).
  CPU latency on the development laptop is *not* a deliverable: Phase 3 showed the CPU is
  non-stationary, and no backend ranking may be drawn from it.

## 1. Research findings (verified 2026-09-19)

| Topic | Finding | Source |
|---|---|---|
| Versions | torch 2.14.0, torchvision 0.29.0, onnx 1.23.0 (opset ≤ 28, IR 14), onnxscript 0.7.2, onnxruntime 1.30.0 | PyPI, installed |
| ORT Python | onnxruntime **and** onnxruntime-gpu 1.30 require **Python ≥ 3.11**; on 3.10 the resolver falls back to 1.23.2 (untested) | PyPI metadata, `uv pip compile` |
| ORT ↔ ONNX | ORT 1.30 is built against ONNX 1.22; an opset-28 model is **rejected** by ORT 1.30 (observed) | ORT release notes, probe |
| Exporter default | `torch.onnx.export` uses `dynamo=True`; **default opset 20**; **static shapes** unless `dynamic_shapes` is given; `optimize=True` | PyTorch 2.14 docs, probe |
| External data | The exporter wrote weights to a separate `.onnx.data` file by default; `external_data=False` gives one self-contained file | probe |
| Reproducibility | Two exports with identical settings were **byte-identical** (same SHA-256) | probe |
| Windows | Exporter verbose output crashes on a cp1252 console (emoji → `UnicodeEncodeError`); `verbose=False` avoids it | probe |
| Providers | The CPU package registers `AzureExecutionProvider` + `CPUExecutionProvider`; selection must be explicit | probe |
| Node placement | ORT profiling events carry a per-node `provider`, so actual placement is measurable | probe |
| CPU fallback | Unsupported nodes fall back to the CPU EP **by default**; `session.disable_cpu_ep_fallback = "1"` makes session creation fail instead. Behaviour when an EP fails to *load* is **not documented** → tested empirically | ORT source (`onnxruntime_session_options_config_keys.h`) |
| CUDA EP TF32 | `use_tf32` **defaults to 1** — same trap as PyTorch | ORT CUDA EP docs |
| CUDA EP versions | ORT 1.27+ targets CUDA 13 / cuDNN 9 (`onnxruntime-gpu[cuda,cudnn]` extras pin them) | ORT CUDA EP docs, PyPI |

## 2. Decisions

| Decision | Choice | Why |
|---|---|---|
| Exporter | `torch.onnx.export(dynamo=True, verbose=False)` | Current default and recommended path; `verbose=False` for Windows |
| Opset | **20, pinned explicitly** | The exporter's native default (no version conversion), supported by ORT 1.30, well below its limit. Pinned so a future torch default change cannot silently alter artifacts. |
| Batch dimension | **dynamic** (`"batch"`), C/H/W static | One artifact serves every batch size; static spatial shape keeps the graph simple |
| Weights storage | `external_data=False` (single file) | The artifact hash then covers the weights; ResNet-50 is far below the 2 GB protobuf limit |
| Optimizer | `optimize=True` (exporter default), recorded | Default behaviour; recorded in the manifest |
| Artifact location | cache dir, path derived from config; **never committed** (~102 MB) | Reproducible from config; byte-identical re-export verified |
| ORT precision | **fp32 only**; tf32 = CUDA EP `use_tf32=1` on SM ≥ 8.0; fp32 forces `use_tf32=0` | A reduced-precision ONNX graph is a different artifact; out of scope |
| Provider selection | exactly one EP, from `device` (`cpu` → CPU EP, `cuda:N` → CUDA EP `device_id=N`); never "whatever is available" | Avoids Azure EP and silent substitution |
| No fallback | (1) pre-check `get_available_providers()`; (2) post-check `session.get_providers()`; (3) CUDA default `session.disable_cpu_ep_fallback=1`; (4) measured node placement recorded | Each layer catches a different fallback mode |
| Lifecycle mapping | validate = provider + artifact checks (+ export if missing, untimed) · load = read + hash artifact · **build = InferenceSession creation** (graph optimisation, EP partitioning) · prepare = inputs + sanity run + placement probe | ORT's session creation is its engine build; it must not leak into latency |
| CUDA data movement | IOBinding with device-resident input/output | Otherwise `run()` times host↔device copies that the PyTorch path does not |
| Timing | `WallClockTimer` around `run()` | ORT returns only after execution completes (CPU: synchronous). For CUDA EP this relies on ORT synchronizing at the end of `Run` — **unverified on hardware** |
| Schema | no result-schema bump; new facts go in `backend.settings`; the export manifest and correctness report get their own versioned schemas | Nothing in the result model needs to change |

## 3. Correctness method (pre-registered before running pinned weights)

Reference = PyTorch eager FP32 (IEEE, as enforced by ADR 0005) on the pinned weights.
Candidate = ONNX Runtime on the exported artifact. **Identical float32 input arrays** are
fed to both (numpy, seeded); preprocessing is outside the model and therefore identical by
construction.

Pass criteria, fixed before the pinned-weight run:

- shape and dtype identical; all values finite;
- elementwise `|cand − ref| ≤ atol + rtol·|ref|` with **rtol = 1e-4** and
  **atol = 1e-4 · max|ref|** (scale-aware);
- **top-1 identical** for every sample.

Why 1e-4: FP32 unit roundoff is 2^-24 ≈ 6e-8; reordered reductions, fused Conv+BN and
different convolution algorithms through ~50 layers plausibly produce 1e-6–1e-5 relative
differences, so 1e-4 leaves margin. It is also ≈ 5× tighter than the unit roundoff of TF32
and FP16 (both 10-bit mantissas: 2^-11 ≈ 4.9e-4), so an accidental reduced-precision
execution *should* fail the check — the reason the tolerance must not be loosened.

That last claim is **tested, not assumed**: a negative control runs the same PyTorch model
in FP16 (same mantissa width as TF32) and requires the comparator to report FAIL. A
tolerance that cannot reject a known-wrong precision is useless. Batches 1, 4 and 8;
three seeds. Reports record max abs,
mean abs, max relative (only where `|ref| > 1e-3·max|ref|`), top-1/top-5 agreement, and the
raw outputs (`.npz`) so the result can be re-derived independently.

## 4. Known gap carried forward

The PyTorch backend generates benchmark inputs with `torch.randn`; the ORT backend uses
numpy. Values differ (shape and dtype match). Latency is essentially value-independent,
and no cross-backend latency comparison is made in Phase 4, but inputs must be unified
**before** the first such comparison. Not changed here to keep Phase 3 untouched.
