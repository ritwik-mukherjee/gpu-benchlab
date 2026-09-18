# Phase 3 plan — PyTorch backend

- **Status:** Implemented (Phase 3 closed 2026-09-18). CPU path executed on real hardware; CUDA path structurally tested only. Deviations from this plan are recorded in ENGINEERING_LOG.md.
- **Depends on:** [models.md](../models.md) (ResNet-50 selected as the primary vision model).
- **Out of scope:** CUDA-specific runtimes other than PyTorch, ONNX Runtime, TensorRT,
  TensorRT-LLM, and LLM generation. The LLM path does **not** go through this backend
  (§6).

---

## 1. How ResNet-50 flows through `PyTorchBackend`

Mapped onto the existing Phase 2 lifecycle. The engine is unchanged except where §2
says otherwise.

| Engine step | `PyTorchBackend` does | Timed as |
|---|---|---|
| `validate(config, environment)` | Checks torch is importable (`unavailable` if not). Checks the requested device exists (`cuda:0` on a GPU-less machine → `unavailable`). Checks precision is valid for that device (§3) → `unsupported` with a reason. Checks the model is in the registry. | not timed |
| `load()` | Builds the architecture from the registry (`torchvision.models.resnet50(weights=None)`). Loads the pinned state dict with `weights_only=True`, or seeded random init. Verifies the parameter count equals the registry's expected value (25,557,032), so the wrong architecture can never be benchmarked. Casts to the requested dtype, moves to the device, calls `.eval()`, applies memory format. Records the weights file SHA-256. | `model_load_ms` |
| `build()` | Nothing, for eager mode. Explicitly a no-op, so `engine_build_ms` is honestly ~0. (`torch.compile` would live here later.) | `engine_build_ms` |
| `prepare(config)` | Seeded synthetic input of `[N, 3, 224, 224]`, in the requested dtype, on the device, in the chosen memory format. Runs **one sanity forward pass**: output shape must be `[N, 1000]` and all values finite, else `BackendError`. Synchronizes. This proves the model actually executed before any timing starts. | `prepare_inputs_ms` |
| `execution_context()` *(new, §2c)* | `torch.inference_mode()` entered once around warmup + measurement, not once per iteration. | — |
| `make_timer()` | CPU: `WallClockTimer()` with no sync hook, since PyTorch CPU ops have completed when the call returns (mechanism `wall_clock`). CUDA: `CudaEventTimer` (§4). | — |
| warmup / measure → `execute(inputs)` | `return self._model(inputs)`. Nothing else in the method. | per-iteration samples |
| `synchronize()` | CUDA: `torch.cuda.synchronize(device)`. CPU: no-op. | — |
| `close()` | Drops the model, empties the CUDA cache, and **restores every process-global torch flag it changed** (§5, risk 3). | — |

---

## 2. Required interface changes

All are additive. `RESULT_SCHEMA_VERSION` goes 1.0 → 1.1, and 1.0 results still load.

| # | Change | Why | Where |
|---|---|---|---|
| a | `Backend.validate(config, environment)`. The engine passes its `EnvironmentReport`. | Precision validity on CUDA depends on the GPU's compute capability. The backend currently cannot see it, which would force it to re-detect hardware or guess. | `core/backend.py`, `core/engine.py`, `backends/fake.py` |
| b | New `UnavailableError`, mapped by the engine to `status: unavailable`. | **Existing gap:** nothing can currently produce `unavailable`. A missing device or missing torch would be mis-recorded as `failed`. | `core/errors.py`, `core/engine.py` |
| c | `Backend.execution_context() -> ContextManager`, default `nullcontext()`. The engine wraps warmup + measurement in it. | `inference_mode` must be active for the loop, but entering it per iteration puts context-manager overhead inside the timed window. An explicit hook is cleaner than a backend entering a context in `prepare()` and leaving it in `close()`. | `core/backend.py`, `core/engine.py` |
| d | `BackendDescriptor.settings: dict[str, str \| int \| float \| bool \| None]` | CLAUDE.md §5 requires recording build flags. For PyTorch these determine the numbers: dtype, FP32 precision policy (§3), `cudnn.benchmark`, memory format, thread count, grad mode, and torch / CUDA / cuDNN versions. | `core/backend.py` |
| e | `ModelConfig.weights: str` (`"IMAGENET1K_V2"` or `"random"`) and `ModelConfig.seed`. New result section `model_info`: resolved source, weights id, file SHA-256, parameter count. | A result must be traceable to exact bytes, not just a model name. | `core/config.py`, `core/schema.py` |
| f | Optional secondary timing: `Timer.secondary_samples() -> tuple[TimingMechanism, list[float]] \| None`, collected once after the loop. Stored as `raw_samples.secondary_latency_ms`, plus `secondary_timing_mechanism` and `secondary_latency` statistics. | methodology.md §1 promises that device time (CUDA events) and host time are both recorded and reported separately. The timer accumulates internally, so the engine adds **no** per-iteration work. | `core/timing.py`, `core/engine.py`, `core/schema.py` |
| g | `ExperimentConfig.backend_options: dict`, validated by a backend-specific pydantic model with `extra="forbid"`. | Backend knobs (`cudnn_benchmark`, `fp32_precision`, `channels_last`, `num_threads`) need a validated home. Putting them in the generic config would pollute it for every backend. | `core/config.py`, `backends/pytorch.py` |
| h | Model registry: `gpu_benchlab/models/registry.py` with a `ModelSpec` (name, builder, default input shape, weights ids, expected parameter count, license note, source URL). Phase 3 contains **only ResNet-50**, plus a test-only tiny CNN. | One place that knows what "resnet50" means. Unknown models fail at config time rather than at load. | new |
| i | The engine tracks the current phase, so every `ErrorRecord.phase` is exact (`validate` / `load` / `build` / `prepare` / `warmup` / `measure` / `close`). | **Existing bug:** an OOM is always tagged `execute` (even when raised in `prepare()`) and generic errors are tagged `run`. | `core/engine.py` |

**Not changed:** the measurement loop, statistics, storage layout and CLI shape.

---

## 3. Precision semantics in the PyTorch backend

Precision means **explicit casting** of weights and inputs (`model.to(dtype)`), not
`torch.autocast`. Autocast is mixed precision: a different experiment that, if added,
gets its own recorded mode. The result records `precision_mode: "cast"`.

| Requested | What runs | CUDA validity | CPU validity |
|---|---|---|---|
| `fp32` | FP32 weights. `cudnn.conv` **and** `cuda.matmul` FP32 precision explicitly set to `"ieee"`. | Always | Always |
| `tf32` | FP32 weights, both FP32 precision settings `"tf32"` | SM ≥ 8.0 (Phase 1 matrix) | `unsupported` |
| `fp16` | `model.half()`, FP16 input | Phase 1 matrix (SM ≥ 5.3). Tensor-core availability recorded. | Probe: a one-op FP16 conv at validate time. Raises → `unsupported` with the error text. |
| `bf16` | `model.to(bfloat16)` | SM ≥ 8.0 | Probe, as above |
| `int8` / `fp8` / `int4` / `fp4` | — | `unsupported`: "eager PyTorch has no general quantized inference path for this model; see the TensorRT backend (Phase 5)" | same |

**Why `fp32` sets precision explicitly.** Verified in the PyTorch 2.14 docs:
`torch.backends.cudnn.allow_tf32` **defaults to True**. Left alone, "FP32" ResNet-50
on Ampere or newer would run its convolutions on TF32 tensor cores. Every "FP32 vs
FP16" speedup would then be understated, and "PyTorch FP32 vs TensorRT FP32" would
compare different numerics. The backend uses the `fp32_precision` API (PyTorch ≥ 2.9)
when present, falls back to the `allow_tf32` flags otherwise, and records which API
it used plus the resulting settings.

**Why the CPU uses probes rather than a table.** CLAUDE.md §3 says to detect
capability, not assume it. Phase 1's matrix covers CUDA compute capabilities. CPU
reduced-precision support depends on instruction sets that matrix does not model. A
probe reports what this build of PyTorch can actually execute. If a probe passes but
the full model fails, that becomes `failed`, which is also honest.

---

## 4. `CudaEventTimer` (designed here; implementation is a decision point, §9)

```
start():  torch.cuda.synchronize(device)          # drain prior work (same contract as WallClockTimer)
          host_t0 = perf_counter_ns()
          start_event.record(stream)
stop():   end_event.record(stream)
          end_event.synchronize()                  # wait for this iteration only
          host_ms = (perf_counter_ns() - host_t0) / 1e6
          device_ms = start_event.elapsed_time(end_event)
          secondary.append(host_ms)
          return device_ms                         # primary = cuda_event
```

One start/end event pair is reused each iteration. That is safe because `stop()`
synchronizes on the end event before returning. Primary mechanism is `cuda_event`
(device time), and the secondary is
`wall_clock_synchronized` (host time, including launch overhead). Both are stored,
so the gap between them, meaning the host-side overhead, becomes a measured quantity
rather than an assumption.

---

## 5. Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Torch install size and time** on this machine (CPU wheel plus torchvision). | Install the CPU build from `download.pytorch.org/whl/cpu` into the dev venv only. Keep it an optional extra. Report the actual installed size rather than guessing. |
| 2 | **Weights download needs network,** and the weights carry ImageNet-derived terms. | Unit tests use random init or a tiny test model, with no download. One opt-in test downloads and SHA-checks the pinned file. Nothing is committed. |
| 3 | **Process-global torch state leaks between experiments.** `cudnn.benchmark`, the TF32 / FP32-precision flags and `set_num_threads` are global. A Phase 7 matrix run would silently inherit the previous experiment's settings. | Snapshot the flags in `load()` and restore them in `close()`. A dedicated test asserts restoration, including after a failure. Record effective settings **after** applying them, never the requested ones. |
| 4 | **`inference_mode` is thread-local.** | The engine is single-threaded; the context is entered and exited on the engine's thread by construction (§2c). |
| 5 | **Random vs trained weights may not cost the same on CPU** (denormals). | Only pinned-weight runs are treated as reportable. The question is logged in models.md §8, and a direct comparison is cheap once torch is installed (step 7). |
| 6 | **The CUDA path cannot be executed here.** | Fake-tested only, marked `@pytest.mark.gpu`, and listed in limitations.md, the same treatment as the Phase 1 NVML success path. |
| 7 | **CPU results mistaken for GPU results** in later comparisons. | Device is recorded in the descriptor and settings. The Phase 8 comparison engine must refuse cross-device speedups unless explicitly requested. This is flagged now so it is not forgotten. |

---

## 6. What does *not* go through this backend

LLM generation (Phase 10). TTFT and inter-token latency need timestamps **inside**
`generate()`, per token. The `prepare → execute → timer` shape measures whole calls
and cannot express that. Per CLAUDE.md §5, generation gets its own
`GenerationBackend` contract, reusing the same statistics, storage, provenance and
environment layers. Qwen3-0.6B on CPU (models.md §4.2) is how that path gets
exercised before a GPU exists. `transformers` is **not** added as a dependency in
Phase 3.

---

## 7. CPU-only testing on this machine

**Real and exercisable here (Intel i7-8565U, no NVIDIA GPU):**

- The full ResNet-50 path on CPU. Registry, load, parameter-count check, FP32,
  sanity forward, inference mode, timer, statistics, storage.
- The resulting samples are **real measurements of this laptop's CPU**:
  `is_simulated: false`, `device: cpu`, `timing_mechanism: wall_clock`. They are
  legitimate data about CPU inference and are never GPU data. The descriptor makes
  that explicit.
- `device: cuda:0` on this machine produces a genuine `unavailable` result. This is a
  real test of change 2b, not a mock.
- Precision probes for FP16 and BF16 on this CPU, whatever they report.
- Global-state restoration, error-phase attribution, OOM exception mapping (with a
  fake `torch.OutOfMemoryError`).
- **A harness-overhead baseline.** methodology.md §9 promises an empty-loop
  measurement, and a no-op backend on a real wall clock measures the harness's own
  per-iteration cost on real hardware. This is the first real number the project can
  legitimately produce.

**Test layout.**

- Unit tests use a registry-registered tiny CNN, a few conv layers at a small input
  size, so the backend mechanics run in milliseconds.
- One integration test runs real ResNet-50 (random init, batch 1, a few iterations),
  marked `slow`.
- All torch tests are marked `torch` and skip cleanly without torch. The existing
  229 tests must keep passing without torch installed.
- CI gains a second job that installs CPU torch.

**Not exercisable here:** `CudaEventTimer` against a real device, CUDA
FP16/BF16/TF32, `cudnn.benchmark` autotuning during warmup, and GPU memory. These
are recorded in limitations.md.

---

## 8. Implementation plan

Each step ends green (tests, ruff, mypy) and is its own commit.

1. **Interface changes 2a–2i** plus FakeBackend updates and tests. Schema → 1.1, and
   1.0 results must still load (test).
2. **Model registry** with the ResNet-50 spec and a test-only tiny CNN; weights
   pinning and SHA-256 recording; `gpu-bench models` (list) and `gpu-bench models
   fetch resnet50` (download pinned weights, report the hash).
3. **Install CPU torch + torchvision** in the dev venv and record the actual versions
   and size. Add `torchvision` to the `[torch]` extra.
4. **`PyTorchBackend` CPU path:** options model, precision policy with probes,
   settings recording, sanity forward, global-state snapshot and restore. Tests
   against the tiny CNN.
5. **CUDA path + `CudaEventTimer`**, if approved (§9): device validation,
   `torch.cuda.synchronize`, OOM mapping, and the timer with a fake Event/stream
   shim. Real-GPU tests marked `gpu` and skipped here.
6. **CLI wiring** for `backend: pytorch`. Example configs
   `examples/resnet50-pytorch-cpu-fp32.yaml` and
   `examples/resnet50-pytorch-cuda-fp16.yaml`; the latter must produce
   `unavailable` here.
7. **First real measurements, CPU only:**
   - the harness-overhead baseline;
   - ResNet-50 FP32 batch 1 with pinned weights;
   - pinned vs random weights (answers risk 5);
   - a warmup-convergence script in `analysis/` using the retained warmup samples.

   All recorded in ENGINEERING_LOG.md as CPU results.
8. **CI job** with CPU torch.
9. **Docs:** limitations, methodology (TF32 trap, precision semantics), ADR 0005
   (explicit precision semantics), ADR 0006 (model selection), CHANGELOG.

---

## 9. Decisions needed before implementation

1. **Write the CUDA path in Phase 3 (step 5) even though it cannot run here?**
   Recommended: **yes.** It is small and its API is documented. Fake-testing it plus
   listing it as unverified matches how the Phase 1 NVML success path was handled.
   The alternative is deferring it until a GPU is available, which means Phase 3
   delivers a CPU-only backend.
2. **Keep Qwen3-0.6B as a secondary LLM?** Recommended: **yes**, for the controlled
   KV-geometry pairing and CPU-feasible generation (models.md §4.2). It costs nothing
   until Phase 10.
3. **Install the CPU torch + torchvision wheels** in this machine's dev venv (step 3).
   Needed for everything in §7. The actual download size will be reported, not
   estimated.
