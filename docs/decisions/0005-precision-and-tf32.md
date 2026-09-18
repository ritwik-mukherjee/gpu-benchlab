# ADR 0005 — Explicit precision and TF32 rules for the PyTorch backend

- **Status:** Accepted
- **Date:** 2026-09-18
- **Phase:** 3

## Context

A precision label on a result is a claim about the numerics that produced it. Two
facts about PyTorch make that claim easy to get wrong:

1. **"FP32" is not IEEE FP32 by default.** The PyTorch 2.14 documentation states
   `torch.backends.cudnn.allow_tf32` "defaults to True", and on the installed
   torch 2.14 build `torch.backends.cudnn.conv.fp32_precision` reads `"tf32"` out of
   the box. On Ampere and newer GPUs, an unconfigured "FP32" ResNet-50 runs its
   convolutions on TF32 tensor cores. A naive "FP32 vs FP16" comparison is then
   really "TF32 vs FP16", and "PyTorch FP32 vs TensorRT FP32" compares different
   numerics.
2. **The two TF32 APIs do not mix.** PyTorch ≥ 2.9 added per-operator
   `fp32_precision` settings (`"ieee"` / `"tf32"` / `"none"`) and marked
   `allow_tf32` for deprecation. Observed on torch 2.14: after setting the new API,
   *reading* legacy `torch.backends.cudnn.allow_tf32` raises `RuntimeError`
   ("a mix of the legacy and new APIs"). It raised even with conv and RNN both set
   to `"ieee"`. Restoring the original new-API values makes the legacy read work
   again.

A third concern: these flags, `cudnn.benchmark` and the CPU thread count are
**process-global**. Experiments run in one process (Phase 7 matrices) would
inherit each other's settings.

## Decision

1. **Precision means explicit casting.** Weights and inputs are cast with
   `.to(dtype)`. `torch.autocast` (mixed precision) is not used; if added later it
   is a separate, separately recorded mode. Results record `precision_mode: "cast"`.
2. **FP32 means IEEE.** For every run, `cudnn.conv`, `cudnn.rnn` and `cuda.matmul`
   `fp32_precision` are set explicitly: `"ieee"` for `fp32`/`fp16`/`bf16`, `"tf32"`
   only when `tf32` is requested. `tf32` requires SM ≥ 8.0 and is `unsupported` on
   CPU.
3. **Use the new API only, when present.** Never read the legacy flags once it
   exists. Fall back to `allow_tf32` only on torch < 2.9, and record which API was
   used (`fp32_precision_api`).
4. **Record effective values, after applying them** — all five precision nodes,
   `cudnn.benchmark`, `cudnn.deterministic`, cuDNN version, thread counts, dtype,
   memory format, input shape. On CPU, `cuda_flags_applicable: false` says the
   CUDA flags were inert rather than implying they applied.
5. **Verify the cast took effect.** The sanity forward pass fails the run if the
   output dtype differs from the requested one.
6. **Restore global state.** Snapshot in `load()`, restore in reverse order in
   `close()`, including after failures. The global RNG is not touched (random init
   uses a forked, seeded RNG).
7. **Reduced integer / FP8 precisions** (`int8`, `fp8`, `int4`, `fp4`) are
   `unsupported` in eager PyTorch, with a pointer to the TensorRT backend.
8. **CPU reduced precision is probed, not assumed.** `fp16`/`bf16` on CPU run a
   one-op convolution at validate time; failure is `unsupported` with PyTorch's own
   message.

## Consequences

- A PyTorch FP32 result from this tool is comparable with an IEEE FP32 result from
  another runtime, instead of silently being TF32.
- Users who want PyTorch's default behaviour must ask for `tf32` explicitly, and the
  result says so.
- The dtype check proves the output dtype, not which kernels ran. **Whether `"ieee"`
  actually prevents TF32 kernel selection on a real GPU is unverified** — this
  machine has no NVIDIA GPU. It is a first check on hardware (limitations.md §2).
- The legacy branch (torch < 2.9) is untested: the installed torch has the new API.
- TensorRT has its own TF32 behaviour (a builder flag). It must be checked against
  current TensorRT documentation in Phase 5 rather than assumed to match.
