# 2026-09-19 — Phase 4 ONNX evidence (CPU only)

**No GPU data.** The development laptop (Intel Core i7-8565U) has no NVIDIA GPU. Every
file here comes from the CPU; nothing here is GPU performance. Everything is exactly as
the tool wrote it, copied unmodified.

Software: torch 2.14.0+cpu, torchvision 0.29.0+cpu, onnx 1.23.0, onnxscript 0.7.2,
onnxruntime 1.30.0 (CPU package). Model: ResNet-50, pinned `IMAGENET1K_V2` (weights
SHA-256 `11ad3fa6…`). Artifact: `resnet50-IMAGENET1K_V2-opset20-dynbatch.onnx`,
102,348,581 bytes, SHA-256 `32dbc686…` (not committed; reproducible with
`gpu-bench onnx export resnet50`, and two exports were byte-identical). On AC power.

## `correctness-1d8f04cba5b5/` — the Phase 4 headline

PyTorch FP32 (reference) vs ONNX Runtime CPU EP (candidate) on identical inputs.
Criterion pre-registered: `|Δ| ≤ 1e-4·max|ref| + 1e-4·|ref|` elementwise, plus identical
top-1.

| batch × seed | max \|Δ\| | worst / allowed | top-1 |
|---|---|---|---|
| 1 × {0,1,2} | 1.7e-6 – 2.1e-6 | 0.0033 – 0.0037 | 100 % |
| 4 × {0,1,2} | 1.9e-6 – 2.6e-6 | 0.0032 – 0.0035 | 100 % |
| 8 × {0,1,2} | 2.6e-6 – 3.1e-6 | 0.0042 – 0.0044 | 100 % |

**9 / 9 pass.** The **FP16 negative control was rejected** as required: 687 / 1000 logits
out of tolerance, worst 14.1× the bound, while its top-1 still matched, so top-1 alone
would not have caught it.

Re-derive independently (the script imports nothing from gpu_benchlab):

```
python analysis/rederive_correctness.py results/published/2026-09-19-phase4-onnx/correctness-1d8f04cba5b5
```

## `cpu-0ecad35290e3/` — ORT CPU EP benchmark run

A real run of `examples/resnet50-onnxruntime-cpu-fp32.yaml` (batch 1, 10 warmup +
100 measured; all 58 nodes on the CPU EP).

**Local CPU measurements; the environment is known to be non-stationary and these
measurements are not suitable for backend performance ranking.** This run also drifted:
20-iteration block means were 46.4, 42.8, 57.1, 47.8 and 50.1 ms. It must not be compared
with the Phase 3 PyTorch CPU run. Inputs come from different generators, thread counts
differ, the runs were on different days and not interleaved, and ORT fused the graph
(122 → 58 nodes).

The result records `git_dirty: true`: Phase 4 code was uncommitted when it ran.

## Real onnxruntime-gpu fallback evidence (not stored as a result)

Run on this machine with `onnxruntime-gpu 1.30.0` installed in a separate venv (CUDA
libraries absent):

```
available providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
providers=['CUDAExecutionProvider']        -> session CREATED; get_providers() = ['CPUExecutionProvider']
  (stderr: onnxruntime_providers_cuda.dll depends on cublasLt64_13.dll which is missing)
+ session.disable_cpu_ep_fallback=1        -> RAISED: "graph nodes ... assigned to the default CPU EP,
                                               but fallback to CPU EP has been explicitly disabled"
gpu-bench backend, device cuda:0           -> status=unavailable, phase=build, samples=0
  "Requested CUDAExecutionProvider, but the created session is using ['CPUExecutionProvider']:
   the EP failed to load and ONNX Runtime substituted another one. Not falling back."
```
