# Benchmark model selection

- **Status:** Selected. **Not yet benchmarked.** Nothing in this document is a
  performance measurement.
- **Researched:** 2026-09-18, against the sources linked in §9.
- **Scope:** the initial suite for PyTorch (Phase 3), ONNX Runtime (Phase 4),
  TensorRT (Phase 5) and TensorRT-LLM (Phase 10).

Every figure below is one of two things, and is labelled as such:

- **Published** — copied from the linked model card, config file, repository API or
  library documentation.
- **Arithmetic** — derived from published architecture parameters (e.g. bytes =
  parameters × bytes-per-element). These are sizing estimates, **not measurements**.
  Actual memory use depends on the runtime, allocator, workspace and activations,
  and will be measured in Phase 6.

This document contains no latency or throughput expectations. There are none worth
writing down before anything has been run.

---

## 1. Decision

| Role | Model | Pinned revision | Status |
|---|---|---|---|
| **Primary vision** | ResNet-50 (torchvision, v1.5 architecture) | weights `ResNet50_Weights.IMAGENET1K_V2`, plus a random-init mode | Selected |
| **Primary LLM** | Qwen3-1.7B | `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Selected |
| **Secondary LLM** | Qwen3-0.6B | `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca` | Selected — see §4.2 for why this passes the "strong reason" bar |
| Secondary vision | — | — | None in the initial suite. ViT-B/16 is the first planned addition (§3.2). |

The two categories are deliberately served by different models. A vision
forward-pass and autoregressive generation have different shapes, different
bottlenecks and different metrics (latency per batch vs time-to-first-token and
inter-token latency). Forcing one model to serve both would make it a poor
benchmark of each.

---

## 2. Criteria

| # | Criterion | How it was assessed |
|---|---|---|
| 1 | NVIDIA / CUDA compatibility | Runs on CUDA through each target runtime without custom kernels. |
| 2 | PyTorch compatibility | Native class in torchvision or transformers; no `trust_remote_code`. |
| 3 | ONNX exportability | Standard operators; known export path with the current exporter. |
| 4 | TensorRT compatibility | Standard ops TensorRT parses; static or profile-able shapes. |
| 5 | TensorRT-LLM compatibility | Architecture listed in the current TensorRT-LLM supported-models page. |
| 6 | VRAM requirements | Weights + KV cache (arithmetic) against 8–16 GB consumer cards. |
| 7 | Consumer GPU suitability | Fits and is meaningful on a single consumer card at BF16/FP16. |
| 8 | Maturity / stability | Released at least a year ago; stable weights; not a moving target. |
| 9 | Reproducibility | Revision can be pinned; weights downloadable without an account. |
| 10 | Benchmark usefulness | Produces an informative experiment, not just a number. |
| 11 | Reference implementations | Official implementation in a maintained library. |
| 12 | License | Permissive; no gating; compatible with an Apache-2.0 benchmark tool that never redistributes weights. |

**Criterion 9 was decisive for the LLMs.** A gated repository means every user must
create an account, accept terms and supply a Hugging Face token before the first
benchmark can run. That breaks PRD §37 ("do not require users to provide secrets")
and adds friction to exactly the first-run experience PRD §52 asks us to minimise.

---

## 3. Vision candidates

### 3.1 ResNet-50 — **selected (primary)**

| Field | Value |
|---|---|
| Architecture | Residual CNN, 50 layers. torchvision implements **ResNet v1.5**: the downsampling stride sits on the 3×3 convolution rather than the first 1×1 (published, torchvision docs). |
| Parameters | 25,557,032 (published) |
| Compute | 4.09 GFLOPS per image at 224×224 (published, torchvision's figure) |
| Weights file | 97.8 MB (published) |
| Weight memory | FP32 ≈ 102 MB, FP16 ≈ 51 MB (arithmetic: params × 4 / × 2 bytes) |
| Activation memory | Not estimated. Depends on batch size and runtime; measured in Phase 6. |
| Input | `float [N, 3, 224, 224]`, NCHW. Fixed spatial size, batch is the only dynamic dimension. |
| Output | `float [N, 1000]` logits |
| Formats | torchvision eager (state dict `.pth`) → ONNX → TensorRT engine |
| PyTorch | Native `torchvision.models.resnet50`. CPU and CUDA. |
| ONNX Runtime | Conv / BatchNorm / ReLU / pooling / GEMM only — all standard ops. |
| TensorRT | Standard CNN; static spatial shape; dynamic batch via an optimisation profile. |
| TensorRT-LLM | Not applicable. |
| License | torchvision code: BSD 3-Clause (published). Pretrained weights: torchvision states pretrained models "may have their own licenses or terms and conditions derived from the dataset used for training". These weights were trained on ImageNet, whose terms are non-commercial research. |
| Source | `torchvision.models.resnet50`, weights `ResNet50_Weights.IMAGENET1K_V2` |

**Why it is included.**

- **It is the most portable graph available.** Every target runtime handles it as a
  first-class case. So when a benchmark shows a difference between backends, it
  reflects the runtime rather than an export workaround.
- **It is still an MLPerf model.** `resnet50-v1.5` remains in the MLPerf Inference
  suite (edge category, v6.1). That gives an external methodology reference for
  phase definitions and scenario design. It is **not** a number to compare against:
  MLPerf uses a different harness, dataset pipeline and rules.
- **It exercises exactly what TensorRT optimises in CNNs:** conv + BN + ReLU fusion,
  tensor-core convolutions and layout selection. This makes the PyTorch → ONNX
  Runtime → TensorRT progression informative.
- **Batch is the only dynamic dimension,** which makes the batch-size sweep
  (PRD §12) clean.
- **It is small.** Weights are ~100 MB at FP32 and the full model runs on this
  development machine's CPU, so the entire pipeline can be exercised locally.

**Weights choice.** V1 and V2 share architecture, parameter count and FLOPs
(published), so the choice does not change the compute being measured. `IMAGENET1K_V2`
is the checkpoint `DEFAULT` resolves to today. We pin it **by explicit name**, never
via `DEFAULT`, because what `DEFAULT` points to can change between torchvision
releases. Trained rather than random weights matter for the cross-backend output
agreement checks planned for Phases 4–5, where logits need real structure.

**Random-init mode.** `weights: random` builds the same architecture with seeded
random weights and downloads nothing. It is used for tests and CI. Real benchmark
results use the pinned weights. The assumption that random and trained weights cost
the same to execute is **unverified**. On CPUs, denormal floating-point values are a
plausible way for them to differ (§8).

**Known conversion issues.**

1. **TF32 trap.** In PyTorch, `torch.backends.cudnn.allow_tf32` **defaults to True**
   (verified in the PyTorch 2.14 docs). On Ampere and newer GPUs, a naive "FP32"
   ResNet-50 therefore runs its convolutions on TF32 tensor cores, and a naive
   "FP32 vs FP16" comparison is really "TF32 vs FP16". The backend must set FP32
   precision explicitly and record the setting. See the Phase 3 plan.
2. **The ONNX exporter changed.** Since PyTorch 2.9, `torch.onnx.export` defaults to
   `dynamo=True`. Dynamic batch is declared with `dynamic_shapes`; the older
   `dynamic_axes` is deprecated. Exporter mode and opset must be recorded with every
   ONNX artifact (Phase 4).
3. **Graph-level differences are part of what is measured.** ONNX Runtime and
   TensorRT fold BatchNorm into convolutions; eager PyTorch does not. That is a
   legitimate part of the backend comparison, but reports must say so.
4. **Memory format matters in PyTorch.** Tensor-core convolutions generally prefer
   `channels_last`. It is a benchmark setting, so it must be recorded rather than
   silently chosen.

**License handling.** The project never commits or redistributes weights (PRD §40).
The user's torchvision downloads them from PyTorch's servers, and the user is
responsible for their use, as torchvision's documentation states. Random-init mode
requires no weights at all.

---

### 3.2 ViT-B/16 — **deferred; first planned addition**

| Field | Value |
|---|---|
| Architecture | Vision Transformer: 16×16 patch embedding + 12 transformer encoder layers |
| Parameters | 86.6 M (published, `IMAGENET1K_V1`) |
| Compute | 17.56 GFLOPS at 224×224 (published) |
| Weights file | 330.3 MB (published) |
| Weight memory | FP32 ≈ 346 MB, FP16 ≈ 173 MB (arithmetic) |
| Input | `float [N, 3, 224, 224]` |
| License | Code BSD 3-Clause. The `SWAG` weight variants link to a separate license; only `IMAGENET1K_V1` would be used. |

**Why deferred, not included.** ViT would add a vision workload dominated by
attention and GEMM, a useful contrast to ResNet's convolutions, and a test of
attention fusion in ONNX Runtime and TensorRT. But it is a second graph to carry
through three backends before the first one has been proven end to end. It becomes
the first addition once ResNet-50 runs through PyTorch, ONNX Runtime and TensorRT.

### 3.3 MobileNetV3-Large — excluded

| Field | Value |
|---|---|
| Parameters | 5,483,032 (published) |
| Compute | 0.22 GFLOPS (published) |
| Weights file | 21.1 MB (published) |

**Why excluded.** It has about 5% of ResNet-50's compute (published figures). At
small batch sizes on a GPU, a model this light risks measuring kernel launch and
framework overhead more than the network itself. That is a valid experiment
("overhead-bound inference"), but it would confound the first suite. It is a good
candidate for that experiment later.

### 3.4 YOLO11 (Ultralytics) — excluded

**Why excluded.**

1. **License.** Ultralytics states its trained models fall under AGPL-3.0 by default,
   with an enterprise license required for proprietary use. That is incompatible with
   keeping this Apache-2.0 tool unencumbered.
2. **Ambiguous measurement boundary.** Detection latency depends heavily on NMS
   post-processing and confidence thresholds. That makes "inference latency" harder
   to define consistently across backends than a classification forward pass.

MLPerf Inference v6.x added YOLO v11 for edge object detection, so a detection model
belongs on the roadmap, via a permissively licensed model, after the core suite works.

---

## 4. LLM candidates

### 4.1 Qwen3-1.7B — **selected (primary)**

| Field | Value |
|---|---|
| Architecture | `Qwen3ForCausalLM`: decoder-only, RoPE, SwiGLU, RMSNorm, grouped-query attention |
| Layers / hidden / FFN | 28 / 2048 / 6144 (published, `config.json`) |
| Heads | 16 query heads, **8 KV heads**, head_dim 128 (published) |
| Vocabulary | 151,936 (published) |
| Context | `max_position_embeddings` 40,960; the model card documents 32,768 |
| Parameters | 1.7 B total, 1.4 B non-embedding (published, model card) |
| Weights on disk | 4,063,479,808 bytes BF16 safetensors (published, index `total_size`) |
| Weights in memory | ≈ 3.44 GB BF16 if embeddings are tied (see known issue 1) |
| KV cache per token | 2 × 28 layers × 8 KV heads × 128 × 2 bytes = **114,688 B (112 KiB)** at BF16 (arithmetic) |
| KV cache, one 4,096-token sequence | ≈ 448 MiB (arithmetic) |
| KV cache, 8 × 4,096 tokens | ≈ 3.5 GiB (arithmetic) |
| Formats | HF safetensors (BF16) |
| PyTorch | Native `transformers` class; requires transformers ≥ 4.51 (published). No `trust_remote_code`. |
| TensorRT-LLM | `Qwen3ForCausalLM` listed for the PyTorch backend on the supported-models page (last updated 2026-09-14). Listed features include CUDA Graph, chunked prefill and KV-cache reuse. |
| ONNX Runtime | Not assessed. The ONNX path for generative LLMs is `onnxruntime-genai`, not plain `torch.onnx.export` of a generation loop. Out of initial scope. |
| License | Apache-2.0, **not gated** (published) |
| Source | `https://huggingface.co/Qwen/Qwen3-1.7B`, commit `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (last modified 2025-07-26) |

**Why it is included.**

- **Its KV cache is large enough to measure.** With 8 KV heads, its per-token KV
  cache is 4× that of Qwen2.5-1.5B (2 KV heads), per the arithmetic in §5. Sequence
  length, batch size and KV-cache experiments (PRD §30) produce real memory pressure
  on a consumer card rather than disappearing into rounding. This is the
  benchmark-usefulness argument, and the main reason it beats Qwen2.5-1.5B.
- **It fits consumer GPUs.** About 3.4 GB of BF16 weights plus a multi-GiB KV budget
  fits on an 8 GB card at moderate batch sizes (arithmetic, to be confirmed by
  measurement).
- **It is permissive and ungated.** Apache-2.0 and ungated, so benchmarks need no
  account or token.
- **It is native in both reference stacks.** `transformers` and the current
  TensorRT-LLM PyTorch backend both support it, with no remote code.
- **It is mature.** Weights last changed 2025-07-26, over a year before selection.

**Known issues.**

1. **The `lm_head` is stored separately despite tied embeddings.** `config.json`
   declares `tie_word_embeddings: true`, yet the safetensors index maps
   `lm_head.weight` alone into the second shard (622,329,984 bytes). That matches
   151,936 × 2,048 × 2 bytes. On-disk size therefore overstates in-memory size by
   ~0.62 GB **if** the loader ties the weights. Different runtimes may not agree.
   Cross-backend VRAM comparisons must check whether both runtimes deduplicated it
   before attributing a memory difference to anything else.
2. **Thinking mode and chat template.** The model card describes a
   `<think>…</think>` reasoning mode and warns that greedy decoding in thinking mode
   can loop. Neither affects performance measurement, which will use fixed synthetic
   token-ID prompts and a forced output length (EOS ignored), bypassing the chat
   template. Generated text is therefore not meaningful, and the report will say so.
3. **The generation config must be overridden.** The shipped
   `generation_config.json` enables sampling. Benchmarks will set decoding
   parameters explicitly (greedy, fixed `max_new_tokens`, `min_new_tokens` equal to
   it) and record them, so output length is identical across backends and runs.
4. **transformers v5 changed the default dtype.** From v5, `from_pretrained` defaults
   to `dtype="auto"`, meaning the saved BF16, not FP32. `torch_dtype` is deprecated
   in favour of `dtype`. Any requested precision must be passed explicitly and
   verified after load.

### 4.2 Qwen3-0.6B — **selected (secondary)**

| Field | Value |
|---|---|
| Architecture | `Qwen3ForCausalLM` (same class as 4.1) |
| Layers / hidden / FFN | 28 / 1024 / 3072 (published) |
| Heads | 16 query, **8 KV**, head_dim 128 (published) |
| Weights on disk | 1.50 GB BF16 safetensors (published). Given the same `tie_word_embeddings: true` config, this probably includes a separately stored `lm_head` as in 4.1, but that is **not verified**. |
| KV cache per token | **114,688 B (112 KiB)** at BF16 (arithmetic): **identical to Qwen3-1.7B** |
| License | Apache-2.0, not gated (published) |
| Source | `https://huggingface.co/Qwen/Qwen3-0.6B`, commit `c1899de289a04d12100db370d81485cdf75e47ca` |

**Why a secondary model passes the "strong reason" bar.**

1. **It is a controlled pair with the primary.** Both models have 28 layers, 8 KV
   heads and head_dim 128, so for a given batch and sequence length **their KV-cache
   footprint is identical by construction**. They differ in hidden size, FFN width
   and weight volume. Comparing them separates weight-driven effects from KV-driven
   effects in decode. No unrelated second model offers that.
2. **It lets real (non-simulated) generation run on this machine.** At FP32 it is
   roughly 2.4–3 GB of weights (arithmetic), which fits in this machine's 15.8 GB of
   RAM. The generation code path, including TTFT and inter-token timing, can then be
   exercised end to end on CPU before any GPU is available. The CPU numbers would
   validate mechanics only, and will be labelled as CPU results.

### 4.3 Qwen2.5-1.5B-Instruct — strong alternative, not selected

| Field | Value |
|---|---|
| Architecture | `Qwen2ForCausalLM`; 28 layers, hidden 1536, 12 query / **2 KV** heads, head_dim 128 (published config; head_dim = 1536/12) |
| Parameters | 1.54 B total, 1.31 B non-embedding (published) |
| Weights on disk | 3,087,467,144 bytes, single file (published) |
| KV cache per token | 2 × 28 × 2 × 128 × 2 = **28,672 B (28 KiB)** at BF16 (arithmetic) |
| License | Apache-2.0, not gated (published) |
| Source | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| TensorRT-LLM | `Qwen2ForCausalLM` supported |

**Why not selected.** It qualifies on every hard criterion. It loses on benchmark
usefulness: with 2 KV heads, its KV cache is a quarter the size of Qwen3-1.7B's.
Sequence-length and batching experiments would show little memory pressure on the
cards this project targets. **It is kept as the fallback** if Qwen3 hits a runtime
incompatibility.

### 4.4 TinyLlama-1.1B-Chat-v1.0 — excluded (architecture fallback)

| Field | Value |
|---|---|
| Architecture | `LlamaForCausalLM`; 22 layers, hidden 2048, 32 query / 4 KV heads (head_dim 64) (published config) |
| KV cache per token | 2 × 22 × 4 × 64 × 2 = 22,528 B (22 KiB) (arithmetic) |
| Context | `max_position_embeddings` **2048** (published) |
| License | Apache-2.0, not gated (published) |

**Why excluded.** A 2,048-token context caps sequence-length experiments at 2K, and
it is 2023-era. It is kept in mind as the most universally supported architecture
(Llama) in case both Qwen generations fail somewhere.

### 4.5 Llama-3.2-1B-Instruct — excluded

1.23 B parameters, 128K context (published). **Gated.** Access requires sharing
contact information and accepting the Llama 3.2 Community License. That license also
carries usage restrictions and attribution requirements (published). Gating alone
fails criterion 9.

### 4.6 Gemma-3-1B-it — excluded

1.0 B parameters (published). **Gated** behind Google's Gemma license and prohibited-use
policy (published). Fails criterion 9.

### 4.7 Phi-4-mini-instruct — excluded from the initial suite

3,836,021,760 parameters, 7,672,066,216 bytes of BF16 safetensors (published). MIT,
not gated. **Why excluded:** BF16 weights alone are ~7.7 GB, leaving essentially no
KV-cache budget on an 8 GB card. It is a candidate for a later "larger model" tier.

### 4.8 Llama-3.1-8B — excluded (noted as the MLPerf reference)

This is MLPerf Inference's current small-LLM benchmark (v5.1 onward). About 16 GB of
BF16 weights (arithmetic: 8 B × 2 bytes) needs a 24 GB-class card or quantisation,
and the repository is gated. Recorded in case the project later aligns with MLPerf's
LLM scenario.

---

## 5. Sizing summary (arithmetic, not measured)

| Model | Weights (BF16/FP16) | KV per token (BF16) | KV at 8 × 4,096 tokens |
|---|---|---|---|
| ResNet-50 | ≈ 51 MB | — | — |
| Qwen3-0.6B | ≈ 1.5 GB on disk | 112 KiB | ≈ 3.5 GiB |
| Qwen3-1.7B | ≈ 3.44 GB (4.06 GB on disk) | 112 KiB | ≈ 3.5 GiB |
| Qwen2.5-1.5B | ≈ 3.09 GB | 28 KiB | ≈ 0.9 GiB |
| TinyLlama-1.1B | ≈ 2.2 GB | 22 KiB | n/a (2K context) |

These are lower bounds on footprint. They exclude activations, workspace, CUDA
context, allocator fragmentation and any runtime-specific overhead, all of which
Phase 6 will measure.

---

## 6. Reproducibility

- **Revisions are pinned.** ResNet-50 by explicit weights enum name, Qwen models by
  Hugging Face commit SHA. Nothing resolves "latest" or "default".
- **Checksums are recorded.** Every result records the resolved weights file and its
  SHA-256 (Phase 3 deliverable), so a result is traceable to the exact bytes.
- **Weights are never committed** (PRD §40). A `gpu-bench models fetch` command will
  download pinned weights into a cache directory.
- **Nothing needs an account or token.** Every selected model is ungated.

## 7. Security (PRD §37)

- torchvision builds the architecture from installed library code and downloads only
  a state dict. It is loaded with `weights_only=True` passed explicitly, so no pickled
  code executes.
- Hugging Face models are loaded from **safetensors only**, with
  `trust_remote_code=False` passed explicitly. Both selected LLMs use architecture
  classes built into transformers, so no repository code is ever executed.

## 8. Open questions

- **Do random and trained ResNet-50 weights cost the same to execute on CPU?**
  Denormal values could make them differ. Until measured, only pinned-weight runs are
  reportable.
- **Do transformers and TensorRT-LLM both tie Qwen3's `lm_head`?** If not, their VRAM
  numbers differ by ~0.6 GB for reasons unrelated to runtime efficiency.
- **What does TensorRT's builder do with TF32 by default?** It needs checking against
  current documentation in Phase 5, for the same reason as the PyTorch trap above.
- **Does onnxruntime-genai support Qwen3?** Unassessed; only relevant if an ONNX
  Runtime LLM path is ever added.

## 9. Sources

- torchvision ResNet-50: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
- torchvision ViT-B/16: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html
- torchvision MobileNetV3-Large: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html
- torchvision weights licensing statement: https://docs.pytorch.org/vision/stable/models.html
- torchvision license: https://github.com/pytorch/vision/blob/main/LICENSE
- PyTorch CUDA semantics (TF32 defaults, events): https://docs.pytorch.org/docs/2.14/notes/cuda.html
- PyTorch ONNX exporter: https://docs.pytorch.org/docs/2.14/onnx.html
- MLPerf Inference reference implementations: https://github.com/mlcommons/inference
- MLPerf Inference v6.0 announcement: https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/
- TensorRT-LLM supported models: https://nvidia.github.io/TensorRT-LLM/models/supported-models.html
- Qwen3-1.7B: https://huggingface.co/Qwen/Qwen3-1.7B
- Qwen3-0.6B: https://huggingface.co/Qwen/Qwen3-0.6B
- Qwen2.5-1.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- TinyLlama-1.1B-Chat-v1.0: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Llama-3.2-1B-Instruct: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- Gemma-3-1B-it: https://huggingface.co/google/gemma-3-1b-it
- Phi-4-mini-instruct: https://huggingface.co/microsoft/Phi-4-mini-instruct
- Ultralytics license: https://www.ultralytics.com/license
- transformers v5 release notes: https://github.com/huggingface/transformers/releases/tag/v5.0.0
