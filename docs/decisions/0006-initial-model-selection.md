# ADR 0006 — Initial benchmark models

- **Status:** Accepted
- **Date:** 2026-09-18
- **Phase:** 3
- **Full evaluation:** [docs/models.md](../models.md) (criteria, all candidates, sources)

## Context

The suite must eventually run through PyTorch, ONNX Runtime, TensorRT and, for LLMs,
TensorRT-LLM, on consumer NVIDIA GPUs. It must be small, reproducible, licensed for
use by an Apache-2.0 tool that never redistributes weights, and usable without an
account or token (PRD §37). Vision forward passes and autoregressive generation have
different shapes and metrics, so one model should not serve both.

## Decision

| Role | Model | Pin |
|---|---|---|
| Primary vision | ResNet-50 (torchvision, v1.5) | weights `IMAGENET1K_V2` by explicit name, URL pinned in the registry (`resnet50-11ad3fa6.pth`), SHA-256 verified and recorded; parameter count 25,557,032 asserted on load |
| Primary LLM | Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Secondary LLM | Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` |

Only ResNet-50 is implemented (Phase 3). The LLMs are selected for Phase 10.

## Rationale (summary)

- **ResNet-50:** handled natively by every target runtime (conv/BN/ReLU/GEMM), so
  backend differences reflect the runtime rather than export workarounds; batch is
  its only dynamic dimension; still in MLPerf Inference (edge, v6.1) as a
  methodology reference; small enough to run end to end on this machine's CPU.
- **Qwen3-1.7B:** Apache-2.0, ungated, native in transformers and in the TensorRT-LLM
  PyTorch backend; 8 KV heads give a KV cache large enough that sequence-length
  experiments create real memory pressure on consumer GPUs (arithmetic from the
  published config, not measured).
- **Qwen3-0.6B:** identical KV geometry to Qwen3-1.7B (28 layers × 8 KV heads × 128),
  so the pair separates weight-driven from KV-driven decode cost; also small enough
  for CPU execution of the generation path before a GPU exists.

## Rejected

Llama-3.2-1B and Gemma-3-1B (gated), Phi-4-mini (≈ 7.7 GB of BF16 weights), YOLO11
(AGPL-3.0; ambiguous NMS measurement boundary), MobileNetV3 (launch-overhead-bound),
TinyLlama (2,048-token context). Qwen2.5-1.5B is the fallback LLM; ViT-B/16 is the
first planned vision addition. Details and sources in models.md.

## Consequences

- Results are traceable to exact weight bytes, not just a model name.
- `weights: random` exists for tests and CI (no download). Whether random and pinned
  weights cost the same to execute is **unknown**: the Phase 3 CPU A/B experiment
  was inconclusive (confounded by a power-source change and a system suspend). Until
  answered, only pinned-weight runs are reportable.
- torchvision's pretrained weights may carry ImageNet-derived terms; the project
  downloads them to the user's cache and never redistributes them.
