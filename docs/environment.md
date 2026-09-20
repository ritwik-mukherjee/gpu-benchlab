# Setting up an execution environment

The core package is CPU-only and installs anywhere. Inference runtimes are optional
extras because **the correct wheel depends on your driver's CUDA version.** Installing
the wrong one produces a package that imports fine and then reports no CUDA device.

All version facts below were checked against PyPI on **2026-09-18**. NVIDIA's stack
moves quickly — re-check before pinning (CLAUDE.md §12).

---

## 1. Find out what your driver supports

```bash
gpu-bench hardware
```

The **CUDA (driver max)** row is the highest CUDA runtime your installed driver can
run. A framework built against a *newer* CUDA than this will not work. A framework
built against an *older* CUDA generally will, thanks to minor-version compatibility.

## 2. Base install

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

This gives you the CLI, the schema, detection and the test suite — no GPU required.

## 3. PyTorch

PyTorch is deliberately **not** pinned to a CUDA build in `pyproject.toml`, because
no single pin is correct across machines.

As of the 2.11 series, the wheels published to PyPI target **CUDA 13.0**:

```bash
uv pip install torch
```

**Measured on the development machine (2026-09-18):** the CPU build
(`--index-url https://download.pytorch.org/whl/cpu`) installed `torch 2.14.0+cpu` and
`torchvision 0.29.0+cpu`. Download ≈ 136 MiB (torch 118.3 MiB, torchvision 1.3 MiB, plus
sympy, pillow, networkx, setuptools); 511.9 MB added to site-packages; 175 s.

For a different CUDA build, use PyTorch's own index — but check that it carries the
version you need. **Checked 2026-09-19:** torch 2.14.0 (the version Phases 3–4 used) was
on `cu126`, `cu130` and `cu132` (Linux and Windows, cp312); `cu129` stopped at 2.13.0
(Linux only) and `cu128` at **2.11.0**, so an unpinned install from `cu128` silently gives
an older torch. Pin the version:

```bash
# CUDA 12.6 (driver R525+ under CUDA 12 minor-version compatibility)
uv pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126

# CUDA 13.0 (driver R580+; required anyway by onnxruntime-gpu 1.30) -- the GPU runbook's choice
uv pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130

# CPU only (harness validation; NOT for performance data)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify you got what you intended:

```bash
gpu-bench doctor
```

`torch` must show CUDA usable = yes. If it shows `no`, `gpu-bench hardware` will tell
you whether you have a CPU-only wheel or a missing device.

## 4. ONNX Runtime

`onnxruntime` and `onnxruntime-gpu` **conflict**. Install exactly one.

```bash
uv pip install -e ".[onnx]"       # onnxruntime-gpu  (latest: 1.30.0, targets CUDA 13)
uv pip install -e ".[onnx-cpu]"   # onnxruntime      (CPU only)
```

`onnxruntime-gpu` ships **both** `CUDAExecutionProvider` and
`TensorrtExecutionProvider`. `gpu-bench hardware` lists exactly which providers are
registered, and every benchmark result records which provider actually ran — a
TensorRT-EP result must never be labelled "ONNX Runtime CUDA".

**Measured on the development machine (2026-09-19):** `onnxruntime 1.30.0` (13.6 MiB
wheel), `onnx 1.23.0`, `onnxscript 0.7.2`; `onnxruntime-gpu 1.30.0` is 153 MiB *without*
CUDA libraries (`[cuda,cudnn]` extras pin CUDA 13 / cuDNN 9). **Both ORT packages require
Python ≥ 3.11.** ONNX export additionally needs the `[onnx-export]` extra (torch,
torchvision, onnx, onnxscript).

A GPU package on a machine where the CUDA EP cannot load still *lists*
`CUDAExecutionProvider`, and ORT will silently run on CPU if asked for it; the backend
detects this and reports `unavailable` (docs/decisions/0007).

## 5. TensorRT

```bash
uv pip install -e ".[tensorrt]"
```

The `tensorrt` PyPI package is a metapackage (latest: 11.3.0.99) that pulls in a
CUDA-specific variant — currently `tensorrt_cu13`. If you are on CUDA 12, install the
matching variant explicitly rather than relying on the metapackage default:

```bash
uv pip install tensorrt-cu12
```

## 6. TensorRT-LLM

**Linux only.** Do not expect it to work on native Windows. Use a Linux GPU instance
or an NVIDIA NGC container. Because TensorRT-LLM pins strict versions of TensorRT,
CUDA and PyTorch, running it in its own environment or container — rather than
forcing it into the same venv as the other backends — is the recommended approach
(PRD §31).

## 7. Telemetry

`nvidia-ml-py` is a **core** dependency, not an extra. It is a pure-Python ctypes
wrapper, so it installs on any machine and fails at `nvmlInit()` when no driver is
present — which is exactly the reportable state the detection layer is built around.

Do **not** install the `pynvml` distribution. It is deprecated; the `pynvml` *module*
is provided by `nvidia-ml-py`.

## 8. Recommended per-machine layouts

| Machine | Install |
|---|---|
| CPU-only dev box | base only; develop framework/schema/tests, run the detection and refusal paths |
| Consumer NVIDIA GPU (Windows or Linux) | base + torch (CUDA) + onnx + tensorrt |
| Linux GPU instance | as above, plus TensorRT-LLM in a separate env/container |
