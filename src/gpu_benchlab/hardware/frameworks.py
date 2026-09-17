"""Detection of installed inference frameworks and their CUDA readiness.

A framework being importable is not the same as it being able to use a GPU. The
classic failure is a CPU-only wheel on a machine that has a perfectly good GPU:
`import torch` succeeds, `torch.cuda.is_available()` is False, and a naive tool
reports "PyTorch available" and then silently benchmarks the CPU.

This module therefore reports `available` and `cuda_available` separately, and
records the CUDA version each framework was *built against* — which is distinct
from the CUDA version the driver supports (see `NVMLInfo.cuda_driver_version`).

Importing heavyweight frameworks is slow, so detection is lazy and each probe is
independently guarded: a broken install of one framework must not prevent the
others from being reported.
"""

from __future__ import annotations

from gpu_benchlab.hardware.types import FrameworkInfo

__all__ = ["KNOWN_FRAMEWORKS", "probe_frameworks"]

KNOWN_FRAMEWORKS = ("torch", "onnxruntime", "tensorrt", "tensorrt_llm")


def _absent(name: str, exc: Exception) -> FrameworkInfo:
    return FrameworkInfo(
        name=name,
        available=False,
        detail=f"Not installed ({type(exc).__name__}: {exc}).",
    )


def _broken(name: str, exc: Exception) -> FrameworkInfo:
    return FrameworkInfo(
        name=name,
        available=True,
        detail=f"Installed but failed during probing: {type(exc).__name__}: {exc}",
    )


def probe_torch() -> FrameworkInfo:
    """Probe PyTorch, including whether its build actually has a usable CUDA device."""
    try:
        import torch
    except ImportError as exc:
        return _absent("torch", exc)

    try:
        cuda_available = bool(torch.cuda.is_available())
        built_cuda = torch.version.cuda  # None for CPU-only builds
        detail = None
        if built_cuda is None:
            detail = "CPU-only build: this wheel has no CUDA support compiled in."
        elif not cuda_available:
            detail = (
                f"Built against CUDA {built_cuda} but no CUDA device is available "
                "(driver missing, no NVIDIA GPU, or device hidden by CUDA_VISIBLE_DEVICES)."
            )
        return FrameworkInfo(
            name="torch",
            available=True,
            version=torch.__version__,
            cuda_available=cuda_available,
            cuda_version=built_cuda,
            device_count=torch.cuda.device_count() if cuda_available else 0,
            detail=detail,
        )
    except Exception as exc:  # noqa: BLE001
        return _broken("torch", exc)


def probe_onnxruntime() -> FrameworkInfo:
    """Probe ONNX Runtime and record which execution providers are actually available.

    The provider list matters more than the package name: `onnxruntime-gpu` ships
    both CUDAExecutionProvider and TensorrtExecutionProvider, and conflating the
    two produces results labelled "ONNX Runtime" that were really TensorRT (PRD §28).
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        return _absent("onnxruntime", exc)

    try:
        providers = list(ort.get_available_providers())
        has_cuda = "CUDAExecutionProvider" in providers
        has_trt = "TensorrtExecutionProvider" in providers
        detail = None
        if not has_cuda and not has_trt:
            detail = "CPU-only install: no CUDA or TensorRT execution provider registered."
        return FrameworkInfo(
            name="onnxruntime",
            available=True,
            version=ort.__version__,
            cuda_available=has_cuda,
            providers=providers,
            detail=detail,
        )
    except Exception as exc:  # noqa: BLE001
        return _broken("onnxruntime", exc)


def probe_tensorrt() -> FrameworkInfo:
    try:
        import tensorrt as trt
    except ImportError as exc:
        return _absent("tensorrt", exc)

    try:
        return FrameworkInfo(name="tensorrt", available=True, version=trt.__version__)
    except Exception as exc:  # noqa: BLE001
        return _broken("tensorrt", exc)


def probe_tensorrt_llm() -> FrameworkInfo:
    try:
        import tensorrt_llm
    except ImportError as exc:
        return _absent("tensorrt_llm", exc)

    try:
        version = getattr(tensorrt_llm, "__version__", None)
        return FrameworkInfo(name="tensorrt_llm", available=True, version=version)
    except Exception as exc:  # noqa: BLE001
        return _broken("tensorrt_llm", exc)


def probe_frameworks() -> list[FrameworkInfo]:
    """Probe every known framework. Never raises."""
    return [
        probe_torch(),
        probe_onnxruntime(),
        probe_tensorrt(),
        probe_tensorrt_llm(),
    ]
