"""Every executing backend must benchmark on the SAME input values.

Phase 5A compared PyTorch and ONNX Runtime while PyTorch drew its inputs from
``torch.randn`` and ORT from numpy: same shape, dtype and distribution, different
values for the same seed. That is an uncontrolled difference in a latency
comparison, so both now take the canonical array from ``core.inputs``.

These tests iterate ``EXECUTING_BACKENDS``, so a backend added later is covered
without touching this file, and they compare **exact bit patterns** -- not
approximate equality -- because the inputs should be the very same float32 values.

Everything runs on the CPU device: the question is which values a backend feeds its
runtime, which is device-independent. The conversion boundary (precision cast,
device placement, memory format) is asserted separately and deliberately.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")
pytest.importorskip("onnxscript")

from gpu_benchlab.cli.run_cmd import EXECUTING_BACKENDS, build_backend  # noqa: E402
from gpu_benchlab.core.config import parse_config  # noqa: E402
from gpu_benchlab.core.inputs import INPUT_GENERATOR, synthetic_input  # noqa: E402
from gpu_benchlab.export.onnx_export import ExportConfig, export_onnx  # noqa: E402
from gpu_benchlab.hardware.detect import detect_environment  # noqa: E402
from gpu_benchlab.models.registry import (  # noqa: E402
    RANDOM_WEIGHTS,
    ModelSpec,
    register_model,
    unregister_model,
)

pytestmark = [pytest.mark.torch, pytest.mark.onnx]

TINY = "tiny-input-identity"
SHAPE = (3, 8, 8)
CASES = [(1, 0), (3, 7), (8, 1)]  # (batch, seed)


class TinyNet(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(4, 5)

    def forward(self, x: Any) -> Any:
        return self.fc(torch.flatten(self.pool(torch.relu(self.conv(x))), 1))


@pytest.fixture(scope="module", autouse=True)
def tiny_model() -> Iterator[None]:
    register_model(
        ModelSpec(
            name=TINY,
            family="vision",
            architecture="TinyNet",
            build=TinyNet,
            input_shape=SHAPE,
            output_shape=(5,),
            expected_parameters=(3 * 4 * 3 * 3 + 4) + (4 * 5 + 5),
            default_weights=RANDOM_WEIGHTS,
            test_only=True,
        )
    )
    yield
    unregister_model(TINY)


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    root = tmp_path_factory.mktemp("identity-cache")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GPU_BENCHLAB_CACHE", str(root))
        export_onnx(ExportConfig(model=TINY))
        yield root


def as_array(prepared: Any) -> np.ndarray:
    """The input a backend handed its runtime, as a numpy array.

    A backend that prepares something this cannot read fails the test on purpose:
    teach the helper what the new shape means rather than skipping the check.
    """
    if isinstance(prepared, np.ndarray):
        return prepared
    if isinstance(prepared, dict):
        if len(prepared) != 1:
            raise AssertionError(f"expected exactly one input, got {sorted(prepared)}")
        return as_array(next(iter(prepared.values())))
    if hasattr(prepared, "detach"):  # torch.Tensor
        return prepared.detach().cpu().numpy()  # type: ignore[no-any-return]
    if hasattr(prepared, "numpy"):
        return prepared.numpy()  # type: ignore[no-any-return]
    raise AssertionError(
        f"Cannot read the input prepared by this backend ({type(prepared).__name__}). "
        "Extend as_array() so the identity check keeps covering it."
    )


def prepare_input(
    backend_name: str, batch: int, seed: int, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[np.ndarray, dict[str, Any]]:
    """Drive a backend's real lifecycle up to prepare() and read back its input."""
    monkeypatch.setenv("GPU_BENCHLAB_CACHE", str(cache))
    config = parse_config(
        {
            "name": f"input-identity-{backend_name}",
            "backend": backend_name,
            "device": "cpu",
            "precision": "fp32",
            "batch_size": batch,
            "model": {"name": TINY, "seed": seed},
            "benchmark": {"warmup_iterations": 1, "measurement_iterations": 1},
            "backend_options": {"allow_export": False} if backend_name == "onnxruntime" else {},
        }
    )
    backend = build_backend(config, seed=seed)
    try:
        backend.validate(config, detect_environment(include_frameworks=False))
        backend.load()
        backend.build()
        return as_array(backend.prepare(config)), dict(backend.descriptor.settings)
    finally:
        backend.close()


@pytest.mark.parametrize("backend_name", EXECUTING_BACKENDS)
@pytest.mark.parametrize(("batch", "seed"), CASES)
def test_backend_uses_the_canonical_input(
    backend_name: str, batch: int, seed: int, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    array, _ = prepare_input(backend_name, batch, seed, cache, monkeypatch)
    expected = synthetic_input((batch, *SHAPE), seed)
    assert array.dtype == expected.dtype == np.float32
    assert array.shape == expected.shape
    assert np.array_equal(array, expected), (
        f"{backend_name} did not benchmark on core.inputs.synthetic_input"
    )


@pytest.mark.parametrize(("batch", "seed"), CASES)
def test_all_backends_receive_identical_inputs(
    batch: int, seed: int, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline claim: bit-identical inputs across every executing backend."""
    arrays = {
        name: prepare_input(name, batch, seed, cache, monkeypatch)[0] for name in EXECUTING_BACKENDS
    }
    first, *rest = EXECUTING_BACKENDS
    for other in rest:
        assert np.array_equal(arrays[first], arrays[other]), (
            f"{first} and {other} benchmarked on different input values"
        )


@pytest.mark.parametrize("backend_name", EXECUTING_BACKENDS)
def test_result_records_which_generator_produced_the_input(
    backend_name: str, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old results used a different stream, so the generator must be recorded."""
    _, settings = prepare_input(backend_name, 2, 3, cache, monkeypatch)
    assert settings["input_generator"] == INPUT_GENERATOR
    assert settings["input_seed"] == 3


def test_precision_cast_is_the_only_conversion(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Documents the conversion boundary.

    A backend may cast the canonical float32 values to the requested precision. That
    cast is deterministic, so the fp16 input must equal the canonical array cast to
    fp16 exactly -- not merely be close to it.
    """
    monkeypatch.setenv("GPU_BENCHLAB_CACHE", str(cache))
    config = parse_config(
        {
            "name": "input-identity-fp16",
            "backend": "pytorch",
            "device": "cpu",
            "precision": "fp16",
            "batch_size": 2,
            "model": {"name": TINY, "seed": 5},
            "benchmark": {"warmup_iterations": 1, "measurement_iterations": 1},
        }
    )
    backend = build_backend(config, seed=5)
    try:
        backend.validate(config, detect_environment(include_frameworks=False))
        backend.load()
        prepared = as_array(backend.prepare(config))
    except Exception as exc:  # pragma: no cover - CPU fp16 conv is not universal
        if "not supported" in str(exc).lower():
            pytest.skip(f"this PyTorch CPU build cannot run fp16 convolutions: {exc}")
        raise
    finally:
        backend.close()
    expected = synthetic_input((2, *SHAPE), 5).astype(np.float16)
    assert prepared.dtype == np.float16
    assert np.array_equal(prepared, expected)
