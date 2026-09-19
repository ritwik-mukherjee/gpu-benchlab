"""PyTorch -> ONNX Runtime numerical correctness check (the Phase 4 headline).

Reference: the pinned registry model in PyTorch eager, FP32, on CPU.
Candidate: the exported artifact in ONNX Runtime, with the *same* session
options the benchmark backend uses (``session_options``), on the requested EP.

Both receive the identical float32 array for each (batch, seed) case. The weights
are checked to be the same bytes (SHA-256) before anything is compared: comparing
outputs of different weights would be meaningless.

A **negative control** is part of every report: the same PyTorch model run in
FP16 (TF32's mantissa width), compared with the same tolerance, must FAIL. If it
passes, the tolerance cannot detect reduced-precision execution and the whole
report is marked failed.

Everything needed to re-derive the verdict is stored: ``report.json`` and the raw
reference / candidate outputs in ``outputs.npz``.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gpu_benchlab.core.correctness import (
    DEFAULT_ATOL_SCALE,
    DEFAULT_RTOL,
    OutputComparison,
    compare_outputs,
)
from gpu_benchlab.core.errors import BackendError, UnavailableError
from gpu_benchlab.core.inputs import INPUT_GENERATOR, synthetic_input
from gpu_benchlab.core.provenance import capture_provenance
from gpu_benchlab.core.schema import Provenance, utc_now_iso
from gpu_benchlab.export.onnx_export import ExportConfig, OnnxManifest, ensure_artifact
from gpu_benchlab.hardware.detect import detect_environment
from gpu_benchlab.hardware.types import EnvironmentReport

__all__ = [
    "CORRECTNESS_SCHEMA_VERSION",
    "CorrectnessCase",
    "CorrectnessReport",
    "save_report",
    "verify_against_pytorch",
]

CORRECTNESS_SCHEMA_VERSION = "1.0"
DEFAULT_BATCHES = (1, 4, 8)
DEFAULT_SEEDS = (0, 1, 2)


class CorrectnessCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_size: int
    seed: int
    comparison: OutputComparison


class RuntimeInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime: str
    version: str
    device: str
    detail: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class CorrectnessReport(BaseModel):
    """A reproducible, self-describing correctness verdict."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = CORRECTNESS_SCHEMA_VERSION
    report_id: str
    created_utc: str
    passed: bool = Field(description="All cases passed AND the negative control failed as it must.")
    cases_passed: int
    cases_total: int
    negative_control: CorrectnessCase
    negative_control_rejected: bool = Field(
        description="True when the FP16 negative control FAILED the tolerance, as required."
    )
    cases: list[CorrectnessCase]
    rtol: float
    atol_scale: float
    input_generator: str = INPUT_GENERATOR
    artifact_sha256: str
    weights_sha256: str | None
    manifest: OnnxManifest
    reference: RuntimeInfo
    candidate: RuntimeInfo
    environment: EnvironmentReport
    provenance: Provenance


def verify_against_pytorch(
    export_config: ExportConfig,
    *,
    device: str = "cpu",
    batch_sizes: tuple[int, ...] = DEFAULT_BATCHES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    rtol: float = DEFAULT_RTOL,
    atol_scale: float = DEFAULT_ATOL_SCALE,
    allow_export: bool = True,
) -> tuple[CorrectnessReport, dict[str, np.ndarray]]:
    """Run the correctness check. Returns the report and the raw outputs to store.

    Raises:
        UnavailableError: torch / onnxruntime / artifact / EP unavailable.
        BackendError: reference and artifact were built from different weights.
    """
    try:
        import torch
    except ImportError as exc:
        raise UnavailableError("The correctness check needs PyTorch for the reference.") from exc

    from gpu_benchlab.backends.ort_backend import (
        CPU_EP,
        CUDA_EP,
        OrtOptions,
        _import_ort,
        create_session,
        measure_placement,
    )
    from gpu_benchlab.models.torch_loader import load_reference_model

    ort = _import_ort()
    artifact, manifest, _ = ensure_artifact(export_config, allow_export=allow_export)
    loaded = load_reference_model(export_config.model, export_config.weights)

    if loaded.info.weights_sha256 != manifest.model.weights_sha256:
        raise BackendError(
            "Refusing to compare: the reference model's weights "
            f"({loaded.info.weights_sha256}) are not the artifact's "
            f"({manifest.model.weights_sha256})."
        )

    provider = CPU_EP if device == "cpu" else CUDA_EP
    provider_options: dict[str, str] = {}
    if provider == CUDA_EP:
        provider_options = {
            "device_id": device.split(":")[1] if ":" in device else "0",
            "use_tf32": "0",
        }
    options = OrtOptions()
    model_bytes = artifact.read_bytes()
    session = create_session(ort, model_bytes, provider, provider_options, options)
    input_name, output_name = manifest.input.name, manifest.output.name

    outputs: dict[str, np.ndarray] = {}
    cases: list[CorrectnessCase] = []
    model = loaded.module
    for batch in batch_sizes:
        for seed in seeds:
            x = synthetic_input((batch, *[int(d) for d in manifest.input.shape[1:]]), seed)
            with torch.inference_mode():
                reference = model(torch.from_numpy(x)).numpy()
            candidate = session.run([output_name], {input_name: x})[0]
            key = f"b{batch}_s{seed}"
            outputs[f"{key}_reference"] = reference
            outputs[f"{key}_candidate"] = candidate
            cases.append(
                CorrectnessCase(
                    batch_size=batch,
                    seed=seed,
                    comparison=compare_outputs(
                        reference, candidate, rtol=rtol, atol_scale=atol_scale
                    ),
                )
            )

    # Negative control: same model in FP16 (TF32's mantissa width), cast back to
    # FP32 so only the numerics differ. The tolerance must reject it.
    control_batch, control_seed = batch_sizes[0], seeds[0]
    x = synthetic_input((control_batch, *[int(d) for d in manifest.input.shape[1:]]), control_seed)
    with torch.inference_mode():
        reference = model(torch.from_numpy(x)).numpy()
        half = model.to(torch.float16)
        degraded = half(torch.from_numpy(x).to(torch.float16)).to(torch.float32).numpy()
        model.to(torch.float32)
    outputs["negative_control_fp16"] = degraded
    control = CorrectnessCase(
        batch_size=control_batch,
        seed=control_seed,
        comparison=compare_outputs(reference, degraded, rtol=rtol, atol_scale=atol_scale),
    )
    control_rejected = not control.comparison.passed

    placement = measure_placement(
        ort,
        model_bytes,
        provider,
        provider_options,
        options,
        {input_name: synthetic_input((1, *[int(d) for d in manifest.input.shape[1:]]), 0)},
    )
    passed_cases = sum(c.comparison.passed for c in cases)
    report = CorrectnessReport(
        report_id=uuid.uuid4().hex[:12],
        created_utc=utc_now_iso(),
        passed=passed_cases == len(cases) and control_rejected,
        cases_passed=passed_cases,
        cases_total=len(cases),
        negative_control=control,
        negative_control_rejected=control_rejected,
        cases=cases,
        rtol=rtol,
        atol_scale=atol_scale,
        artifact_sha256=manifest.artifact.sha256,
        weights_sha256=manifest.model.weights_sha256,
        manifest=manifest,
        reference=RuntimeInfo(
            runtime="pytorch",
            version=torch.__version__,
            device="cpu",
            detail={
                "dtype": "float32",
                "mode": "eager, inference_mode",
                "threads": torch.get_num_threads(),
            },
        ),
        candidate=RuntimeInfo(
            runtime="onnxruntime",
            version=ort.__version__,
            device=device,
            detail={
                "requested_provider": provider,
                "active_providers": ",".join(session.get_providers()),
                "graph_optimization_level": options.graph_optimization_level,
                **{f"node_placement.{ep}": n for ep, n in placement.items()},
                **{f"provider_option.{k}": v for k, v in provider_options.items()},
            },
        ),
        environment=detect_environment(include_frameworks=True),
        provenance=capture_provenance(),
    )
    return report, outputs


def save_report(report: CorrectnessReport, outputs: dict[str, Any], root: Path) -> Path:
    """Write ``correctness-<id>/report.json`` and ``outputs.npz``."""
    directory = root / f"correctness-{report.report_id}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    np.savez(directory / "outputs.npz", **outputs)
    return directory
