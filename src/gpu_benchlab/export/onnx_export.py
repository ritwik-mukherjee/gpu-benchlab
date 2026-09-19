"""Reproducible ONNX export of registry models, with a provenance manifest.

Decisions (docs/plans/phase-4-onnxruntime.md, ADR 0007):

* ``torch.onnx.export(dynamo=True)`` -- the current default exporter.
* **Opset 20, pinned.** The torch 2.14 exporter's native default, so no version
  conversion happens; supported by ONNX Runtime 1.30. Pinned so a future torch
  default cannot silently change the artifact.
* **Dynamic batch** (``"batch"``), static C/H/W -- one artifact for every batch size.
* ``external_data=False`` -- the exporter otherwise writes weights to a separate
  ``.onnx.data`` file (observed), and a hash of the ``.onnx`` alone would not cover
  them.
* ``verbose=False`` -- the exporter's progress output crashes on a Windows cp1252
  console (observed ``UnicodeEncodeError`` on an emoji).

Artifacts live in a cache directory, never in git (~102 MB for ResNet-50). Two
exports with identical settings were observed to be byte-identical, so the
artifact SHA-256 in the manifest is a meaningful identity.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gpu_benchlab.core.backend import ModelInfo
from gpu_benchlab.core.errors import BackendError, UnavailableError
from gpu_benchlab.core.inputs import INPUT_GENERATOR, synthetic_input
from gpu_benchlab.core.provenance import capture_provenance
from gpu_benchlab.core.schema import Provenance, utc_now_iso
from gpu_benchlab.models.registry import get_model
from gpu_benchlab.models.weights import CACHE_ENV_VAR, sha256_of

__all__ = [
    "DEFAULT_OPSET",
    "MANIFEST_SCHEMA_VERSION",
    "ExportConfig",
    "OnnxManifest",
    "artifact_paths",
    "default_onnx_dir",
    "ensure_artifact",
    "export_onnx",
    "load_manifest",
    "validate_artifact",
]

DEFAULT_OPSET = 20
MANIFEST_SCHEMA_VERSION = "1.0"
INPUT_NAME = "input"
OUTPUT_NAME = "logits"
BATCH_DIM = "batch"


def default_onnx_dir() -> Path:
    """``$GPU_BENCHLAB_CACHE/onnx`` if set, else ``~/.cache/gpu-benchlab/onnx``."""
    root = os.environ.get(CACHE_ENV_VAR)
    base = Path(root) if root else Path.home() / ".cache" / "gpu-benchlab"
    return base / "onnx"


class ExportConfig(BaseModel):
    """Everything that determines the exported bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    model: str
    weights: str | None = None
    opset: int = Field(default=DEFAULT_OPSET, ge=7)
    dynamic_batch: bool = True
    trace_batch_size: int = Field(
        default=1, gt=0, description="Batch of the example input used to trace the graph."
    )
    optimize: bool = Field(default=True, description="torch.onnx.export(optimize=...).")

    def stem(self, weights_id: str) -> str:
        batch = "dynbatch" if self.dynamic_batch else f"b{self.trace_batch_size}"
        return f"{self.model}-{weights_id}-opset{self.opset}-{batch}"


class ExporterInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    exporter: str = "torch.onnx.export(dynamo=True)"
    torch_version: str
    torchvision_version: str | None
    onnx_version: str
    onnxscript_version: str | None
    opset: int
    optimize: bool
    external_data: bool = False
    dynamic_shapes: dict[str, dict[int, str]] = Field(default_factory=dict)
    trace_batch_size: int
    trace_input_generator: str = INPUT_GENERATOR
    trace_input_seed: int = 0


class TensorSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str
    shape: list[int | str] = Field(description="Symbolic dimensions are strings.")


class ArtifactInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    filename: str
    sha256: str
    bytes: int


class OnnxManifest(BaseModel):
    """Provenance of one exported artifact, stored beside it."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    schema_version: str = MANIFEST_SCHEMA_VERSION
    created_utc: str
    artifact: ArtifactInfo
    model: ModelInfo
    exporter: ExporterInfo
    input: TensorSpec
    output: TensorSpec
    export_seconds: float = Field(description="Wall time of the export. Not a benchmark.")
    provenance: Provenance


def artifact_paths(config: ExportConfig, directory: Path | None = None) -> tuple[Path, Path]:
    """(artifact.onnx, manifest.json) for a config, weights resolved via the registry."""
    weights_id = get_model(config.model).resolve_weights(config.weights)
    base = (directory or default_onnx_dir()) / config.stem(weights_id)
    return base.with_suffix(".onnx"), base.with_suffix(".manifest.json")


def load_manifest(path: Path) -> OnnxManifest:
    return OnnxManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _version(module: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(module)
    except Exception:  # noqa: BLE001 - absent or unreadable metadata is just "unknown"
        return None


def export_onnx(
    config: ExportConfig, *, directory: Path | None = None, allow_download: bool = True
) -> tuple[Path, OnnxManifest]:
    """Export a registry model to ONNX and write its manifest. Overwrites.

    Raises:
        UnavailableError: torch / onnx / onnxscript missing, or weights unobtainable.
        BackendError: export or validation failed.
    """
    try:
        import onnx
        import torch
    except ImportError as exc:
        raise UnavailableError(
            "ONNX export needs torch, torchvision, onnx and onnxscript "
            "(install the [onnx-export] extra)."
        ) from exc
    if _version("onnxscript") is None:
        raise UnavailableError("ONNX export needs onnxscript (install the [onnx-export] extra).")

    from gpu_benchlab.models.torch_loader import load_reference_model

    spec = get_model(config.model)
    loaded = load_reference_model(config.model, config.weights, allow_download=allow_download)
    onnx_path, manifest_path = artifact_paths(config, directory)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    example = torch.from_numpy(
        synthetic_input((config.trace_batch_size, *spec.input_shape), seed=0)
    )
    dynamic = {0: BATCH_DIM} if config.dynamic_batch else None

    fd, tmp_name = tempfile.mkstemp(dir=onnx_path.parent, suffix=".onnx.part")
    os.close(fd)
    tmp = Path(tmp_name)
    started = time.perf_counter()
    try:
        torch.onnx.export(
            loaded.module,
            (example,),
            str(tmp),
            dynamo=True,
            verbose=False,
            opset_version=config.opset,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_shapes=(dynamic,) if dynamic else None,
            external_data=False,
            optimize=config.optimize,
        )
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise BackendError(f"ONNX export failed: {type(exc).__name__}: {exc}") from exc
    seconds = time.perf_counter() - started

    stray = [p for p in tmp.parent.glob(tmp.name + "*") if p != tmp]
    if stray:
        for p in [tmp, *stray]:
            p.unlink(missing_ok=True)
        raise BackendError(f"Exporter wrote external data despite external_data=False: {stray}")

    tmp.replace(onnx_path)
    batch: int | str = BATCH_DIM if config.dynamic_batch else config.trace_batch_size
    manifest = OnnxManifest(
        created_utc=utc_now_iso(),
        artifact=ArtifactInfo(
            filename=onnx_path.name,
            sha256=sha256_of(onnx_path),
            bytes=onnx_path.stat().st_size,
        ),
        model=loaded.info,
        exporter=ExporterInfo(
            torch_version=torch.__version__,
            torchvision_version=_version("torchvision"),
            onnx_version=onnx.__version__,
            onnxscript_version=_version("onnxscript"),
            opset=config.opset,
            optimize=config.optimize,
            dynamic_shapes={INPUT_NAME: {0: BATCH_DIM}} if config.dynamic_batch else {},
            trace_batch_size=config.trace_batch_size,
        ),
        input=TensorSpec(name=INPUT_NAME, dtype="float32", shape=[batch, *spec.input_shape]),
        output=TensorSpec(name=OUTPUT_NAME, dtype="float32", shape=[batch, *spec.output_shape]),
        export_seconds=seconds,
        provenance=capture_provenance(),
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    validate_artifact(onnx_path, manifest)
    return onnx_path, manifest


_ONNX_DTYPES = {1: "float32", 10: "float16", 11: "float64", 16: "bfloat16"}


def _dims(value_info: Any) -> list[int | str]:
    out: list[int | str] = []
    for d in value_info.type.tensor_type.shape.dim:
        out.append(d.dim_value if d.HasField("dim_value") else d.dim_param)
    return out


def validate_artifact(onnx_path: Path, manifest: OnnxManifest) -> list[str]:
    """Check the artifact is exactly what its manifest says. Returns the checks passed.

    Raises:
        UnavailableError: the artifact or the onnx package is missing.
        BackendError: any check fails -- the artifact must not be benchmarked.
    """
    if not onnx_path.is_file():
        raise UnavailableError(f"ONNX artifact not found: {onnx_path.name}")
    try:
        import onnx
    except ImportError as exc:
        raise UnavailableError("The onnx package is required to validate artifacts.") from exc

    passed: list[str] = []
    problems: list[str] = []

    size = onnx_path.stat().st_size
    digest = sha256_of(onnx_path)
    if digest != manifest.artifact.sha256 or size != manifest.artifact.bytes:
        raise BackendError(
            f"Artifact {onnx_path.name} does not match its manifest "
            f"(sha256 {digest[:16]} vs {manifest.artifact.sha256[:16]}, "
            f"{size} vs {manifest.artifact.bytes} bytes). Re-export it."
        )
    passed.append("sha256 matches manifest")

    model = onnx.load(str(onnx_path), load_external_data=False)
    external = [
        t.name for t in model.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL
    ]
    if external:
        problems.append(f"{len(external)} initializer(s) reference external data")
    else:
        passed.append("self-contained (no external data)")

    try:
        onnx.checker.check_model(model, full_check=True)
        passed.append("onnx.checker full_check")
    except Exception as exc:  # noqa: BLE001 - any checker failure disqualifies the artifact
        problems.append(f"onnx.checker failed: {exc}")

    opsets = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}
    if opsets.get("ai.onnx") != manifest.exporter.opset:
        problems.append(f"opset {opsets.get('ai.onnx')} != manifest {manifest.exporter.opset}")
    else:
        passed.append(f"opset {manifest.exporter.opset}")

    for kind, actual, expected in (
        ("input", list(model.graph.input), manifest.input),
        ("output", list(model.graph.output), manifest.output),
    ):
        if len(actual) != 1:
            problems.append(f"expected 1 graph {kind}, found {len(actual)}")
            continue
        vi = actual[0]
        dtype = _ONNX_DTYPES.get(vi.type.tensor_type.elem_type, str(vi.type.tensor_type.elem_type))
        shape = _dims(vi)
        mismatches = [
            f"{kind} {what} {got!r} != {want!r}"
            for what, got, want in (
                ("name", vi.name, expected.name),
                ("dtype", dtype, expected.dtype),
                ("shape", shape, expected.shape),
            )
            if got != want
        ]
        problems.extend(mismatches)
        if not mismatches:
            passed.append(f"{kind} {vi.name} {dtype} {shape}")

    if problems:
        raise BackendError(
            f"ONNX artifact {onnx_path.name} failed validation: " + "; ".join(problems)
        )
    return passed


def ensure_artifact(
    config: ExportConfig,
    *,
    directory: Path | None = None,
    allow_export: bool = True,
    allow_download: bool = True,
) -> tuple[Path, OnnxManifest, bool]:
    """Return a validated artifact for ``config``, exporting it if missing.

    Returns ``(path, manifest, exported_now)``.

    Raises:
        UnavailableError: not present and export not allowed or not possible here.
        BackendError: present but invalid.
    """
    onnx_path, manifest_path = artifact_paths(config, directory)
    if onnx_path.is_file() and manifest_path.is_file():
        manifest = load_manifest(manifest_path)
        validate_artifact(onnx_path, manifest)
        return onnx_path, manifest, False
    if not allow_export:
        raise UnavailableError(
            f"ONNX artifact {onnx_path.name} is not in {onnx_path.parent} and export is "
            "disabled. Run `gpu-bench onnx export`."
        )
    path, manifest = export_onnx(config, directory=directory, allow_download=allow_download)
    return path, manifest, True
