"""Build a TensorRT engine from the canonical ONNX artifact.

An engine is a **compiled artifact**, not a model: its bytes depend on the TensorRT
version, the GPU architecture, the optimization profile, the precision policy and the
builder settings. All of those are recorded in a manifest beside the engine, and a
cached engine is reused only when every one of them still matches.

Precision, on TensorRT 11.x (verified against 11.3 on an L4, 2026-09-20):

* Networks are **strongly typed** by default, so `BuilderFlag.FP16` / `INT8` no longer
  exist -- the ONNX graph's own types decide arithmetic precision.
* `BuilderFlag.TF32` **does** still exist and is **enabled by default**, so an "FP32"
  engine silently runs TF32 unless the flag is cleared. `ieee_fp32` clears it; `tf32`
  leaves it set and exists only as a negative control.

Engines are cached outside the repository and never committed.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_benchlab.core.backend import ModelInfo
from gpu_benchlab.core.errors import BackendError, UnavailableError
from gpu_benchlab.core.provenance import capture_provenance
from gpu_benchlab.core.schema import Provenance, utc_now_iso
from gpu_benchlab.export.onnx_export import (
    ExportConfig,
    OnnxManifest,
    TensorSpec,
    default_onnx_dir,
    ensure_artifact,
)
from gpu_benchlab.models.weights import CACHE_ENV_VAR, sha256_of  # noqa: F401 - cache root

__all__ = [
    "TENSORRT_MANIFEST_VERSION",
    "PrecisionPolicy",
    "TensorRtManifest",
    "TrtBuildConfig",
    "build_engine",
    "default_engine_dir",
    "engine_paths",
    "ensure_engine",
    "import_tensorrt",
]

TENSORRT_MANIFEST_VERSION = "1.0"

PrecisionPolicy = Literal["ieee_fp32", "tf32"]
"""``ieee_fp32`` clears BuilderFlag.TF32; ``tf32`` leaves TensorRT's default in place."""


def import_tensorrt() -> tuple[Any, Any]:
    """Import ``tensorrt`` and the CUDA runtime bindings, or explain what is missing."""
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise UnavailableError(
            "tensorrt is not installed, so the TensorRT backend cannot run. Install the "
            "[tensorrt] extra (see docs/environment.md); the wheels come from NVIDIA's "
            "index, not PyPI."
        ) from exc
    try:
        from cuda.bindings import runtime as cudart
    except ImportError as exc:
        raise UnavailableError(
            "cuda-python is not installed; the TensorRT backend needs it for device "
            "buffers, streams and events. Install the [tensorrt] extra."
        ) from exc
    return trt, cudart


def cuda_call(cudart: Any, result: Any) -> Any:
    """Unpack a cuda-python ``(error, *values)`` tuple, raising with the CUDA message."""
    error, *rest = result if isinstance(result, tuple) else (result,)
    if int(error) != 0:
        _, message = cudart.cudaGetErrorString(error)
        raise BackendError(f"CUDA error {int(error)}: {message.decode()}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else tuple(rest)


class TrtBuildConfig(BaseModel):
    """Everything that changes the engine bytes, and therefore its identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    weights: str | None = None
    opset: int = 20
    min_batch: int = Field(default=1, gt=0)
    opt_batch: int = Field(default=8, gt=0)
    max_batch: int = Field(default=8, gt=0)
    precision: PrecisionPolicy = "ieee_fp32"
    optimization_level: int | None = Field(
        default=None, description="None keeps the builder default, which is recorded."
    )
    workspace_bytes: int | None = Field(
        default=None, description="None keeps the builder default, which is recorded."
    )
    disable_timing_cache: bool = False

    def profile(self, chw: tuple[int, ...]) -> dict[str, list[int]]:
        return {
            "min": [self.min_batch, *chw],
            "opt": [self.opt_batch, *chw],
            "max": [self.max_batch, *chw],
        }

    def stem(self, artifact_stem: str, trt_version: str, capability: str) -> str:
        batches = f"b{self.min_batch}_{self.opt_batch}_{self.max_batch}"
        return (
            f"{artifact_stem}-trt{trt_version}-sm{capability.replace('.', '')}-"
            f"{batches}-{self.precision}"
        )


class TrtProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    min: list[int]
    opt: list[int]
    max: list[int]


class TensorRtManifest(BaseModel):
    """Provenance for one compiled engine. Written beside the ``.plan``."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = TENSORRT_MANIFEST_VERSION
    created_utc: str
    engine_file: str
    engine_sha256: str
    engine_size_bytes: int
    build_seconds: float

    source_onnx_sha256: str
    model: ModelInfo
    input: TensorSpec
    output: TensorSpec

    tensorrt_version: str
    cuda_runtime_version: int
    cuda_driver_version: int
    gpu_name: str
    compute_capability: str

    precision_policy: PrecisionPolicy
    profile: TrtProfile
    builder_settings: dict[str, str | int | bool | None]
    provenance: Provenance

    def matches(self, other: TensorRtManifest) -> bool:
        """Whether a cached engine was built for exactly this situation."""
        return (
            self.source_onnx_sha256 == other.source_onnx_sha256
            and self.tensorrt_version == other.tensorrt_version
            and self.compute_capability == other.compute_capability
            and self.precision_policy == other.precision_policy
            and self.profile == other.profile
        )


def default_engine_dir() -> Path:
    """``$GPU_BENCHLAB_CACHE/engines`` if set, else ``~/.cache/gpu-benchlab/engines``."""
    return default_onnx_dir().parent / "engines"


def engine_paths(
    config: TrtBuildConfig,
    artifact_stem: str,
    trt_version: str,
    capability: str,
    directory: Path | None = None,
) -> tuple[Path, Path]:
    """``(engine.plan, engine.manifest.json)`` for this build configuration."""
    base = (directory or default_engine_dir()) / config.stem(artifact_stem, trt_version, capability)
    return base.with_suffix(".plan"), base.with_suffix(".manifest.json")


def _gpu_identity(cudart: Any, device_index: int) -> tuple[str, str, int, int]:
    props = cuda_call(cudart, cudart.cudaGetDeviceProperties(device_index))
    runtime_version = cuda_call(cudart, cudart.cudaRuntimeGetVersion())
    driver_version = cuda_call(cudart, cudart.cudaDriverGetVersion())
    return (
        props.name.decode() if isinstance(props.name, bytes) else str(props.name),
        f"{props.major}.{props.minor}",
        int(runtime_version),
        int(driver_version),
    )


def build_engine(
    config: TrtBuildConfig,
    artifact: Path,
    onnx_manifest: OnnxManifest,
    *,
    directory: Path | None = None,
    device_index: int = 0,
) -> tuple[Path, TensorRtManifest]:
    """Parse the canonical ONNX and build, serialize and record one engine.

    Raises:
        BackendError: the parser or the builder failed; every parser error is included.
    """
    trt, cudart = import_tensorrt()
    model_bytes = artifact.read_bytes()
    source_sha = hashlib.sha256(model_bytes).hexdigest()
    if source_sha != onnx_manifest.artifact.sha256:
        raise BackendError(
            f"ONNX artifact changed on disk (sha256 {source_sha[:16]}); refusing to build."
        )

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(model_bytes) or parser.num_errors:
        problems = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise BackendError(f"TensorRT could not parse the ONNX artifact: {problems or 'unknown'}")
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise BackendError(
            f"Expected one input and one output, got {network.num_inputs}/{network.num_outputs}."
        )

    builder_config = builder.create_builder_config()
    tf32_default = bool(builder_config.get_flag(trt.BuilderFlag.TF32))
    if config.precision == "ieee_fp32":
        # TensorRT enables TF32 by default: without this, "FP32" is TF32.
        builder_config.clear_flag(trt.BuilderFlag.TF32)
    if config.optimization_level is not None:
        builder_config.builder_optimization_level = config.optimization_level
    if config.workspace_bytes is not None:
        builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, config.workspace_bytes)
    if config.disable_timing_cache:
        builder_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)

    chw = tuple(int(d) for d in onnx_manifest.input.shape[1:])
    shapes = config.profile(chw)
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, tuple(shapes["min"]), tuple(shapes["opt"]), tuple(shapes["max"]))
    builder_config.add_optimization_profile(profile)

    settings: dict[str, str | int | bool | None] = {
        "tf32_enabled_by_default": tf32_default,
        "tf32_flag_after_policy": bool(builder_config.get_flag(trt.BuilderFlag.TF32)),
        "strict_nans": bool(builder_config.get_flag(trt.BuilderFlag.STRICT_NANS)),
        "disable_timing_cache": bool(builder_config.get_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)),
        "disable_compilation_cache": bool(
            builder_config.get_flag(trt.BuilderFlag.DISABLE_COMPILATION_CACHE)
        ),
        "builder_optimization_level": int(builder_config.builder_optimization_level),
        "workspace_pool_limit_bytes": int(
            builder_config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE)
        ),
        "strongly_typed_network": bool(
            network.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        ),
        "network_layers": int(network.num_layers),
        "timing_cache": "builder default (not supplied by this tool)",
    }

    started = time.perf_counter_ns()
    plan = builder.build_serialized_network(network, builder_config)
    build_seconds = (time.perf_counter_ns() - started) / 1e9
    if plan is None:
        raise BackendError(
            "TensorRT returned no engine for this network and configuration "
            f"(precision={config.precision}, profile={shapes})."
        )
    plan_bytes = bytes(plan)

    gpu_name, capability, runtime_version, driver_version = _gpu_identity(cudart, device_index)
    engine_path, manifest_path = engine_paths(
        config, artifact.stem, trt.__version__, capability, directory
    )
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan_bytes)

    manifest = TensorRtManifest(
        created_utc=utc_now_iso(),
        engine_file=engine_path.name,
        engine_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        engine_size_bytes=len(plan_bytes),
        build_seconds=build_seconds,
        source_onnx_sha256=source_sha,
        model=onnx_manifest.model,
        input=onnx_manifest.input,
        output=onnx_manifest.output,
        tensorrt_version=trt.__version__,
        cuda_runtime_version=runtime_version,
        cuda_driver_version=driver_version,
        gpu_name=gpu_name,
        compute_capability=capability,
        precision_policy=config.precision,
        profile=TrtProfile(min=shapes["min"], opt=shapes["opt"], max=shapes["max"]),
        builder_settings=settings,
        provenance=capture_provenance(),
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return engine_path, manifest


def load_manifest(path: Path) -> TensorRtManifest:
    return TensorRtManifest.model_validate_json(path.read_text(encoding="utf-8"))


def ensure_engine(
    config: TrtBuildConfig,
    *,
    directory: Path | None = None,
    allow_build: bool = True,
    allow_export: bool = True,
    allow_download: bool = True,
    device_index: int = 0,
) -> tuple[Path, TensorRtManifest, bool]:
    """Return a usable engine for this configuration, building it if necessary.

    Returns:
        ``(engine_path, manifest, built_now)``.

    Raises:
        UnavailableError: TensorRT is missing, or no engine exists and building is off.
        BackendError: the cached engine does not match its manifest.
    """
    trt, cudart = import_tensorrt()
    artifact, onnx_manifest, _ = ensure_artifact(
        ExportConfig(model=config.model, weights=config.weights, opset=config.opset),
        allow_export=allow_export,
        allow_download=allow_download,
    )
    _, capability, _, _ = _gpu_identity(cudart, device_index)
    engine_path, manifest_path = engine_paths(
        config, artifact.stem, trt.__version__, capability, directory
    )

    if engine_path.exists() and manifest_path.exists():
        cached = load_manifest(manifest_path)
        digest = hashlib.sha256(engine_path.read_bytes()).hexdigest()
        if digest != cached.engine_sha256:
            raise BackendError(
                f"Cached engine {engine_path.name} does not match its manifest "
                f"(sha256 {digest[:16]}); delete it and rebuild."
            )
        source_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if (
            cached.source_onnx_sha256 == source_sha
            and cached.tensorrt_version == trt.__version__
            and cached.compute_capability == capability
            and cached.precision_policy == config.precision
        ):
            return engine_path, cached, False

    if not allow_build:
        raise UnavailableError(
            f"No engine for this configuration at {engine_path} and building is disabled. "
            "Build one with the TensorRT backend, or enable allow_build."
        )
    built_path, manifest = build_engine(
        config, artifact, onnx_manifest, directory=directory, device_index=device_index
    )
    return built_path, manifest, True
