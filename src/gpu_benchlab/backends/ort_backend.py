"""ONNX Runtime backend: an exported ONNX artifact on the CPU or CUDA execution provider.

Design: docs/plans/phase-4-onnxruntime.md and ADR 0007.

Lifecycle mapping (same engine, same phases as PyTorch):

* ``validate`` -- ORT importable; exactly one explicitly selected EP; precision
  valid; artifact present and structurally validated (exported now if missing,
  untimed).
* ``load`` -- read the artifact bytes from disk and re-verify their SHA-256.
* ``build`` -- create the ``InferenceSession`` from those bytes: graph
  optimisation, EP partitioning, kernel set-up. ORT's equivalent of an engine
  build, timed as ``engine_build_ms`` so it never leaks into inference latency.
  Then verify the requested EP is really active (see below).
* ``prepare`` -- inputs, a sanity run, and a measured node-placement probe.
* ``execute`` -- one ``session.run`` (CPU) or ``run_with_iobinding`` (CUDA).

No silent fallback -- verified against the real onnxruntime-gpu 1.30 package on a
machine without CUDA libraries (ENGINEERING_LOG 2026-09-19):

* ``get_available_providers()`` listed ``CUDAExecutionProvider`` although its DLL
  could not load, so the pre-check alone is insufficient;
* requesting only ``CUDAExecutionProvider`` *created a session that ran on CPU*,
  with only a stderr warning.

Hence three checks: availability before; ``session.get_providers()`` after
(missing EP -> ``unavailable``); and measured per-node placement (any node on the
CPU EP under a CUDA request -> ``unsupported``, unless explicitly allowed).

The CUDA path is implemented but **unverified on NVIDIA hardware**.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gpu_benchlab.core.backend import (
    Backend,
    BackendDescriptor,
    DeviceKind,
    ModelInfo,
    SettingValue,
)
from gpu_benchlab.core.config import ExperimentConfig
from gpu_benchlab.core.errors import (
    BackendError,
    ConfigurationError,
    UnavailableError,
    UnsupportedConfigurationError,
)
from gpu_benchlab.core.inputs import INPUT_GENERATOR, synthetic_input
from gpu_benchlab.core.timing import Timer, WallClockTimer
from gpu_benchlab.export.onnx_export import (
    DEFAULT_OPSET,
    ExportConfig,
    OnnxManifest,
    ensure_artifact,
)
from gpu_benchlab.hardware.capability import Precision, precision_support
from gpu_benchlab.models.registry import get_model

if TYPE_CHECKING:
    from gpu_benchlab.hardware.types import EnvironmentReport

__all__ = [
    "CPU_EP",
    "CUDA_EP",
    "OnnxRuntimeBackend",
    "OrtOptions",
    "create_session",
    "measure_placement",
    "session_options",
]

CPU_EP = "CPUExecutionProvider"
CUDA_EP = "CUDAExecutionProvider"

_OPT_LEVELS = {
    "disable": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}


class OrtOptions(BaseModel):
    """``backend_options`` accepted by ``backend: onnxruntime``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    opset: int = Field(default=DEFAULT_OPSET, ge=7)
    dynamic_batch: bool = True
    allow_export: bool = Field(default=True, description="Export the artifact if missing.")
    allow_download: bool = Field(default=True, description="Download pinned weights for export.")
    graph_optimization_level: Literal["disable", "basic", "extended", "all"] = "all"
    intra_op_num_threads: int | None = Field(default=None, gt=0)
    inter_op_num_threads: int | None = Field(default=None, gt=0)
    measure_placement: bool = Field(
        default=True, description="Profile one untimed run to record per-node EP placement."
    )
    allow_partial_placement: bool = Field(
        default=False,
        description="CUDA only: tolerate nodes placed on the CPU EP (recorded) instead of "
        "reporting the configuration unsupported.",
    )
    cudnn_conv_algo_search: Literal["EXHAUSTIVE", "HEURISTIC", "DEFAULT"] = Field(
        default="EXHAUSTIVE", description="CUDA EP option; EXHAUSTIVE is ORT's default."
    )
    use_io_binding: bool = Field(
        default=True,
        description="CUDA only: keep input/output on the device so run() does not time "
        "host<->device copies the PyTorch path does not perform.",
    )


def _import_ort() -> Any:
    try:
        import onnxruntime
    except ImportError as exc:
        raise UnavailableError(
            "onnxruntime is not installed, so backend 'onnxruntime' cannot run. Install the "
            "[onnx-cpu] or [onnx] extra."
        ) from exc
    return onnxruntime


def preload_cuda_libraries(ort: Any) -> str:
    """Load CUDA/cuDNN from the installed ``nvidia-*`` wheels. Returns what happened.

    Without this, ONNX Runtime's CUDA EP builds a session that reports itself active
    and then fails at the first Conv with ``dlopen failed for libcudnn.so`` -- unless
    something else in the process (importing torch) already loaded cuDNN. Observed on
    an NVIDIA L4 with onnxruntime-gpu 1.30 (2026-09-20): the same script passed with
    torch imported first and failed without it. Relying on import order is not a
    contract, so the backend loads the libraries itself.
    """
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return "unavailable: this onnxruntime has no preload_dlls"
    try:
        preload()
    except Exception as exc:  # noqa: BLE001 - never fail a run over a best-effort preload
        return f"failed: {type(exc).__name__}: {exc}"
    return "ok"


def session_options(ort: Any, options: OrtOptions, *, profile_prefix: str | None = None) -> Any:
    """The SessionOptions every session for these options uses (benchmark, probe, verify)."""
    so = ort.SessionOptions()
    so.graph_optimization_level = getattr(
        ort.GraphOptimizationLevel, _OPT_LEVELS[options.graph_optimization_level]
    )
    if options.intra_op_num_threads is not None:
        so.intra_op_num_threads = options.intra_op_num_threads
    if options.inter_op_num_threads is not None:
        so.inter_op_num_threads = options.inter_op_num_threads
    so.log_severity_level = 3  # errors only; EP load warnings are checked explicitly instead
    if profile_prefix is not None:
        so.enable_profiling = True
        so.profile_file_prefix = profile_prefix
    return so


def create_session(
    ort: Any,
    model: bytes | str,
    provider: str,
    provider_options: dict[str, str],
    options: OrtOptions,
    *,
    profile_prefix: str | None = None,
) -> Any:
    """Create a session with exactly one EP and verify that EP is actually active.

    Raises:
        UnavailableError: the requested EP is not active in the created session --
            ORT substituted another EP (observed with onnxruntime-gpu on a machine
            without CUDA: the session silently ran on CPU).
    """
    if provider == CUDA_EP:
        preload_cuda_libraries(ort)
    so = session_options(ort, options, profile_prefix=profile_prefix)
    session = ort.InferenceSession(model, so, providers=[(provider, provider_options)])
    active = list(session.get_providers())
    if provider not in active:
        raise UnavailableError(
            f"Requested {provider}, but the created session is using {active}: the EP failed "
            "to load and ONNX Runtime substituted another one. Not falling back."
        )
    return session


def measure_placement(
    ort: Any,
    model: bytes | str,
    provider: str,
    provider_options: dict[str, str],
    options: OrtOptions,
    feed: dict[str, Any],
) -> dict[str, int]:
    """Count executed nodes per EP by profiling one run in a separate, identical session.

    Placement is decided at session creation from the model, options and EPs, so a
    session with identical settings plus profiling places nodes identically. The
    benchmark session itself is never profiled (profiling costs time on every run).
    """
    with tempfile.TemporaryDirectory() as tmp:
        session = create_session(
            ort,
            model,
            provider,
            provider_options,
            options,
            profile_prefix=str(Path(tmp) / "placement"),
        )
        session.run(None, feed)
        profile = Path(session.end_profiling())
        events = json.loads(profile.read_text(encoding="utf-8"))
        del session
    counts: dict[str, int] = {}
    for event in events:
        if event.get("cat") != "Node":
            continue
        ep = event.get("args", {}).get("provider")
        if ep and event.get("name", "").endswith("_kernel_time"):
            counts[ep] = counts.get(ep, 0) + 1
    return counts


class OnnxRuntimeBackend(Backend):
    """ONNX Runtime inference for registry models exported to ONNX (Phase 4: ResNet-50)."""

    def __init__(self, config: ExperimentConfig) -> None:
        try:
            self._options = OrtOptions.model_validate(config.backend_options)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            raise ConfigurationError(
                f"Invalid backend_options for onnxruntime: {problems}"
            ) from exc

        self._device_str, self._device_kind, self._device_id = self._parse_device(config.device)
        self._spec = get_model(config.model.name)
        self._weights_id = self._spec.resolve_weights(config.model.weights)
        self._precision = config.precision
        self._seed = config.model.seed
        self._input_shape = tuple(config.model.input_shape or self._spec.input_shape)
        self._export_config = ExportConfig(
            model=self._spec.name,
            weights=self._weights_id,
            opset=self._options.opset,
            dynamic_batch=self._options.dynamic_batch,
            trace_batch_size=1 if self._options.dynamic_batch else config.batch_size,
        )
        self._provider = CUDA_EP if self._device_kind is DeviceKind.CUDA else CPU_EP

        self._ort: Any = None
        self._artifact: Path | None = None
        self._manifest: OnnxManifest | None = None
        self._model_bytes: bytes | None = None
        self._session: Any = None
        self._provider_options: dict[str, str] = {}
        self._runner: Callable[[Any], Any] | None = None
        self._device_input: Any = None  # CUDA + IOBinding only; see prepare()
        self._settings: dict[str, SettingValue] = {
            "device": self._device_str,
            "requested_provider": self._provider,
            "requested_precision": self._precision.value,
            "graph_optimization_level": self._options.graph_optimization_level,
            "intra_op_num_threads": self._options.intra_op_num_threads,
            "inter_op_num_threads": self._options.inter_op_num_threads,
            "input_generator": INPUT_GENERATOR,
            "input_seed": self._seed,
        }

    @staticmethod
    def _parse_device(device: str | None) -> tuple[str, DeviceKind, int]:
        if device is None:
            raise ConfigurationError(
                "backend 'onnxruntime' requires an explicit device ('cpu', 'cuda' or "
                "'cuda:N'). There is no implicit choice and no fallback between them."
            )
        normalized = device.strip().lower()
        if normalized == "cpu":
            return "cpu", DeviceKind.CPU, 0
        if normalized == "cuda":
            return "cuda:0", DeviceKind.CUDA, 0
        if normalized.startswith("cuda:") and normalized[5:].isdigit():
            return normalized, DeviceKind.CUDA, int(normalized[5:])
        raise ConfigurationError(
            f"Unrecognised device {device!r} for backend 'onnxruntime'. "
            "Use 'cpu', 'cuda' or 'cuda:N'."
        )

    # -- identity ----------------------------------------------------------------------

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            name="onnxruntime",
            version=self._ort.__version__ if self._ort is not None else None,
            execution_provider=self._provider,
            device=self._device_str,
            device_kind=self._device_kind,
            settings=dict(self._settings),
            is_simulated=False,
            detail=(
                "ONNX Runtime, CPU execution provider. Real measurements of the host CPU, "
                "not GPU performance."
                if self._device_kind is DeviceKind.CPU
                else "ONNX Runtime, CUDA execution provider (implemented, unverified on "
                "NVIDIA hardware)."
            ),
        )

    @property
    def model_info(self) -> ModelInfo | None:
        return self._manifest.model if self._manifest is not None else None

    # -- validate ----------------------------------------------------------------------

    def validate(self, config: Any, environment: EnvironmentReport) -> None:
        ort = _import_ort()
        self._ort = ort
        self._settings["onnxruntime_version"] = ort.__version__

        if self._precision not in (Precision.FP32, Precision.TF32):
            raise UnsupportedConfigurationError(
                f"Precision {self._precision.value} is not supported by the onnxruntime "
                "backend: the exported artifact is an FP32 graph. A reduced-precision graph "
                "is a different artifact and is out of scope for Phase 4."
            )

        available = list(ort.get_available_providers())
        self._settings["available_providers"] = ",".join(available)
        if self._provider not in available:
            raise UnavailableError(
                f"Requested {self._provider}, but the installed onnxruntime package does not "
                f"provide it (available: {available}). "
                + (
                    "Install onnxruntime-gpu for CUDA. Not falling back to CPU."
                    if self._provider == CUDA_EP
                    else ""
                )
            )

        if self._device_kind is DeviceKind.CUDA:
            # Recorded, because whether cuDNN was loadable decides whether the EP can
            # execute at all -- a session can report CUDA active and still fail at the
            # first Conv (observed on an L4).
            self._settings["cuda_libraries_preloaded"] = preload_cuda_libraries(ort)
            self._provider_options = self._cuda_provider_options(environment)
        elif self._precision is Precision.TF32:
            raise UnsupportedConfigurationError(
                "TF32 is a CUDA tensor-core execution mode; it does not exist on CPU."
            )
        for key, value in self._provider_options.items():
            self._settings[f"provider_option.{key}"] = value

        artifact, manifest, exported = ensure_artifact(
            self._export_config,
            allow_export=self._options.allow_export,
            allow_download=self._options.allow_download,
        )
        self._artifact, self._manifest = artifact, manifest
        self._settings.update(
            {
                "artifact_exported_this_run": exported,
                "artifact_filename": manifest.artifact.filename,
                "artifact_sha256": manifest.artifact.sha256,
                "opset": manifest.exporter.opset,
                "dynamic_batch": bool(manifest.exporter.dynamic_shapes),
                "exporter_torch_version": manifest.exporter.torch_version,
                "exporter_onnx_version": manifest.exporter.onnx_version,
                "input_name": manifest.input.name,
                "input_dtype": manifest.input.dtype,
            }
        )

    def _cuda_provider_options(self, environment: EnvironmentReport) -> dict[str, str]:
        """CUDA EP options. TF32 is ORT's default and is switched off unless requested."""
        if self._precision is Precision.TF32:
            # Capability from NVML. NVML enumerates in PCI order; CUDA may not (unless
            # CUDA_DEVICE_ORDER=PCI_BUS_ID), so index mapping is recorded as a caveat.
            gpus = {g.index: g for g in environment.gpus}
            gpu = gpus.get(self._device_id)
            if gpu is None or gpu.compute_capability_major is None:
                raise UnsupportedConfigurationError(
                    f"TF32 requires SM >= 8.0, and the compute capability of GPU "
                    f"{self._device_id} could not be determined from NVML."
                )
            support = {
                p.precision: p
                for p in precision_support(
                    gpu.compute_capability_major, gpu.compute_capability_minor or 0
                )
            }[Precision.TF32]
            if not support.supported:
                raise UnsupportedConfigurationError(
                    f"tf32 on {gpu.name} (SM {gpu.compute_capability}): {support.note}"
                )
            self._settings["cuda_capability_source"] = "nvml (index order may differ from CUDA)"
        return {
            "device_id": str(self._device_id),
            "use_tf32": "1" if self._precision is Precision.TF32 else "0",
            "cudnn_conv_algo_search": self._options.cudnn_conv_algo_search,
        }

    # -- lifecycle ---------------------------------------------------------------------

    def load(self) -> None:
        """Read the artifact from disk and confirm the bytes are the validated ones."""
        if self._artifact is None or self._manifest is None:
            raise BackendError("load() called before validate().")
        data = self._artifact.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != self._manifest.artifact.sha256:
            raise BackendError(
                f"Artifact changed on disk between validation and load (sha256 {digest[:16]})."
            )
        self._model_bytes = data

    def build(self) -> None:
        """Create the InferenceSession: ORT's engine build. Verifies the EP is active."""
        if self._model_bytes is None:
            raise BackendError("build() called before load().")
        self._session = create_session(
            self._ort, self._model_bytes, self._provider, self._provider_options, self._options
        )
        self._settings["active_providers"] = ",".join(self._session.get_providers())
        active_options = self._session.get_provider_options().get(self._provider, {})
        for key in ("use_tf32", "cudnn_conv_algo_search", "device_id"):
            if key in active_options:
                self._settings[f"active_provider_option.{key}"] = str(active_options[key])

    def prepare(self, config: Any) -> Any:
        if self._session is None or self._manifest is None:
            raise BackendError("prepare() called before build().")
        if not isinstance(config, ExperimentConfig):
            raise BackendError(
                f"prepare() expects an ExperimentConfig, got {type(config).__name__}."
            )

        shape = (config.batch_size, *self._input_shape)
        self._settings["input_shape"] = "x".join(str(d) for d in shape)
        array = synthetic_input(shape, self._seed)
        name = self._manifest.input.name
        output = self._manifest.output.name
        feed = {name: array}

        if self._options.measure_placement:
            assert self._model_bytes is not None  # noqa: S101 - set in load()
            counts = measure_placement(
                self._ort,
                self._model_bytes,
                self._provider,
                self._provider_options,
                self._options,
                feed,
            )
            for ep, n in counts.items():
                self._settings[f"node_placement.{ep}"] = n
            self._settings["node_placement_method"] = "ORT profiling, identical session options"
            foreign = {ep: n for ep, n in counts.items() if ep != self._provider}
            partial = foreign and self._device_kind is DeviceKind.CUDA
            if partial and not self._options.allow_partial_placement:
                raise UnsupportedConfigurationError(
                    f"{sum(foreign.values())} node(s) were placed on {sorted(foreign)} "
                    f"instead of {self._provider}; the model cannot run entirely on the "
                    "requested EP. Set allow_partial_placement to benchmark it anyway "
                    "(placement is recorded either way)."
                )

        if self._device_kind is DeviceKind.CUDA and self._options.use_io_binding:
            ort = self._ort
            binding = self._session.io_binding()
            device_input = ort.OrtValue.ortvalue_from_numpy(array, "cuda", self._device_id)
            # Kept so the device-resident input can be read back for evidence: IOBinding
            # exposes bound outputs but not bound inputs. Holding the reference also
            # makes this side's ownership of the device buffer explicit.
            self._device_input = device_input
            binding.bind_ortvalue_input(name, device_input)
            binding.bind_output(output, "cuda", self._device_id)
            session = self._session
            self._runner = session.run_with_iobinding
            self._settings["io_binding"] = True
            self._run_once_and_check(binding, shape, lambda: binding.copy_outputs_to_cpu()[0])
            return binding

        run = self._session.run
        self._runner = lambda f: run([output], f)
        self._settings["io_binding"] = False
        self._run_once_and_check(feed, shape, None)
        return feed

    def _run_once_and_check(
        self, inputs: Any, shape: tuple[int, ...], fetch: Callable[[], Any] | None
    ) -> None:
        """Sanity run, outside all timing: right shape, dtype and finite values."""
        assert self._runner is not None  # noqa: S101 - set just before
        result = self._runner(inputs)
        output = fetch() if fetch is not None else result[0]
        import numpy as np

        expected = (shape[0], *self._spec.output_shape)
        if tuple(output.shape) != expected:
            raise BackendError(
                f"Sanity run returned shape {tuple(output.shape)}, expected {expected}."
            )
        if output.dtype != np.float32:
            raise BackendError(f"Sanity run returned {output.dtype}, expected float32.")
        if not bool(np.isfinite(output).all()):
            raise BackendError("Sanity run produced non-finite outputs (NaN/inf).")
        self._settings["sanity_check"] = "passed"

    def execute(self, inputs: Any) -> Any:
        return self._runner(inputs)  # type: ignore[misc]

    def make_timer(self) -> Timer:
        # session.run returns after execution completes. On CPU that is plainly
        # synchronous. On the CUDA EP it relies on ORT synchronizing its stream at
        # the end of Run (the default) -- unverified on NVIDIA hardware.
        self._settings["timing_note"] = (
            "host wall clock around session.run; ORT returns after execution completes"
            + (
                ""
                if self._device_kind is DeviceKind.CPU
                else " (CUDA EP: the end-of-Run stream synchronization was verified on an "
                "NVIDIA L4, 2026-09-20 -- see docs/limitations.md)"
            )
        )
        return WallClockTimer()

    def close(self) -> None:
        self._session = None
        self._runner = None
        self._model_bytes = None
        self._device_input = None
