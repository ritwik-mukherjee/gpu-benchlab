"""TensorRT backend: engine build, device-resident execution and CUDA-event timing.

Design and rationale: ``docs/plans/phase-6-tensorrt.md``.

What this backend guarantees:

* **The same canonical ONNX artifact** every other backend uses, by SHA-256. TensorRT
  never gets its own export path.
* **The engine build is not inference.** Building happens in the ``build`` lifecycle
  phase and is recorded as ``engine_build_ms``; deserialization, context creation and
  buffer allocation happen in ``prepare``. None of it is inside the measured loop.
* **FP32 means IEEE FP32.** TensorRT 11 enables ``BuilderFlag.TF32`` by default, so an
  engine built without clearing it is TF32. The precision policy is applied at build
  time, recorded in the engine manifest, and re-recorded on every result.
* **Input and output stay on the device** for the whole measured loop, like ORT's
  IOBinding and PyTorch's device tensors. The only host copy is the sanity pass.
* **Timing matches the PyTorch contract**: CUDA events on the execution stream as the
  primary series, synchronized host time as an explicitly secondary one.
* **A shape outside the engine's optimization profile is ``unsupported``**, never
  silently clamped.

The CUDA runtime is reached through ``cuda-python`` rather than torch, so the backend
does not require a deep learning framework to run an engine.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
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
from gpu_benchlab.core.timing import NANOSECONDS_PER_MILLISECOND, Timer, TimingMechanism
from gpu_benchlab.export.tensorrt_build import (
    TensorRtManifest,
    TrtBuildConfig,
    cuda_call,
    ensure_engine,
    import_tensorrt,
)
from gpu_benchlab.hardware.capability import Precision
from gpu_benchlab.models.registry import get_model

if TYPE_CHECKING:
    from gpu_benchlab.hardware.types import EnvironmentReport

__all__ = ["TensorRtBackend", "TensorRtOptions", "TrtCudaEventTimer"]


class TensorRtOptions(BaseModel):
    """Options accepted under ``backend_options`` for ``backend: tensorrt``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_batch: int = Field(default=1, gt=0, description="Smallest batch the engine accepts.")
    opt_batch: int = Field(
        default=8,
        gt=0,
        description=(
            "Batch TensorRT optimises tactics for. One engine serves the whole matrix, so "
            "this can disadvantage other batch sizes; it is recorded on every result."
        ),
    )
    max_batch: int = Field(default=8, gt=0, description="Largest batch the engine accepts.")
    optimization_level: int | None = Field(default=None, ge=0, le=5)
    workspace_bytes: int | None = Field(default=None, gt=0)
    disable_timing_cache: bool = False
    allow_build: bool = Field(default=True, description="Build the engine if none is cached.")
    allow_export: bool = Field(default=True, description="Export the ONNX artifact if missing.")
    allow_download: bool = Field(default=True, description="Download pinned weights if missing.")


class TrtCudaEventTimer(Timer):
    """CUDA-event timer on the execution stream, mirroring the PyTorch contract.

    Per iteration::

        start():  synchronize the stream        -- previous work drained
                  host clock t0
                  record start event on the stream
        <execute>
        stop():   record end event on the stream
                  synchronize the end event     -- wait for THIS iteration
                  host clock t1
                  return elapsed_time(start, end)            # primary, device time

    ``t1 - t0`` is kept as a **secondary** host measurement, exactly as the PyTorch
    backend does, so the two backends are comparable on either basis.
    """

    def __init__(self, cudart: Any, stream: Any) -> None:
        self._cudart = cudart
        self._stream = stream
        self._start = cuda_call(cudart, cudart.cudaEventCreate())
        self._end = cuda_call(cudart, cudart.cudaEventCreate())
        self._host_start_ns: int | None = None
        self._host_samples: list[float] = []
        self.mechanism = TimingMechanism.CUDA_EVENT

    def start(self) -> None:
        cuda_call(self._cudart, self._cudart.cudaStreamSynchronize(self._stream))
        self._host_start_ns = time.perf_counter_ns()
        cuda_call(self._cudart, self._cudart.cudaEventRecord(self._start, self._stream))

    def stop(self) -> float:
        if self._host_start_ns is None:
            raise RuntimeError("Timer.stop() called before start().")
        cuda_call(self._cudart, self._cudart.cudaEventRecord(self._end, self._stream))
        cuda_call(self._cudart, self._cudart.cudaEventSynchronize(self._end))
        host_ms = (time.perf_counter_ns() - self._host_start_ns) / NANOSECONDS_PER_MILLISECOND
        self._host_start_ns = None
        self._host_samples.append(host_ms)
        return float(
            cuda_call(self._cudart, self._cudart.cudaEventElapsedTime(self._start, self._end))
        )

    def drain_secondary(self) -> tuple[TimingMechanism, list[float]] | None:
        samples, self._host_samples = self._host_samples, []
        return TimingMechanism.WALL_CLOCK_SYNCHRONIZED, samples

    def close(self) -> None:
        for event in (self._start, self._end):
            self._cudart.cudaEventDestroy(event)


class TensorRtBackend(Backend):
    """Inference through a TensorRT engine compiled from the canonical ONNX artifact."""

    def __init__(self, config: ExperimentConfig) -> None:
        try:
            self._options = TensorRtOptions.model_validate(config.backend_options)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            raise ConfigurationError(f"Invalid backend_options for tensorrt: {problems}") from exc
        if not (self._options.min_batch <= self._options.opt_batch <= self._options.max_batch):
            raise ConfigurationError(
                "TensorRT optimization profile must satisfy min_batch <= opt_batch <= "
                f"max_batch, got {self._options.min_batch}/{self._options.opt_batch}/"
                f"{self._options.max_batch}."
            )

        self._device_str, self._device_id = self._parse_device(config.device)
        self._spec = get_model(config.model.name)
        self._weights_id = self._spec.resolve_weights(config.model.weights)
        self._seed = config.model.seed
        self._precision = config.precision
        self._input_shape = tuple(config.model.input_shape or self._spec.input_shape)

        self._trt: Any = None
        self._cudart: Any = None
        self._engine_path: Any = None
        self._manifest: TensorRtManifest | None = None
        self._runtime: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._stream: Any = None
        self._device_input: Any = None
        self._device_output: Any = None
        self._timer: TrtCudaEventTimer | None = None
        self._settings: dict[str, SettingValue] = {
            "device": self._device_str,
            "requested_precision": self._precision.value,
            "input_generator": INPUT_GENERATOR,
            "input_seed": self._seed,
        }

    @staticmethod
    def _parse_device(device: str | None) -> tuple[str, int]:
        if device is None:
            raise ConfigurationError(
                "backend 'tensorrt' requires an explicit CUDA device ('cuda' or 'cuda:N'). "
                "TensorRT has no CPU execution path."
            )
        normalized = device.strip().lower()
        if normalized == "cuda":
            return "cuda:0", 0
        if normalized.startswith("cuda:") and normalized[5:].isdigit():
            return normalized, int(normalized[5:])
        raise ConfigurationError(
            f"Unrecognised device {device!r} for backend 'tensorrt'. TensorRT runs on CUDA "
            "only: use 'cuda' or 'cuda:N'."
        )

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            name="tensorrt",
            version=self._trt.__version__ if self._trt is not None else None,
            device=self._device_str,
            device_kind=DeviceKind.CUDA,
            settings=dict(self._settings),
            is_simulated=False,
            detail="TensorRT engine execution. Primary timing: CUDA events on the stream.",
        )

    @property
    def model_info(self) -> ModelInfo | None:
        return self._manifest.model if self._manifest is not None else None

    # -- validate ----------------------------------------------------------------------

    def validate(self, config: Any, environment: EnvironmentReport) -> None:
        trt, cudart = import_tensorrt()
        self._trt, self._cudart = trt, cudart
        self._settings["tensorrt_version"] = trt.__version__

        if self._precision is not Precision.FP32:
            raise UnsupportedConfigurationError(
                f"Precision {self._precision.value} is not implemented for the tensorrt "
                "backend yet: Phase 6 builds IEEE FP32 engines only (TensorRT 11 is "
                "strongly typed, so other precisions need a different artifact or "
                "explicit cast layers)."
            )

        count = cuda_call(cudart, cudart.cudaGetDeviceCount())
        if not count:
            raise UnavailableError(
                "TensorRT requires a CUDA device and the CUDA runtime reports none. "
                "Not falling back to CPU."
            )
        if self._device_id >= int(count):
            raise UnavailableError(
                f"Requested {self._device_str}, but only {int(count)} CUDA device(s) are "
                "visible. Not falling back to CPU."
            )
        cuda_call(cudart, cudart.cudaSetDevice(self._device_id))
        props = cuda_call(cudart, cudart.cudaGetDeviceProperties(self._device_id))
        self._settings["cuda_device_name"] = (
            props.name.decode() if isinstance(props.name, bytes) else str(props.name)
        )
        self._settings["cuda_capability"] = f"{props.major}.{props.minor}"
        self._settings["cuda_runtime_version"] = int(
            cuda_call(cudart, cudart.cudaRuntimeGetVersion())
        )
        self._settings["cuda_driver_version"] = int(
            cuda_call(cudart, cudart.cudaDriverGetVersion())
        )

        batch = config.batch_size if isinstance(config, ExperimentConfig) else 1
        if not (self._options.min_batch <= batch <= self._options.max_batch):
            raise UnsupportedConfigurationError(
                f"batch_size {batch} lies outside the engine's optimization profile "
                f"[{self._options.min_batch}, {self._options.max_batch}]. Widen the "
                "profile in backend_options; the shape is never silently clamped."
            )

        self._build_config = TrtBuildConfig(
            model=self._spec.name,
            weights=self._weights_id,
            min_batch=self._options.min_batch,
            opt_batch=self._options.opt_batch,
            max_batch=self._options.max_batch,
            precision="ieee_fp32",
            optimization_level=self._options.optimization_level,
            workspace_bytes=self._options.workspace_bytes,
            disable_timing_cache=self._options.disable_timing_cache,
        )
        self._settings["precision_policy"] = "ieee_fp32 (BuilderFlag.TF32 cleared at build)"
        self._settings["profile_min_batch"] = self._options.min_batch
        self._settings["profile_opt_batch"] = self._options.opt_batch
        self._settings["profile_max_batch"] = self._options.max_batch

    # -- lifecycle ---------------------------------------------------------------------

    def load(self) -> None:
        """Obtain the engine: the ONNX artifact is fetched and verified here."""
        path, manifest, built = ensure_engine(
            self._build_config,
            allow_build=self._options.allow_build,
            allow_export=self._options.allow_export,
            allow_download=self._options.allow_download,
            device_index=self._device_id,
        )
        self._engine_path, self._manifest = path, manifest
        self._settings["engine_built_this_run"] = built
        self._settings["engine_sha256"] = manifest.engine_sha256
        self._settings["engine_size_bytes"] = manifest.engine_size_bytes
        self._settings["engine_build_seconds"] = round(manifest.build_seconds, 3)
        self._settings["source_onnx_sha256"] = manifest.source_onnx_sha256
        for key, value in manifest.builder_settings.items():
            self._settings[f"builder.{key}"] = value

    def build(self) -> None:
        """Deserialize the engine. This is TensorRT's engine-load step, never inference."""
        if self._manifest is None or self._engine_path is None:
            raise BackendError("build() called before load().")
        started = time.perf_counter_ns()
        self._runtime = self._trt.Runtime(self._trt.Logger(self._trt.Logger.WARNING))
        self._engine = self._runtime.deserialize_cuda_engine(self._engine_path.read_bytes())
        if self._engine is None:
            raise BackendError(
                f"TensorRT could not deserialize {self._engine_path.name}. An engine is "
                "specific to its TensorRT version and GPU architecture."
            )
        self._settings["engine_deserialization_ms"] = round(
            (time.perf_counter_ns() - started) / NANOSECONDS_PER_MILLISECOND, 3
        )
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise BackendError("TensorRT returned no execution context for this engine.")

    def prepare(self, config: Any) -> Any:
        if self._context is None or self._manifest is None:
            raise BackendError("prepare() called before build().")
        if not isinstance(config, ExperimentConfig):
            raise BackendError(
                f"prepare() expects an ExperimentConfig, got {type(config).__name__}."
            )
        cudart = self._cudart
        shape = (config.batch_size, *self._input_shape)
        self._settings["input_shape"] = "x".join(str(d) for d in shape)

        profile = self._engine.get_tensor_profile_shape(self._manifest.input.name, 0)
        recorded = [list(s) for s in profile]
        self._settings["engine_profile"] = "/".join(str(s) for s in recorded)
        if not (recorded[0][0] <= config.batch_size <= recorded[2][0]):
            raise UnsupportedConfigurationError(
                f"batch_size {config.batch_size} is outside the engine's profile {recorded}."
            )

        host_input = synthetic_input(shape, self._seed)
        out_shape = (config.batch_size, *self._spec.output_shape)
        # Pre-filled with NaN, not np.empty: the sanity check below rejects non-finite
        # values, so a device-to-host copy that never happens fails deterministically
        # rather than passing whenever uninitialised memory looks finite.
        self._host_output = np.full(out_shape, np.nan, dtype=np.float32)
        self._device_input = cuda_call(cudart, cudart.cudaMalloc(host_input.nbytes))
        self._device_output = cuda_call(cudart, cudart.cudaMalloc(self._host_output.nbytes))
        cuda_call(
            cudart,
            cudart.cudaMemcpy(
                self._device_input,
                host_input.ctypes.data,
                host_input.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            ),
        )
        self._stream = cuda_call(cudart, cudart.cudaStreamCreate())
        self._context.set_input_shape(self._manifest.input.name, shape)
        self._context.set_tensor_address(self._manifest.input.name, int(self._device_input))
        self._context.set_tensor_address(self._manifest.output.name, int(self._device_output))
        self._settings["io_device_resident"] = True

        # Sanity pass, outside all timing: prove the engine really runs before measuring.
        if not self._context.execute_async_v3(self._stream):
            raise BackendError("TensorRT refused to enqueue the sanity inference.")
        cuda_call(cudart, cudart.cudaStreamSynchronize(self._stream))
        cuda_call(
            cudart,
            cudart.cudaMemcpy(
                self._host_output.ctypes.data,
                self._device_output,
                self._host_output.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            ),
        )
        if tuple(self._host_output.shape) != out_shape:
            raise BackendError(
                f"Sanity inference returned {self._host_output.shape}, expected {out_shape}."
            )
        if not bool(np.isfinite(self._host_output).all()):
            raise BackendError("Sanity inference produced non-finite outputs (NaN/inf).")
        self._settings["sanity_check"] = "passed"
        return self._stream

    def execute(self, inputs: Any) -> Any:
        return self._context.execute_async_v3(inputs)

    def synchronize(self) -> None:
        if self._stream is not None:
            cuda_call(self._cudart, self._cudart.cudaStreamSynchronize(self._stream))

    def make_timer(self) -> Timer:
        self._timer = TrtCudaEventTimer(self._cudart, self._stream)
        return self._timer

    def close(self) -> None:
        cudart = self._cudart
        if cudart is not None:
            if self._timer is not None:
                self._timer.close()
            for buffer in (self._device_input, self._device_output):
                if buffer is not None:
                    cudart.cudaFree(buffer)
            if self._stream is not None:
                cudart.cudaStreamDestroy(self._stream)
        self._timer = None
        self._device_input = self._device_output = self._stream = None
        self._context = self._engine = self._runtime = None
