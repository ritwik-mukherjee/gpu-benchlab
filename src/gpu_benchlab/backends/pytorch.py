"""PyTorch backend: eager-mode inference on CPU or CUDA.

Design and rationale: ``docs/plans/phase-3-pytorch-backend.md`` and ADR 0005.

What this backend guarantees:

* **No implicit device and no fallback.** The device must be named. A request for
  CUDA on a machine without it produces ``status: unavailable`` -- it is never
  quietly run on the CPU.
* **Precision is explicit and verified.** Weights and inputs are cast to the
  requested dtype, and the sanity forward pass checks the output actually came
  back in that dtype.
* **FP32 means IEEE FP32.** PyTorch's cuDNN convolutions default to TF32 on
  Ampere and newer (``cudnn.conv.fp32_precision == "tf32"`` out of the box --
  confirmed on torch 2.14). This backend sets conv, RNN and matmul FP32
  precision explicitly for every run, and records the effective values.
* **The model is what it claims to be.** Class name and parameter count are
  checked against the registry after every load.
* **Process-global torch state is restored** on ``close()``, including after a
  failure, so one experiment cannot leak its settings into the next.
* **CUDA timing uses CUDA events** (device time) as the primary measurement.
  Synchronized host wall time is kept only as an explicitly secondary one.

The CUDA path is implemented but **unverified on real NVIDIA hardware** -- see
``docs/limitations.md``.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

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
from gpu_benchlab.core.timing import (
    NANOSECONDS_PER_MILLISECOND,
    Timer,
    TimingMechanism,
    WallClockTimer,
)
from gpu_benchlab.hardware.capability import Precision, precision_support
from gpu_benchlab.models.registry import RANDOM_WEIGHTS, ModelSpec, get_model
from gpu_benchlab.models.weights import ResolvedWeights, ensure_weights

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from gpu_benchlab.hardware.types import EnvironmentReport

__all__ = ["CudaEventTimer", "PyTorchBackend", "PyTorchOptions"]

_QUANTIZED = {Precision.INT8, Precision.FP8, Precision.INT4, Precision.FP4}

# The new (PyTorch >= 2.9) FP32-precision API. Once any of these is written,
# reading the legacy `allow_tf32` flags can raise, so this backend never reads the
# legacy flags when the new API exists (verified on torch 2.14).
_FP32_PRECISION_NODES = ("global", "cudnn", "cudnn.conv", "cudnn.rnn", "cuda.matmul")


class PyTorchOptions(BaseModel):
    """Backend options accepted under ``backend_options`` for ``backend: pytorch``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cudnn_benchmark: bool = Field(
        default=True,
        description=(
            "Let cuDNN autotune convolution algorithms for the fixed input shape. "
            "Autotuning happens during warmup, which is one reason warmup exists."
        ),
    )
    channels_last: bool = Field(default=False, description="Use NHWC memory format.")
    num_threads: int | None = Field(
        default=None, gt=0, description="CPU intra-op threads. None = PyTorch's default."
    )
    allow_download: bool = Field(
        default=True, description="Download pinned weights if they are not cached."
    )


class CudaEventTimer(Timer):
    """Times each iteration with CUDA events on the execution stream.

    Contract per iteration::

        start():  synchronize the device       -- previous work fully drained
                  host clock t0
                  start_event.record(stream)
        <execute>
        stop():   end_event.record(stream)
                  end_event.synchronize()      -- wait for THIS iteration only
                  host clock t1
                  return start_event.elapsed_time(end_event)    # primary

    The primary sample is **device time between the two events on the stream**.
    It is not the sum of kernel durations: if the host launches kernels more
    slowly than the GPU runs them, the GPU idles between kernels and that idle
    time is inside the interval. That is the honest meaning of "how long did the
    GPU take to finish this iteration", and it is documented as such.

    ``t1 - t0`` (host time, including launch overhead and the synchronization)
    is kept as a **secondary** measurement only, drained by the engine after the
    loop. It is never a substitute for the event time.

    One event pair is reused each iteration; that is safe because ``stop()``
    synchronizes on the end event before returning.
    """

    def __init__(self, torch_module: Any, device: Any) -> None:
        self._torch = torch_module
        self._device = device
        self._stream = torch_module.cuda.current_stream(device)
        self._start_event = torch_module.cuda.Event(enable_timing=True)
        self._end_event = torch_module.cuda.Event(enable_timing=True)
        self._host_start_ns: int | None = None
        self._host_samples: list[float] = []
        self.mechanism = TimingMechanism.CUDA_EVENT

    def start(self) -> None:
        self._torch.cuda.synchronize(self._device)
        self._host_start_ns = time.perf_counter_ns()
        self._start_event.record(self._stream)

    def stop(self) -> float:
        if self._host_start_ns is None:
            raise RuntimeError("Timer.stop() called before start().")
        self._end_event.record(self._stream)
        self._end_event.synchronize()
        host_ms = (time.perf_counter_ns() - self._host_start_ns) / NANOSECONDS_PER_MILLISECOND
        self._host_start_ns = None
        self._host_samples.append(host_ms)
        return float(self._start_event.elapsed_time(self._end_event))

    def drain_secondary(self) -> tuple[TimingMechanism, list[float]] | None:
        samples, self._host_samples = self._host_samples, []
        return TimingMechanism.WALL_CLOCK_SYNCHRONIZED, samples


class PyTorchBackend(Backend):
    """Eager PyTorch inference for registry models (Phase 3: ResNet-50).

    Construction validates everything that does not need torch (options, device
    string, model name, weights id) and raises :class:`ConfigurationError`, so a
    malformed config fails before anything runs. Everything that needs torch or
    hardware happens in :meth:`validate`.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        try:
            self._options = PyTorchOptions.model_validate(config.backend_options)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            raise ConfigurationError(f"Invalid backend_options for pytorch: {problems}") from exc

        self._device_str, self._device_kind = self._parse_device(config.device)
        self._spec: ModelSpec = get_model(config.model.name)
        self._weights_id = self._spec.resolve_weights(config.model.weights)
        self._seed = config.model.seed
        self._precision = config.precision
        self._input_shape = tuple(config.model.input_shape or self._spec.input_shape)

        self._torch: Any = None
        self._model: Any = None
        self._device: Any = None
        self._dtype: Any = None
        self._resolved_weights: ResolvedWeights | None = None
        self._model_info: ModelInfo | None = None
        self._saved_state: dict[str, Any] | None = None
        self._settings: dict[str, SettingValue] = {
            "device": self._device_str,
            "precision_mode": "cast",
            "requested_precision": self._precision.value,
        }

    # -- identity ----------------------------------------------------------------------

    @staticmethod
    def _parse_device(device: str | None) -> tuple[str, DeviceKind]:
        if device is None:
            raise ConfigurationError(
                "backend 'pytorch' requires an explicit device ('cpu', 'cuda' or "
                "'cuda:N'). There is no implicit choice and no fallback between them."
            )
        normalized = device.strip().lower()
        if normalized == "cpu":
            return "cpu", DeviceKind.CPU
        if normalized == "cuda":
            return "cuda:0", DeviceKind.CUDA
        if normalized.startswith("cuda:") and normalized[5:].isdigit():
            return normalized, DeviceKind.CUDA
        raise ConfigurationError(
            f"Unrecognised device {device!r} for backend 'pytorch'. Use 'cpu', 'cuda' or 'cuda:N'."
        )

    @property
    def descriptor(self) -> BackendDescriptor:
        version = self._torch.__version__ if self._torch is not None else None
        return BackendDescriptor(
            name="pytorch",
            version=version,
            device=self._device_str,
            device_kind=self._device_kind,
            settings=dict(self._settings),
            is_simulated=False,
            detail=(
                "Eager PyTorch. CPU results are real measurements of the host CPU, not "
                "GPU performance."
                if self._device_kind is DeviceKind.CPU
                else "Eager PyTorch on CUDA. Primary timing: CUDA events."
            ),
        )

    @property
    def model_info(self) -> ModelInfo | None:
        return self._model_info

    # -- validate ----------------------------------------------------------------------

    def validate(self, config: Any, environment: EnvironmentReport) -> None:
        try:
            import torch
        except ImportError as exc:
            raise UnavailableError(
                "PyTorch is not installed, so backend 'pytorch' cannot run. Install the "
                "[torch] extra (see docs/environment.md)."
            ) from exc
        self._torch = torch
        self._settings["torch_version"] = torch.__version__
        self._settings["torch_cuda_build"] = torch.version.cuda

        if self._precision in _QUANTIZED:
            raise UnsupportedConfigurationError(
                f"Precision {self._precision.value} is not supported by eager PyTorch for "
                "this model: there is no general-purpose quantized inference path. Use the "
                "TensorRT backend (Phase 5) for reduced-precision integer/FP8 inference."
            )

        if self._device_kind is DeviceKind.CUDA:
            self._validate_cuda(environment)
        else:
            self._validate_cpu()

        self._dtype = {
            Precision.FP32: torch.float32,
            Precision.TF32: torch.float32,
            Precision.FP16: torch.float16,
            Precision.BF16: torch.bfloat16,
        }[self._precision]

        if self._weights_id != RANDOM_WEIGHTS:
            self._resolved_weights = ensure_weights(
                self._spec.weights[self._weights_id],
                allow_download=self._options.allow_download,
            )
            self._settings["weights_downloaded_this_run"] = self._resolved_weights.downloaded

    def _validate_cuda(self, environment: EnvironmentReport) -> None:
        torch = self._torch
        detected = f"NVIDIA detection reports '{environment.detection_status.value}'" + (
            f" ({environment.detection_error})" if environment.detection_error else ""
        )
        no_fallback = "Not falling back to CPU."

        if torch.version.cuda is None:
            raise UnavailableError(
                f"Requested {self._device_str}, but this PyTorch build ({torch.__version__}) "
                f"has no CUDA support compiled in. {detected}. {no_fallback}"
            )
        if not torch.cuda.is_available():
            raise UnavailableError(
                f"Requested {self._device_str}, but PyTorch reports no usable CUDA device. "
                f"{detected}. {no_fallback}"
            )
        index = int(self._device_str.split(":")[1])
        count = torch.cuda.device_count()
        if index >= count:
            raise UnavailableError(
                f"Requested {self._device_str}, but only {count} CUDA device(s) are visible. "
                f"{no_fallback}"
            )

        # Capability comes from torch for the exact device torch will use. NVML and
        # CUDA can enumerate GPUs in different orders, so the NVML report is used
        # for the error message above, not for this check.
        major, minor = torch.cuda.get_device_capability(index)
        support = {p.precision: p for p in precision_support(major, minor)}[self._precision]
        if not support.supported:
            raise UnsupportedConfigurationError(
                f"{self._precision.value} on {torch.cuda.get_device_name(index)} "
                f"(SM {major}.{minor}): {support.note}"
            )
        self._settings["cuda_device_name"] = torch.cuda.get_device_name(index)
        self._settings["cuda_capability"] = f"{major}.{minor}"
        self._settings["tensor_core_for_precision"] = support.tensor_core

    def _validate_cpu(self) -> None:
        torch = self._torch
        if self._precision is Precision.TF32:
            raise UnsupportedConfigurationError(
                "TF32 is a CUDA tensor-core execution mode; it does not exist on CPU."
            )
        if self._precision in (Precision.FP16, Precision.BF16):
            # Detect rather than assume (CLAUDE.md §3): run one tiny convolution in
            # the requested dtype. If this build cannot, the configuration is
            # unsupported here, with PyTorch's own reason.
            dtype = torch.float16 if self._precision is Precision.FP16 else torch.bfloat16
            try:
                x = torch.zeros(1, 3, 8, 8, dtype=dtype)
                w = torch.zeros(4, 3, 3, 3, dtype=dtype)
                torch.nn.functional.conv2d(x, w)
            except (RuntimeError, NotImplementedError) as exc:
                raise UnsupportedConfigurationError(
                    f"{self._precision.value} convolution is not supported by this PyTorch "
                    f"CPU build: {exc}"
                ) from exc
        self._settings["cpu_capability"] = torch.backends.cpu.get_cpu_capability()

    # -- global state ------------------------------------------------------------------

    def _precision_node(self, name: str) -> Any:
        backends = self._torch.backends
        return {
            "global": backends,
            "cudnn": backends.cudnn,
            "cudnn.conv": backends.cudnn.conv,
            "cudnn.rnn": backends.cudnn.rnn,
            "cuda.matmul": backends.cuda.matmul,
        }[name]

    def _has_fp32_precision_api(self) -> bool:
        backends = self._torch.backends
        return hasattr(backends.cudnn, "conv") and hasattr(backends.cudnn.conv, "fp32_precision")

    def _snapshot_global_state(self) -> dict[str, Any]:
        torch = self._torch
        state: dict[str, Any] = {
            "cudnn.benchmark": torch.backends.cudnn.benchmark,
            "num_threads": torch.get_num_threads(),
        }
        if self._has_fp32_precision_api():
            for node in _FP32_PRECISION_NODES:
                state[f"fp32:{node}"] = self._precision_node(node).fp32_precision
        else:
            state["legacy:cudnn.allow_tf32"] = torch.backends.cudnn.allow_tf32
            state["legacy:matmul.allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        return state

    def _restore_global_state(self) -> None:
        if self._saved_state is None or self._torch is None:
            return
        torch = self._torch
        state = self._saved_state
        torch.backends.cudnn.benchmark = state["cudnn.benchmark"]
        torch.set_num_threads(state["num_threads"])
        if "legacy:cudnn.allow_tf32" in state:
            torch.backends.cudnn.allow_tf32 = state["legacy:cudnn.allow_tf32"]
            torch.backends.cuda.matmul.allow_tf32 = state["legacy:matmul.allow_tf32"]
        else:
            # Children before parents, the reverse of the snapshot order.
            for node in reversed(_FP32_PRECISION_NODES):
                self._precision_node(node).fp32_precision = state[f"fp32:{node}"]
        self._saved_state = None

    def _apply_global_state(self) -> None:
        """Apply precision and performance flags, then record their EFFECTIVE values."""
        torch = self._torch
        policy = "tf32" if self._precision is Precision.TF32 else "ieee"

        torch.backends.cudnn.benchmark = self._options.cudnn_benchmark
        if self._options.num_threads is not None:
            torch.set_num_threads(self._options.num_threads)

        if self._has_fp32_precision_api():
            for node in ("cudnn.conv", "cudnn.rnn", "cuda.matmul"):
                self._precision_node(node).fp32_precision = policy
            self._settings["fp32_precision_api"] = "fp32_precision"
            for node in _FP32_PRECISION_NODES:
                self._settings[f"fp32_precision.{node}"] = self._precision_node(node).fp32_precision
        else:
            allow = policy == "tf32"
            torch.backends.cudnn.allow_tf32 = allow
            torch.backends.cuda.matmul.allow_tf32 = allow
            self._settings["fp32_precision_api"] = "legacy_allow_tf32"
            self._settings["cudnn.allow_tf32"] = torch.backends.cudnn.allow_tf32
            self._settings["cuda.matmul.allow_tf32"] = torch.backends.cuda.matmul.allow_tf32

        self._settings["cudnn.benchmark"] = torch.backends.cudnn.benchmark
        self._settings["cudnn.deterministic"] = torch.backends.cudnn.deterministic
        self._settings["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        )
        self._settings["num_threads"] = torch.get_num_threads()
        self._settings["num_interop_threads"] = torch.get_num_interop_threads()
        # TF32 flags only have meaning on CUDA; recorded everywhere, but say so.
        self._settings["tf32_flags_applicable"] = self._device_kind is DeviceKind.CUDA

    # -- lifecycle ---------------------------------------------------------------------

    def load(self) -> None:
        torch = self._torch
        if torch is None:
            raise BackendError("load() called before validate().")

        self._saved_state = self._snapshot_global_state()
        self._apply_global_state()
        self._device = torch.device(self._device_str)

        # Construct under a forked, seeded RNG: random-init weights are then
        # reproducible, and the global RNG is left untouched.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._seed)
            model = self._spec.build()

        architecture = type(model).__name__
        if architecture != self._spec.architecture:
            raise BackendError(
                f"Registry model {self._spec.name!r} built a {architecture}, expected "
                f"{self._spec.architecture}."
            )

        weights_url: str | None = None
        weights_sha: str | None = None
        if self._resolved_weights is not None:
            state_dict = torch.load(
                self._resolved_weights.path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state_dict, strict=True)
            weights_url = self._spec.weights[self._weights_id].url
            weights_sha = self._resolved_weights.sha256

        parameter_count = sum(int(p.numel()) for p in model.parameters())
        if parameter_count != self._spec.expected_parameters:
            raise BackendError(
                f"Model {self._spec.name!r} has {parameter_count:,} parameters; expected "
                f"{self._spec.expected_parameters:,}. Refusing to benchmark a different "
                "architecture under this name."
            )

        model = model.eval().to(device=self._device, dtype=self._dtype)
        if self._options.channels_last:
            model = model.to(memory_format=torch.channels_last)
        self._model = model

        self._settings["dtype"] = str(self._dtype).replace("torch.", "")
        self._settings["memory_format"] = (
            "channels_last" if self._options.channels_last else "contiguous"
        )
        self._settings["grad_mode"] = "inference_mode"
        self._model_info = ModelInfo(
            name=self._spec.name,
            architecture=architecture,
            source=self._spec.source,
            weights=self._weights_id,
            weights_url=weights_url,
            weights_sha256=weights_sha,
            random_init_seed=self._seed if self._weights_id == RANDOM_WEIGHTS else None,
            parameter_count=parameter_count,
            expected_parameter_count=self._spec.expected_parameters,
        )

    def prepare(self, config: Any) -> Any:
        torch = self._torch
        if self._model is None:
            raise BackendError("prepare() called before load().")
        if not isinstance(config, ExperimentConfig):
            raise BackendError(
                f"prepare() expects an ExperimentConfig, got {type(config).__name__}."
            )

        shape = (config.batch_size, *self._input_shape)
        generator = torch.Generator(device="cpu").manual_seed(self._seed)
        inputs = torch.randn(shape, generator=generator).to(device=self._device, dtype=self._dtype)
        if self._options.channels_last and inputs.dim() == 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)

        # Sanity forward pass, outside all timing: prove the model really ran, in
        # the requested precision, with the expected output, before measuring it.
        with torch.inference_mode():
            output = self._model(inputs)
        self.synchronize()

        expected = (config.batch_size, *self._spec.output_shape)
        if tuple(output.shape) != expected:
            raise BackendError(
                f"Sanity forward pass returned shape {tuple(output.shape)}, expected {expected}."
            )
        if output.dtype != self._dtype:
            raise BackendError(
                f"Sanity forward pass returned {output.dtype}, but {self._dtype} was requested: "
                "the requested precision did not take effect."
            )
        if not bool(torch.isfinite(output).all()):
            raise BackendError("Sanity forward pass produced non-finite outputs (NaN/inf).")
        self._settings["sanity_check"] = "passed"
        return inputs

    def execution_context(self) -> AbstractContextManager[object]:
        if self._torch is None:
            return contextlib.nullcontext()
        return self._torch.inference_mode()  # type: ignore[no-any-return]

    def execute(self, inputs: Any) -> Any:
        return self._model(inputs)

    def synchronize(self) -> None:
        if self._device_kind is DeviceKind.CUDA and self._device is not None:
            self._torch.cuda.synchronize(self._device)

    def make_timer(self) -> Timer:
        if self._device_kind is DeviceKind.CUDA:
            return CudaEventTimer(self._torch, self._device)
        # PyTorch CPU ops have completed when the call returns: no sync hook needed,
        # and the mechanism honestly records that none was used.
        return WallClockTimer()

    def close(self) -> None:
        self._model = None
        try:
            self._restore_global_state()
        finally:
            torch = self._torch
            if (
                torch is not None
                and self._device_kind is DeviceKind.CUDA
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
