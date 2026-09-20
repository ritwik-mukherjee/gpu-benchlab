"""Phase 6 Stage 1: canonical ONNX -> TensorRT engine -> real inference on the GPU.

Independent of the benchmark architecture on purpose: nothing here touches the engine,
the timing abstraction or the result schema. It answers one question — does this
TensorRT install parse our artifact, build, serialize, deserialize and execute on this
GPU — and records what it took, including which shared libraries were actually loaded.

**This is not a benchmark.** It times the engine build and the deserialization because
those are setup costs worth recording, and deliberately does not time inference.

Written against TensorRT 11.3 as installed, whose API differs from 10.x tutorials:
strongly typed networks are the default, `BuilderFlag.FP16`/`INT8` no longer exist, and
`BuilderFlag.TF32` is **enabled by default** — so IEEE FP32 requires clearing it.

Usage: python analysis/gpu_validation/tensorrt_smoke.py <out_dir> [--batch N]...
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from _common import INCONCLUSIVE, INFO, PASS, Checks, compute_app_pids, loaded_libraries

PROFILE_MIN, PROFILE_OPT, PROFILE_MAX = 1, 8, 8


def require_tensorrt() -> tuple[Any, Any]:
    """Import TensorRT and the CUDA runtime, or refuse (exit 2). Never falls back."""
    try:
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
    except ImportError as exc:
        print(
            f"TensorRT stack unavailable ({exc}). Install the [tensorrt] extra on a "
            "machine with an NVIDIA GPU; nothing was measured.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    return trt, cudart


def cuda_check(cudart: Any, result: Any) -> Any:
    """Unpack a cuda-python (err, value...) tuple, raising with the CUDA message."""
    err, *rest = result if isinstance(result, tuple) else (result,)
    if int(err) != 0:
        _, message = cudart.cudaGetErrorString(err)
        raise RuntimeError(f"CUDA error {int(err)}: {message.decode()}")
    return rest[0] if len(rest) == 1 else tuple(rest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--batch", action="append", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    batches = args.batch or [1, 8]

    trt, cudart = require_tensorrt()
    from gpu_benchlab.core.inputs import INPUT_GENERATOR, synthetic_input
    from gpu_benchlab.export.onnx_export import ExportConfig, ensure_artifact
    from gpu_benchlab.models.registry import get_model

    checks = Checks("tensorrt_smoke")
    provenance: dict[str, Any] = {"input_generator": INPUT_GENERATOR, "seed": args.seed}

    # --- versions and the CUDA stack actually in use ---------------------------------
    runtime_version = cuda_check(cudart, cudart.cudaRuntimeGetVersion())
    driver_version = cuda_check(cudart, cudart.cudaDriverGetVersion())
    props = cuda_check(cudart, cudart.cudaGetDeviceProperties(0))
    provenance["versions"] = {
        "tensorrt": trt.__version__,
        "tensorrt_module": trt.__file__,
        "cuda_runtime": runtime_version,
        "cuda_driver": driver_version,
        "gpu_name": props.name.decode(),
        "compute_capability": f"{props.major}.{props.minor}",
        "multiprocessor_count": props.multiProcessorCount,
        "total_memory_bytes": props.totalGlobalMem,
    }
    checks.add(
        "T1",
        INFO,
        "TensorRT and CUDA versions",
        **provenance["versions"],
        note="CUDA runtime is newer than the driver's reported version; NVIDIA documents "
        "r580+ as the requirement for CUDA 13.x builds",
    )

    # --- step 1: the canonical artifact, same bytes the other backends use -----------
    weights = get_model("resnet50").resolve_weights(None)
    artifact, manifest, _ = ensure_artifact(
        ExportConfig(model="resnet50", weights=weights), allow_export=False
    )
    model_bytes = artifact.read_bytes()
    digest = hashlib.sha256(model_bytes).hexdigest()
    checks.expect(
        "T2",
        digest == manifest.artifact.sha256,
        "canonical ONNX artifact loaded and its SHA-256 matches the manifest",
        artifact=str(artifact),
        sha256=digest,
        weights_sha256=manifest.model.weights_sha256,
    )

    # --- step 2: parse, surfacing every parser error ---------------------------------
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    onnx_parser = trt.OnnxParser(network, logger)
    parsed = onnx_parser.parse(model_bytes)
    errors = [str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)]
    checks.expect(
        "T3", parsed and not errors, "TensorRT parsed the canonical ONNX", parser_errors=errors
    )
    if not parsed or errors:
        checks.finish(args.out_dir, provenance)

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    network_info = {
        "layers": network.num_layers,
        "strongly_typed": network.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED),
        "inputs": {t.name: (str(t.dtype), list(t.shape)) for t in inputs},
        "outputs": {t.name: (str(t.dtype), list(t.shape)) for t in outputs},
    }
    provenance["network"] = network_info
    checks.expect(
        "T4",
        len(inputs) == 1 and len(outputs) == 1 and network_info["strongly_typed"],
        "one input, one output, network is strongly typed (TensorRT 11 default)",
        **network_info,
    )
    input_name, output_name = inputs[0].name, outputs[0].name

    # --- step 3: build, IEEE FP32 (TF32 must be cleared: it is ON by default) ---------
    config = builder.create_builder_config()
    tf32_default = config.get_flag(trt.BuilderFlag.TF32)
    config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    chw = tuple(int(d) for d in manifest.input.shape[1:])
    profile.set_shape(input_name, (PROFILE_MIN, *chw), (PROFILE_OPT, *chw), (PROFILE_MAX, *chw))
    config.add_optimization_profile(profile)
    builder_settings = {
        "tf32_enabled_by_default": tf32_default,
        "tf32_after_clear": config.get_flag(trt.BuilderFlag.TF32),
        "strict_nans": config.get_flag(trt.BuilderFlag.STRICT_NANS),
        "disable_timing_cache": config.get_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE),
        "disable_compilation_cache": config.get_flag(trt.BuilderFlag.DISABLE_COMPILATION_CACHE),
        "builder_optimization_level": config.builder_optimization_level,
        "workspace_pool_limit_bytes": config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE),
        "profile": {
            "min": [PROFILE_MIN, *chw],
            "opt": [PROFILE_OPT, *chw],
            "max": [PROFILE_MAX, *chw],
        },
    }
    provenance["builder"] = builder_settings
    checks.expect(
        "T5",
        tf32_default is True and config.get_flag(trt.BuilderFlag.TF32) is False,
        "TF32 is on by default and was explicitly cleared for IEEE FP32",
        **builder_settings,
    )

    started = time.perf_counter_ns()
    plan = builder.build_serialized_network(network, config)
    build_ms = (time.perf_counter_ns() - started) / 1e6
    checks.expect("T6", plan is not None, "engine built", engine_build_ms=round(build_ms, 1))
    if plan is None:
        checks.finish(args.out_dir, provenance)

    plan_bytes = bytes(plan)
    engine_sha = hashlib.sha256(plan_bytes).hexdigest()
    plan_path = args.out_dir / "resnet50-fp32-smoke.plan"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(plan_bytes)
    provenance["engine"] = {
        "sha256": engine_sha,
        "size_bytes": len(plan_bytes),
        "build_ms": round(build_ms, 1),
        "path": str(plan_path),
    }
    checks.add("T7", INFO, "engine serialized", **provenance["engine"])

    # --- step 4: deserialize -----------------------------------------------------------
    started = time.perf_counter_ns()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_bytes)
    deser_ms = (time.perf_counter_ns() - started) / 1e6
    checks.expect(
        "T8",
        engine is not None,
        "engine deserialized",
        engine_deserialization_ms=round(deser_ms, 1),
    )
    if engine is None:
        checks.finish(args.out_dir, provenance)
    provenance["engine"]["deserialization_ms"] = round(deser_ms, 1)

    recorded = engine.get_tensor_profile_shape(input_name, 0)
    checks.expect(
        "T9",
        [list(s) for s in recorded]
        == [[PROFILE_MIN, *chw], [PROFILE_OPT, *chw], [PROFILE_MAX, *chw]],
        "the engine reports the optimization profile that was requested",
        engine_profile=[list(s) for s in recorded],
    )

    # --- step 5: real inference on the GPU ---------------------------------------------
    context = engine.create_execution_context()
    stream = cuda_check(cudart, cudart.cudaStreamCreate())
    out_classes = int(manifest.output.shape[-1])
    for batch in batches:
        host_in = synthetic_input((batch, *chw), args.seed)
        host_out = np.empty((batch, out_classes), dtype=np.float32)
        d_in = cuda_check(cudart, cudart.cudaMalloc(host_in.nbytes))
        d_out = cuda_check(cudart, cudart.cudaMalloc(host_out.nbytes))
        try:
            cuda_check(
                cudart,
                cudart.cudaMemcpy(
                    d_in,
                    host_in.ctypes.data,
                    host_in.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                ),
            )
            context.set_input_shape(input_name, (batch, *chw))
            context.set_tensor_address(input_name, int(d_in))
            context.set_tensor_address(output_name, int(d_out))
            residency = {}
            for label, pointer in (("input", d_in), ("output", d_out)):
                attributes = cuda_check(cudart, cudart.cudaPointerGetAttributes(pointer))
                residency[label] = {
                    "type": str(attributes.type),
                    "device": attributes.device,
                    "is_device_memory": attributes.type
                    == cudart.cudaMemoryType.cudaMemoryTypeDevice,
                }
            ok = context.execute_async_v3(stream)
            cuda_check(cudart, cudart.cudaStreamSynchronize(stream))
            cuda_check(
                cudart,
                cudart.cudaMemcpy(
                    host_out.ctypes.data,
                    d_out,
                    host_out.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                ),
            )
            checks.expect(
                f"T10.b{batch}",
                bool(ok)
                and host_out.shape == (batch, out_classes)
                and host_out.dtype == np.float32
                and bool(np.isfinite(host_out).all()),
                f"inference executed and produced a finite ({batch}, {out_classes}) float32 output",
                enqueued=bool(ok),
                shape=list(host_out.shape),
                dtype=str(host_out.dtype),
                logit_range=[float(host_out.min()), float(host_out.max())],
            )
            checks.expect(
                f"T11.b{batch}",
                all(r["is_device_memory"] for r in residency.values()),
                f"input and output buffers are GPU-resident (batch {batch})",
                **residency,
            )
        finally:
            cudart.cudaFree(d_in)
            cudart.cudaFree(d_out)

    pids = compute_app_pids()
    own = os.getpid()
    if pids is None or own not in pids:
        checks.add("T12", INCONCLUSIVE, "PID not visible to nvidia-smi (hidden in containers)")
    else:
        checks.add("T12", PASS, "this process is listed by nvidia-smi as using the GPU")

    libraries = loaded_libraries("libnvinfer", "libcudart", "libcublas", "libcudnn", "libnvrtc")
    if libraries is None:
        checks.add("T13", INCONCLUSIVE, "no /proc/self/maps; loaded libraries unchecked")
    else:
        checks.expect(
            "T13",
            bool(libraries.get("libnvinfer")),
            "TensorRT's own library is loaded; every CUDA library path is recorded",
            libraries=libraries,
        )
    provenance["libraries"] = libraries

    cudart.cudaStreamDestroy(stream)
    checks.finish(args.out_dir, provenance)


if __name__ == "__main__":
    main()
