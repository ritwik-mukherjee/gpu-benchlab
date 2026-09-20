"""Phase 6 Stage 4: TensorRT numerical correctness against PyTorch and ONNX Runtime.

The criterion is the Phase 4 one, UNCHANGED: elementwise
|cand - ref| <= 1e-4 * max|ref| + 1e-4 * |ref|, plus identical top-1, via
gpu_benchlab.core.correctness.compare_outputs. Changing the tolerance to make TensorRT
pass is forbidden; a failure is a finding and stops the phase.

Inputs are the canonical ones (core.inputs.synthetic_input), batches {1,4,8} x seeds
{0,1,2}, and every runtime consumes the same weights via the same ONNX artifact.

    PyTorch CPU FP32  (reference)
         |
    +----+--------------------+--------------------+
    |                         |                    |
  PyTorch CUDA IEEE      ORT CUDA FP32       TensorRT IEEE FP32
    +----+--------------------+--------------------+
                          compare

Negative controls, because a check that cannot fail proves nothing:
  * PyTorch CUDA FP16 vs the CPU reference must FAIL the tolerance.
  * A TensorRT engine built with BuilderFlag.TF32 left ENABLED must differ from the
    IEEE FP32 engine. If it does not, TF32 had no observable effect on this model and
    the IEEE claim rests on the flag alone -- which is reported, not hidden.

Execution here is deliberately independent of the TensorRT backend: it drives the build
layer and its own CUDA plumbing, so a bug in the backend cannot validate itself.

Usage: python analysis/gpu_validation/correctness_tensorrt.py <out_dir>
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
from _common import FAIL, INFO, PASS, Checks, require_cuda_torch

BATCHES, SEEDS = (1, 4, 8), (0, 1, 2)
CUDA_EP = "CUDAExecutionProvider"


class TrtRunner:
    """Engine + context + device buffers for one batch size, with host-side I/O."""

    def __init__(self, trt: Any, cudart: Any, plan: bytes, spec: Any) -> None:
        from gpu_benchlab.export.tensorrt_build import cuda_call

        self._trt, self._cudart, self._call = trt, cudart, cuda_call
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self._runtime.deserialize_cuda_engine(plan)
        if self.engine is None:
            raise RuntimeError("could not deserialize the engine")
        self.context = self.engine.create_execution_context()
        self._spec = spec
        self._stream = cuda_call(cudart, cudart.cudaStreamCreate())

    def infer(self, host_input: np.ndarray, input_name: str, output_name: str) -> np.ndarray:
        cudart, call = self._cudart, self._call
        batch = host_input.shape[0]
        host_output = np.full((batch, *self._spec), np.nan, dtype=np.float32)
        d_in = call(cudart, cudart.cudaMalloc(host_input.nbytes))
        d_out = call(cudart, cudart.cudaMalloc(host_output.nbytes))
        try:
            call(
                cudart,
                cudart.cudaMemcpy(
                    d_in,
                    host_input.ctypes.data,
                    host_input.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                ),
            )
            self.context.set_input_shape(input_name, host_input.shape)
            self.context.set_tensor_address(input_name, int(d_in))
            self.context.set_tensor_address(output_name, int(d_out))
            if not self.context.execute_async_v3(self._stream):
                raise RuntimeError("TensorRT refused to enqueue the inference")
            call(cudart, cudart.cudaStreamSynchronize(self._stream))
            call(
                cudart,
                cudart.cudaMemcpy(
                    host_output.ctypes.data,
                    d_out,
                    host_output.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                ),
            )
        finally:
            cudart.cudaFree(d_in)
            cudart.cudaFree(d_out)
        return host_output

    def close(self) -> None:
        self._cudart.cudaStreamDestroy(self._stream)


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase6/correctness")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch = require_cuda_torch()
    import onnxruntime as ort

    from gpu_benchlab.backends.ort_backend import OrtOptions, create_session
    from gpu_benchlab.core.correctness import DEFAULT_ATOL_SCALE, DEFAULT_RTOL, compare_outputs
    from gpu_benchlab.core.inputs import synthetic_input
    from gpu_benchlab.export.onnx_export import ExportConfig, ensure_artifact
    from gpu_benchlab.export.tensorrt_build import TrtBuildConfig, ensure_engine, import_tensorrt
    from gpu_benchlab.models.torch_loader import load_reference_model

    trt, cudart = import_tensorrt()
    checks = Checks("correctness_tensorrt")
    loaded = load_reference_model("resnet50")
    artifact, manifest, _ = ensure_artifact(
        ExportConfig(model="resnet50", weights=loaded.info.weights), allow_export=True
    )
    same_weights = loaded.info.weights_sha256 == manifest.model.weights_sha256
    checks.expect(
        "W1",
        same_weights,
        "the reference model and the ONNX artifact carry the same weights",
        loaded=loaded.info.weights_sha256,
        artifact=manifest.model.weights_sha256,
    )
    if not same_weights:
        checks.finish(out_dir)

    chw = tuple(int(d) for d in manifest.input.shape[1:])
    out_spec = (int(manifest.output.shape[-1]),)
    input_name, output_name = manifest.input.name, manifest.output.name

    # --- engines: IEEE FP32, and TF32 as the TensorRT-specific negative control -------
    engines: dict[str, Any] = {}
    for label, policy in (("trt_fp32", "ieee_fp32"), ("trt_tf32", "tf32")):
        path, engine_manifest, built = ensure_engine(
            TrtBuildConfig(model="resnet50", weights=loaded.info.weights, precision=policy),
            allow_build=True,
        )
        engines[label] = {
            "manifest": engine_manifest.model_dump(mode="json"),
            "runner": TrtRunner(trt, cudart, path.read_bytes(), out_spec),
            "built_now": built,
        }
    checks.expect(
        "E1",
        engines["trt_fp32"]["manifest"]["builder_settings"]["tf32_flag_after_policy"] is False
        and engines["trt_tf32"]["manifest"]["builder_settings"]["tf32_flag_after_policy"] is True,
        "the IEEE FP32 engine cleared BuilderFlag.TF32 and the control engine did not",
        fp32_engine=engines["trt_fp32"]["manifest"]["engine_sha256"],
        tf32_engine=engines["trt_tf32"]["manifest"]["engine_sha256"],
    )

    # --- the other runtimes ------------------------------------------------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    cpu_model = loaded.module
    gpu_model = copy.deepcopy(cpu_model).to("cuda:0")
    model_bytes = artifact.read_bytes()
    ort_session = create_session(
        ort,
        model_bytes,
        CUDA_EP,
        {"device_id": "0", "use_tf32": "0", "cudnn_conv_algo_search": "EXHAUSTIVE"},
        OrtOptions(),
    )

    outputs: dict[str, np.ndarray] = {}
    for batch in BATCHES:
        for seed in SEEDS:
            key = f"b{batch}_s{seed}"
            x = synthetic_input((batch, *chw), seed)
            with torch.inference_mode():
                outputs[f"cpu_{key}"] = cpu_model(torch.from_numpy(x)).numpy()
                outputs[f"pt_cuda_{key}"] = (
                    gpu_model(torch.from_numpy(x).to("cuda:0")).float().cpu().numpy()
                )
            outputs[f"ort_cuda_{key}"] = ort_session.run([output_name], {input_name: x})[0]
            for label in ("trt_fp32", "trt_tf32"):
                outputs[f"{label}_{key}"] = engines[label]["runner"].infer(
                    x, input_name, output_name
                )

    # --- negative control: FP16 on the GPU --------------------------------------------
    x = synthetic_input((1, *chw), 0)
    half = copy.deepcopy(cpu_model).to("cuda:0", dtype=torch.float16)
    with torch.inference_mode():
        outputs["pt_fp16_b1_s0"] = (
            half(torch.from_numpy(x).to("cuda:0", torch.float16)).float().cpu().numpy()
        )

    def compare(reference: str, candidate: str) -> list[dict[str, Any]]:
        rows = []
        for batch in BATCHES:
            for seed in SEEDS:
                key = f"b{batch}_s{seed}"
                result = compare_outputs(
                    outputs[f"{reference}_{key}"],
                    outputs[f"{candidate}_{key}"],
                    rtol=DEFAULT_RTOL,
                    atol_scale=DEFAULT_ATOL_SCALE,
                )
                rows.append({"case": key, **result.model_dump(mode="json")})
        return rows

    table = {
        "C1 cpu vs trt_fp32": compare("cpu", "trt_fp32"),
        "C2 pt_cuda vs trt_fp32": compare("pt_cuda", "trt_fp32"),
        "C3 ort_cuda vs trt_fp32": compare("ort_cuda", "trt_fp32"),
        "C4 cpu vs trt_tf32": compare("cpu", "trt_tf32"),
    }
    for label in ("C1 cpu vs trt_fp32", "C2 pt_cuda vs trt_fp32", "C3 ort_cuda vs trt_fp32"):
        rows = table[label]
        checks.expect(
            label.split()[0],
            all(row["passed"] for row in rows),
            f"{label}: all 9 cases within the Phase 4 tolerance",
            passed=sum(row["passed"] for row in rows),
            max_abs_error=max(row["max_abs_error"] for row in rows),
            worst_violation_ratio=max(row["worst_violation_ratio"] for row in rows),
            top1_min=min(row["top1_agreement"] for row in rows),
            top5_min=min(row["top5_overlap"] for row in rows),
        )

    control = compare_outputs(
        outputs["cpu_b1_s0"],
        outputs["pt_fp16_b1_s0"],
        rtol=DEFAULT_RTOL,
        atol_scale=DEFAULT_ATOL_SCALE,
    )
    checks.expect(
        "NC1",
        not control.passed,
        "FP16 negative control is REJECTED by the tolerance",
        violations=control.violations,
        worst_violation_ratio=control.worst_violation_ratio,
        top1=control.top1_agreement,
    )

    differing = [
        key
        for key in outputs
        if key.startswith("trt_tf32_")
        and not np.array_equal(outputs[key], outputs[key.replace("tf32", "fp32")])
    ]
    checks.add(
        "NC2",
        PASS if differing else INFO,
        "TensorRT TF32 engine differs from the IEEE FP32 engine (TF32 observable)"
        if differing
        else "TF32 engine produced identical values: TF32 had no observable effect here",
        differing_cases=len(differing),
        tf32_vs_cpu_passed=sum(row["passed"] for row in table["C4 cpu vs trt_tf32"]),
        tf32_worst_ratio=max(row["worst_violation_ratio"] for row in table["C4 cpu vs trt_tf32"]),
    )

    repeat = engines["trt_fp32"]["runner"].infer(
        synthetic_input((1, *chw), 0), input_name, output_name
    )
    checks.add(
        "R1",
        INFO,
        "same-process repeatability of the TensorRT engine (max |delta|)",
        max_abs_delta=float(np.abs(repeat - outputs["trt_fp32_b1_s0"]).max()),
    )

    np.savez(out_dir / "outputs.npz", **outputs)
    for entry in engines.values():
        entry["runner"].close()
    checks.add(
        "versions",
        INFO,
        "software",
        tensorrt=trt.__version__,
        torch=torch.__version__,
        onnxruntime=ort.__version__,
    )
    failures = [item for item in checks.items if item["status"] == FAIL]
    checks.finish(
        out_dir,
        {
            "criterion": {"rtol": DEFAULT_RTOL, "atol_scale": DEFAULT_ATOL_SCALE},
            "comparisons": table,
            "negative_control_fp16": control.model_dump(mode="json"),
            "engines": {k: v["manifest"] for k, v in engines.items()},
            "artifact_sha256": manifest.artifact.sha256,
            "weights_sha256": manifest.model.weights_sha256,
            "outputs_npz": str(out_dir / "outputs.npz"),
            "failed_checks": [item["id"] for item in failures],
        },
    )


if __name__ == "__main__":
    main()
