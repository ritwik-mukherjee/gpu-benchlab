"""Phase 5A runbook §7 + §9: numerical correctness on CUDA.

The criterion is the Phase 4 one, UNCHANGED (ADR 0007): elementwise
|cand - ref| <= 1e-4 * max|ref| + 1e-4 * |ref|, plus identical top-1, via
gpu_benchlab.core.correctness.compare_outputs with its default rtol / atol_scale.
Changing the tolerance to make a comparison pass is forbidden; a failure is a finding.

Inputs: core.inputs.synthetic_input (numpy PCG64, float32), batches 1/4/8 x seeds 0/1/2.
Weights: the pinned IMAGENET1K_V2; the ONNX artifact's recorded weights SHA-256 must
equal the loaded model's, or nothing is compared.

  cpu      PyTorch eager FP32 on CPU (the Phase 4 reference)
  pt_ieee  PyTorch CUDA, fp32_precision "ieee" (cudnn.conv + cuda.matmul), cudnn.benchmark on
  pt_tf32  PyTorch CUDA, fp32_precision "tf32"        -- only if SM >= 8.0
  ort_fp32 ONNX Runtime CUDA EP, use_tf32 = 0, gpu_benchlab session options
  ort_tf32 ONNX Runtime CUDA EP, use_tf32 = 1         -- only if SM >= 8.0
  pt_fp16  PyTorch CUDA FP16 (negative control, batch 1 seed 0)

Pre-registered checks:
  W1  weights identical (artifact manifest SHA-256 == loaded weights SHA-256)
  C1  cpu vs pt_ieee      all cases pass                                  (§7)
  C3  pt_ieee vs ort_fp32 all cases pass                                  (§9)
  C4  cpu vs ort_fp32     all cases pass (what `gpu-bench onnx verify --device cuda:0` checks)
  NC  cpu vs pt_fp16      must FAIL: the tolerance can detect reduced precision on this GPU
  T1  pt_tf32 differs bitwise from pt_ieee in at least one case: TF32 actually engaged.
      If identical, "TF32 had no observable effect" is the finding (no TF32 claim).
  C2 / C5  cpu vs pt_tf32 / ort_tf32: OBSERVED ONLY. Pass or fail is recorded, not required.
  R1  repeatability: max |delta| between two pt_ieee / ort_fp32 runs of the same input (INFO)

Usage: python analysis/gpu_validation/correctness_gpu.py <out_dir>
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
from _common import INCONCLUSIVE, INFO, PASS, Checks, require_cuda_torch

BATCHES, SEEDS = (1, 4, 8), (0, 1, 2)
CUDA_EP = "CUDAExecutionProvider"


def set_precision(torch: Any, policy: str) -> dict[str, str]:
    torch.backends.cudnn.conv.fp32_precision = policy
    torch.backends.cuda.matmul.fp32_precision = policy
    return {
        "cudnn.conv": torch.backends.cudnn.conv.fp32_precision,
        "cuda.matmul": torch.backends.cuda.matmul.fp32_precision,
    }


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase5a/correctness")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch = require_cuda_torch()
    import onnxruntime as ort

    from gpu_benchlab.backends.ort_backend import OrtOptions, create_session, measure_placement
    from gpu_benchlab.core.correctness import DEFAULT_ATOL_SCALE, DEFAULT_RTOL, compare_outputs
    from gpu_benchlab.core.inputs import synthetic_input
    from gpu_benchlab.export.onnx_export import ExportConfig, ensure_artifact
    from gpu_benchlab.models.torch_loader import load_reference_model

    checks = Checks("correctness_gpu")
    loaded = load_reference_model("resnet50")
    artifact, manifest, _ = ensure_artifact(
        ExportConfig(model="resnet50", weights=loaded.info.weights), allow_export=True
    )
    same = loaded.info.weights_sha256 == manifest.model.weights_sha256
    checks.expect(
        "W1",
        same,
        "reference weights are the artifact's weights",
        loaded=loaded.info.weights_sha256,
        artifact=manifest.model.weights_sha256,
    )
    if not same:
        checks.finish(out_dir)

    major, minor = torch.cuda.get_device_capability(0)
    tf32_ok = (major, minor) >= (8, 0)
    torch.backends.cudnn.benchmark = True  # as in the benchmark (PyTorchOptions default)
    cpu_model = loaded.module
    gpu_model = copy.deepcopy(cpu_model).to("cuda:0")
    model_bytes = artifact.read_bytes()

    def ort_session(use_tf32: str) -> tuple[Any, dict[str, Any]]:
        opts = {"device_id": "0", "use_tf32": use_tf32, "cudnn_conv_algo_search": "EXHAUSTIVE"}
        session = create_session(ort, model_bytes, CUDA_EP, opts, OrtOptions())
        feed = {manifest.input.name: synthetic_input((1, 3, 224, 224), 0)}
        return session, {
            "active_providers": session.get_providers(),
            "effective_options": session.get_provider_options().get(CUDA_EP, {}),
            "node_placement": measure_placement(
                ort, model_bytes, CUDA_EP, opts, OrtOptions(), feed
            ),
        }

    sessions = {"ort_fp32": ort_session("0")}
    if tf32_ok:
        sessions["ort_tf32"] = ort_session("1")

    def run_pt(x: np.ndarray, policy: str) -> np.ndarray:
        set_precision(torch, policy)
        with torch.inference_mode():
            y = gpu_model(torch.from_numpy(x).to("cuda:0"))
        return y.float().cpu().numpy()

    outputs: dict[str, np.ndarray] = {}
    for b in BATCHES:
        for s in SEEDS:
            x = synthetic_input((b, 3, 224, 224), s)
            key = f"b{b}_s{s}"
            with torch.inference_mode():
                outputs[f"cpu_{key}"] = cpu_model(torch.from_numpy(x)).numpy()
            outputs[f"pt_ieee_{key}"] = run_pt(x, "ieee")
            outputs[f"pt_ieee_repeat_{key}"] = run_pt(x, "ieee")
            if tf32_ok:
                outputs[f"pt_tf32_{key}"] = run_pt(x, "tf32")
            for name, (session, _) in sessions.items():
                outputs[f"{name}_{key}"] = session.run(
                    [manifest.output.name], {manifest.input.name: x}
                )[0]
            outputs[f"ort_fp32_repeat_{key}"] = sessions["ort_fp32"][0].run(
                [manifest.output.name], {manifest.input.name: x}
            )[0]
    precision_ieee = set_precision(torch, "ieee")

    x = synthetic_input((1, 3, 224, 224), 0)
    half = copy.deepcopy(cpu_model).to("cuda:0", dtype=torch.float16)
    with torch.inference_mode():
        outputs["pt_fp16_b1_s0"] = (
            half(torch.from_numpy(x).to("cuda:0", torch.float16)).float().cpu().numpy()
        )

    def compare(ref: str, cand: str) -> list[dict[str, Any]]:
        rows = []
        for b in BATCHES:
            for s in SEEDS:
                k = f"b{b}_s{s}"
                if f"{cand}_{k}" not in outputs:
                    continue
                c = compare_outputs(
                    outputs[f"{ref}_{k}"],
                    outputs[f"{cand}_{k}"],
                    rtol=DEFAULT_RTOL,
                    atol_scale=DEFAULT_ATOL_SCALE,
                )
                rows.append({"case": k, **c.model_dump(mode="json")})
        return rows

    table = {
        "C1 cpu vs pt_ieee": compare("cpu", "pt_ieee"),
        "C3 pt_ieee vs ort_fp32": compare("pt_ieee", "ort_fp32"),
        "C4 cpu vs ort_fp32": compare("cpu", "ort_fp32"),
        "C2 cpu vs pt_tf32": compare("cpu", "pt_tf32"),
        "C5 cpu vs ort_tf32": compare("cpu", "ort_tf32"),
    }
    for label in ("C1 cpu vs pt_ieee", "C3 pt_ieee vs ort_fp32", "C4 cpu vs ort_fp32"):
        rows = table[label]
        checks.expect(
            label.split()[0],
            len(rows) == 9 and all(r["passed"] for r in rows),
            f"{label}: all 9 cases within the Phase 4 tolerance",
            passed=sum(r["passed"] for r in rows),
            max_abs_error=max(r["max_abs_error"] for r in rows),
            worst_violation_ratio=max(r["worst_violation_ratio"] for r in rows),
            top1_min=min(r["top1_agreement"] for r in rows),
        )
    control = compare_outputs(
        outputs["cpu_b1_s0"],
        outputs["pt_fp16_b1_s0"],
        rtol=DEFAULT_RTOL,
        atol_scale=DEFAULT_ATOL_SCALE,
    )
    checks.expect(
        "NC",
        not control.passed,
        "FP16 negative control is REJECTED",
        violations=control.violations,
        worst_violation_ratio=control.worst_violation_ratio,
        top1=control.top1_agreement,
    )

    if tf32_ok:
        differs = [
            k
            for k in outputs
            if k.startswith("pt_tf32_")
            and not np.array_equal(outputs[k], outputs[k.replace("tf32", "ieee")])
        ]
        checks.add(
            "T1",
            PASS if differs else INCONCLUSIVE,
            "TF32 output differs bitwise from IEEE (TF32 engaged)"
            if differs
            else "TF32 had NO observable effect on outputs",
            differing_cases=differs,
        )
        for label in ("C2 cpu vs pt_tf32", "C5 cpu vs ort_tf32"):
            rows = table[label]
            checks.add(
                label.split()[0],
                INFO,
                f"{label}: observed, not required",
                passed=sum(r["passed"] for r in rows),
                of=len(rows),
                worst_violation_ratio=max(r["worst_violation_ratio"] for r in rows),
            )
    else:
        checks.add("T1", INFO, f"TF32 not applicable: SM {major}.{minor} < 8.0")

    repeat = {
        name: max(
            float(np.abs(outputs[f"{name}_{k}"] - outputs[f"{name}_repeat_{k}"]).max())
            for k in (f"b{b}_s{s}" for b in BATCHES for s in SEEDS)
        )
        for name in ("pt_ieee", "ort_fp32")
    }
    checks.add("R1", INFO, "same-process repeatability (max |delta|)", **repeat)

    np.savez(out_dir / "outputs.npz", **outputs)
    checks.finish(
        out_dir,
        {
            "criterion": {"rtol": DEFAULT_RTOL, "atol_scale": DEFAULT_ATOL_SCALE},
            "comparisons": table,
            "negative_control": control.model_dump(mode="json"),
            "pytorch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cudnn_benchmark": True,
                "fp32_precision_ieee_readback": precision_ieee,
                "device": torch.cuda.get_device_name(0),
                "capability": f"{major}.{minor}",
            },
            "onnxruntime": {
                "version": ort.__version__,
                **{name: info for name, (_, info) in sessions.items()},
            },
            "artifact_sha256": manifest.artifact.sha256,
            "weights_sha256": manifest.model.weights_sha256,
            "outputs_npz": str(out_dir / "outputs.npz"),
        },
    )


if __name__ == "__main__":
    main()
