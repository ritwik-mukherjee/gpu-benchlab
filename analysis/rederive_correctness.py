"""Independently re-derive a stored PyTorch-vs-ONNX Runtime correctness verdict.

Deliberately imports nothing from gpu_benchlab: it reads only `report.json` and
`outputs.npz`, re-implements the pre-registered criterion from its definition
(docs/plans/phase-4-onnxruntime.md §3) and checks it against what was stored. A
bug in gpu_benchlab.core.correctness therefore cannot validate itself here.

Usage:
    python analysis/rederive_correctness.py <correctness-report-directory>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TOL = 1e-12


def rederive(ref: np.ndarray, cand: np.ndarray, rtol: float, atol_scale: float) -> dict:
    r = ref.astype(np.float64)
    c = cand.astype(np.float64)
    diff = np.abs(c - r)
    bound = atol_scale * np.abs(r).max() + rtol * np.abs(r)
    top1 = float((ref.argmax(axis=1) == cand.argmax(axis=1)).mean())
    violations = int((diff > bound).sum())
    return {
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "worst_violation_ratio": float((diff / bound).max()),
        "violations": violations,
        "top1_agreement": top1,
        "passed": violations == 0
        and top1 == 1.0
        and ref.shape == cand.shape
        and ref.dtype == cand.dtype
        and bool(np.isfinite(ref).all() and np.isfinite(cand).all()),
    }


def main() -> int:
    directory = Path(sys.argv[1])
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    outputs = np.load(directory / "outputs.npz")
    rtol, atol_scale = report["rtol"], report["atol_scale"]
    print(f"report {report['report_id']}  rtol={rtol}  atol_scale={atol_scale}")
    print(f"artifact {report['artifact_sha256'][:16]}  weights {report['weights_sha256'][:16]}")

    mismatches = 0
    for case in report["cases"]:
        key = f"b{case['batch_size']}_s{case['seed']}"
        mine = rederive(outputs[f"{key}_reference"], outputs[f"{key}_candidate"], rtol, atol_scale)
        stored = case["comparison"]
        agree = (
            all(
                abs(mine[k] - stored[k]) <= TOL * max(1.0, abs(stored[k]))
                for k in (
                    "max_abs_error",
                    "mean_abs_error",
                    "worst_violation_ratio",
                    "top1_agreement",
                )
            )
            and mine["violations"] == stored["violations"]
            and mine["passed"] == stored["passed"]
        )
        mismatches += not agree
        print(
            f"  {key:<7} max_abs={mine['max_abs_error']:.4e}"
            f" worst/tol={mine['worst_violation_ratio']:.4f} "
            f"top1={mine['top1_agreement']:.2f} passed={mine['passed']}  "
            f"{'matches stored' if agree else 'DISAGREES WITH STORED'}"
        )

    ctl = report["negative_control"]
    key = f"b{ctl['batch_size']}_s{ctl['seed']}"
    ctl_mine = rederive(
        outputs[f"{key}_reference"], outputs["negative_control_fp16"], rtol, atol_scale
    )
    ctl_agree = (
        ctl_mine["passed"] == ctl["comparison"]["passed"]
        and ctl_mine["violations"] == ctl["comparison"]["violations"]
    )
    mismatches += not ctl_agree
    print(
        f"  control fp16: passed={ctl_mine['passed']} violations={ctl_mine['violations']} "
        f"worst/tol={ctl_mine['worst_violation_ratio']:.2f}  "
        f"{'matches stored' if ctl_agree else 'DISAGREES WITH STORED'}"
    )

    verdict = (
        all(
            rederive(
                outputs[f"b{c['batch_size']}_s{c['seed']}_reference"],
                outputs[f"b{c['batch_size']}_s{c['seed']}_candidate"],
                rtol,
                atol_scale,
            )["passed"]
            for c in report["cases"]
        )
        and not ctl_mine["passed"]
    )
    print(
        f"re-derived verdict: {'PASSED' if verdict else 'FAILED'}   stored verdict: "
        f"{'PASSED' if report['passed'] else 'FAILED'}   disagreements: {mismatches}"
    )
    return 0 if mismatches == 0 and verdict == report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
