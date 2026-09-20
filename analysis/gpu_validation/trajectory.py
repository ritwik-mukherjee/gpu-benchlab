"""Phase 5A runbook §6 + §11: per-iteration trajectory, warmup sufficiency, drift, spread.

Standard library only; reads the stored result.json and raw.json of each run, so
nothing here depends on gpu_benchlab's own statistics code.

Pre-registered definitions (fixed before any GPU data exists):
  trajectory   warmup samples followed by measured samples (primary timing series)
  m*           median of the last 500 measured samples (all of them if fewer)
  blocks       consecutive 10-iteration blocks of the trajectory, summarised by median
  settle k     first iteration index from which EVERY later block median is within
               +/-2% of m*; None = never settled inside the run
  sufficient   k <= the run's configured warmup_iterations
  drift        (median of 2nd half - median of 1st half of measured) / m*;
               |drift| > 2% => UNSTABLE
  containment  CUDA runs: synchronized host time (secondary) >= event time (primary)
               for every iteration
  spread       across repeats of one configuration: (max - min) / median of the
               per-run medians; > 5% => UNSTABLE, investigate before interpreting

Usage: python analysis/gpu_validation/trajectory.py [--json out.json] <result_dir>...
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

BLOCK, BAND, TAIL, DRIFT_LIMIT, SPREAD_LIMIT = 10, 0.02, 500, 0.02, 0.05


def settle_index(trajectory: list[float], reference: float) -> int | None:
    medians = [st.median(trajectory[i : i + BLOCK]) for i in range(0, len(trajectory), BLOCK)]
    inside = [abs(m - reference) <= BAND * reference for m in medians]
    for b in range(len(inside)):
        if all(inside[b:]):
            return b * BLOCK
    return None


def analyse(directory: Path) -> dict[str, Any]:
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    raw = json.loads((directory / "raw.json").read_text(encoding="utf-8"))
    measured, warmup = raw["latency_ms"], raw["warmup_latency_ms"]
    if not measured:
        return {"run": directory.name, "status": result["status"], "note": "no samples"}
    cfg = result["configuration"]
    reference = st.median(measured[-TAIL:])
    half = len(measured) // 2
    drift = (st.median(measured[half:]) - st.median(measured[:half])) / reference
    k = settle_index(warmup + measured, reference)
    out: dict[str, Any] = {
        "run": directory.name,
        "status": result["status"],
        "backend": result["backend"]["name"],
        "device": result["backend"]["device"],
        "batch": cfg["batch_size"],
        "precision": cfg["precision"],
        "timing": result["timing_mechanism"],
        "warmup_configured": cfg["benchmark"]["warmup_iterations"],
        "median_ms": st.median(measured),
        "reference_m_star_ms": reference,
        "settle_iteration": k,
        "warmup_sufficient": k is not None and k <= cfg["benchmark"]["warmup_iterations"],
        "first_timed_over_m_star": (warmup or measured)[0] / reference,
        "drift": drift,
        "unstable_drift": abs(drift) > DRIFT_LIMIT,
        "phases_ms": result["phases"],
        "notes": result["notes"],
        "block_medians_over_m_star": [
            round(st.median((warmup + measured)[i : i + BLOCK]) / reference, 4)
            for i in range(0, min(len(warmup + measured), 30 * BLOCK), BLOCK)
        ],
    }
    secondary = raw.get("secondary_latency_ms") or []
    if secondary:
        pairs = list(zip(secondary, measured, strict=True))
        out["secondary_timing"] = result.get("secondary_timing_mechanism")
        out["containment_violations"] = sum(h < e for h, e in pairs)
        out["median_host_minus_event_ms"] = st.median(h - e for h, e in pairs)
        out["secondary_median_ms"] = st.median(secondary)
    return out


def main() -> None:
    args = sys.argv[1:]
    json_path = None
    if args[:1] == ["--json"]:
        json_path, args = Path(args[1]), args[2:]
    runs = [analyse(Path(a)) for a in args]
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        if "median_ms" in r:
            key = f"{r['backend']} {r['device']} b{r['batch']} {r['precision']}"
            groups.setdefault(key, []).append(r)
    summary = {}
    for key, members in groups.items():
        medians = [m["median_ms"] for m in members]
        spread = (max(medians) - min(medians)) / st.median(medians)
        summary[key] = {
            "repeats": len(members),
            "run_medians_ms": medians,
            "spread": spread,
            "unstable_spread": len(members) > 1 and spread > SPREAD_LIMIT,
            "settle_iterations": [m["settle_iteration"] for m in members],
            "any_unstable_drift": any(m["unstable_drift"] for m in members),
            "any_containment_violation": any(m.get("containment_violations") for m in members),
        }
    for r in runs:
        print(json.dumps({k: v for k, v in r.items() if k != "block_medians_over_m_star"}))
    print(json.dumps(summary, indent=2))
    if json_path is not None:
        json_path.write_text(json.dumps({"runs": runs, "groups": summary}, indent=2), "utf-8")


if __name__ == "__main__":
    main()
