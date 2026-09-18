"""Pinned vs random ResNet-50 weights on the host CPU: re-derivation and comparison.

Question (docs/models.md §8): do seeded random weights cost the same to execute on
CPU as the pinned IMAGENET1K_V2 weights? If they do, random-init runs could stand
in for pinned runs when validating the harness. If not, they cannot.

This script uses ONLY the Python standard library and reads ONLY stored raw
samples, so every number it prints is re-derived independently of gpu_benchlab's
own statistics code, and is checked against what gpu_benchlab stored.

Usage:
    python analysis/ab_weights_cpu.py [results_root]

Default results_root: results/ab   (expects pinned/ and random/ subdirectories)
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

TOL = 1e-9


def pct(values: list[float], p: float) -> float:
    """Linear interpolation between order statistics (numpy 'linear' / Excel INC)."""
    v = sorted(values)
    k = (len(v) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def load_runs(root: Path) -> list[dict]:
    runs = []
    for arm in ("pinned", "random"):
        for directory in sorted((root / arm).glob("*/")):
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            raw = json.loads((directory / "raw.json").read_text(encoding="utf-8"))
            runs.append({"arm": arm, "dir": directory, "result": result, "raw": raw})
    runs.sort(key=lambda r: r["result"]["timestamp_utc"])
    return runs


def rederive(run: dict) -> dict[str, float]:
    x = run["raw"]["latency_ms"]
    stored = run["result"]["latency"]
    mine = {
        "mean_ms": st.fmean(x),
        "median_ms": st.median(x),
        "stddev_ms": st.stdev(x),
        "min_ms": min(x),
        "max_ms": max(x),
        **{f"p{p}_ms": pct(x, p) for p in (50, 90, 95, 99)},
    }
    mismatches = [k for k, v in mine.items() if abs(v - stored[k]) > TOL]
    if mismatches:
        raise SystemExit(f"{run['dir']}: stored statistics disagree with raw samples: {mismatches}")
    return mine


def blocks(x: list[float], size: int = 20) -> list[float]:
    return [st.fmean(x[i : i + size]) for i in range(0, len(x), size)]


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/ab")
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"No results under {root}")

    print(f"Runs under {root}, in execution order:\n")
    for i, run in enumerate(runs, 1):
        r = run["result"]
        info = r["model_info"]
        prefix = "P" if run["arm"] == "pinned" else "R"
        run["label"] = f"{prefix}{sum(1 for q in runs[:i] if q['arm'] == run['arm'])}"
        print(
            f"  {run['label']}  {r['timestamp_utc']}  {run['arm']:<6}"
            f"  weights={info['weights']:<14}"
            f" sha256={(info['weights_sha256'] or '-')[:12]:<12}  status={r['status']}"
            f"  device={r['device_kind']}  mechanism={r['timing_mechanism']}"
        )

    print(
        "\nPer-run statistics, re-derived from raw.json with the stdlib"
        " (all match stored to 1e-9):\n"
    )
    columns = ("mean", "p50", "p95", "p99", "stdev", "min", "max")
    header = f"  {'run':<4}" + "".join(f"{c:>9}" for c in columns) + f"{'thru(s/s)':>11}"
    print(header)
    for run in runs:
        s = rederive(run)
        run["stats"] = s
        run["throughput"] = 1000.0 / s["mean_ms"]  # batch 1: samples/sec = 1 / mean latency
        print(
            f"  {run['label']:<4}{s['mean_ms']:9.2f}{s['p50_ms']:9.2f}{s['p95_ms']:9.2f}"
            f"{s['p99_ms']:9.2f}{s['stddev_ms']:9.2f}{s['min_ms']:9.2f}{s['max_ms']:9.2f}"
            f"{run['throughput']:11.3f}"
        )
    print("  (ms; throughput = batch_size / mean latency, batch_size 1, samples/sec)")

    print("\nWarmup samples (ms) — excluded from all statistics above:\n")
    for run in runs:
        w = run["raw"]["warmup_latency_ms"]
        print(f"  {run['label']:<4} mean {st.fmean(w):7.2f}  [{', '.join(f'{v:.0f}' for v in w)}]")

    print("\nTemporal drift within each run — mean of consecutive 20-iteration blocks (ms):\n")
    for run in runs:
        b = blocks(run["raw"]["latency_ms"])
        print(
            f"  {run['label']:<4} "
            + "  ".join(f"{v:7.1f}" for v in b)
            + f"   (last-first {b[-1] - b[0]:+.1f})"
        )

    print("\nBetween-arm vs within-arm variation (p50, ms):\n")
    by_arm = {
        arm: [r["stats"]["p50_ms"] for r in runs if r["arm"] == arm] for arm in ("pinned", "random")
    }
    for arm, vals in by_arm.items():
        print(
            f"  {arm:<6} p50 per run: {', '.join(f'{v:.2f}' for v in vals)}"
            f"   range {max(vals) - min(vals):.2f}"
        )
    arm_means = {arm: st.fmean(v) for arm, v in by_arm.items()}
    diff = arm_means["random"] - arm_means["pinned"]
    within = max(max(v) - min(v) for v in by_arm.values())
    print(
        f"\n  mean of per-run p50: pinned {arm_means['pinned']:.2f},"
        f" random {arm_means['random']:.2f}"
    )
    print(f"  difference (random - pinned): {diff:+.2f} ms")
    print(f"  largest within-arm run-to-run range: {within:.2f} ms")
    print(
        "  => difference is "
        + ("SMALLER" if abs(diff) < within else "LARGER")
        + " than the run-to-run range within a single arm."
    )
    print(
        f"\n  Runs per arm: {', '.join(f'{k}={len(v)}' for k, v in by_arm.items())}."
        " No significance test is computed: with two runs per arm, one alternation"
    )
    print("  sequence and visible within-run drift, the design does not support one.")


if __name__ == "__main__":
    main()
