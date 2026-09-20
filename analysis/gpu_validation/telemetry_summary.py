"""Phase 5A runbook §10: summarise an nvidia-smi telemetry log captured during a run.

The sampler is nvidia-smi itself, in a separate process (runbook §10) -- gpu_benchlab
has no in-run telemetry yet (Phase 6), and the benchmark loop must not poll anything.

Reports, per GPU: achieved sampling interval vs requested, min / median / max of each
numeric field, and every clock-event-reason bitmask seen, decoded with the constants
of the installed pynvml module (not hard-coded here).

Pre-registered adequacy rule for run-level context: median achieved interval
<= 2x requested, >= 10 samples inside the run, and no field entirely unavailable.
Per-iteration attribution is NOT claimed when an iteration is shorter than the
sampling interval -- NVML utilization is a time fraction over its own sample period.

Usage: python analysis/gpu_validation/telemetry_summary.py <telemetry.csv> <requested_ms>
"""

from __future__ import annotations

import csv
import json
import re
import statistics as st
import sys
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any


def reason_names() -> dict[int, str]:
    try:
        import pynvml
    except ImportError:
        return {}
    names: dict[int, str] = {}
    for attr in dir(pynvml):
        match = re.match(r"nvmlClocks(?:Event|Throttle)Reason(\w+)", attr)
        value = getattr(pynvml, attr)
        if match and isinstance(value, int) and value and value & (value - 1) == 0:
            names.setdefault(value, match.group(1))
    return names


def number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None  # "[N/A]", "[Not Supported]", "P0", hex masks...


def main() -> None:
    path, requested_ms = Path(sys.argv[1]), float(sys.argv[2])
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle, skipinitialspace=True))
    header = [re.sub(r"\s*\[.*\]$", "", h.strip()) for h in rows[0]]
    records = [dict(zip(header, (v.strip() for v in r), strict=True)) for r in rows[1:] if r]
    decode = reason_names()
    report: dict[str, Any] = {"file": str(path), "requested_interval_ms": requested_ms}
    for gpu in sorted({r["index"] for r in records}):
        mine = [r for r in records if r["index"] == gpu]
        stamps = [datetime.strptime(r["timestamp"], "%Y/%m/%d %H:%M:%S.%f") for r in mine]
        gaps = [(b - a).total_seconds() * 1000 for a, b in pairwise(stamps)]
        fields: dict[str, Any] = {}
        for name in header:
            if name in ("timestamp", "index"):
                continue
            values = [number(r[name]) for r in mine]
            present = [v for v in values if v is not None]
            if present:
                fields[name] = {
                    "min": min(present),
                    "median": st.median(present),
                    "max": max(present),
                    "missing": len(values) - len(present),
                }
            else:
                fields[name] = {"distinct": sorted({r[name] for r in mine})}
        reasons: dict[str, Any] = {}
        for column in (h for h in header if "reasons" in h):
            for raw_mask in sorted({r[column] for r in mine}):
                mask = int(raw_mask, 16) if raw_mask.lower().startswith("0x") else None
                bits = [n for bit, n in decode.items() if mask and mask & bit]
                reasons[raw_mask] = {
                    "samples": sum(r[column] == raw_mask for r in mine),
                    "decoded": bits,
                }
        adequate = (
            bool(gaps)
            and st.median(gaps) <= 2 * requested_ms
            and len(mine) >= 10
            and not any(
                "distinct" in f and f["distinct"] in (["[N/A]"], ["[Not Supported]"])
                for f in fields.values()
            )
        )
        report[f"gpu{gpu}"] = {
            "samples": len(mine),
            "window": [stamps[0].isoformat(), stamps[-1].isoformat()] if stamps else None,
            "achieved_interval_ms": {"median": st.median(gaps), "max": max(gaps)} if gaps else None,
            "fields": fields,
            "clock_event_reasons": reasons,
            "adequate_for_run_level_context": adequate,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
