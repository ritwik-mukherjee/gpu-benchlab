"""Phase 5A runbook §3: gpu_benchlab's NVML success path vs nvidia-smi, field by field.

nvidia-smi is read immediately before AND after gpu_benchlab's detection, so a
dynamic value (temperature, power, clocks, utilization, memory used) is checked
against the bracket of two independent readings rather than a single instant.

Pre-registered rules:
  static fields     exact match (after unit conversion)              else FAIL
  dynamic fields    inside the smi bracket widened by a tolerance    else INCONCLUSIVE
                    (investigate and explain; never "fixed" by editing values)
  availability      tool None where smi has a value                  FAIL
                    tool 0 where smi says N/A (unavailable -> zero)  FAIL
  SM count          GPUInfo.multiprocessor_count vs torch's multi_processor_count;
                    a mismatch means the field does not hold the SM count  FAIL

Usage: python analysis/gpu_validation/nvml_vs_smi.py <out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from _common import FAIL, INCONCLUSIVE, INFO, PASS, Checks, nvidia_smi

from gpu_benchlab.hardware.detect import detect_environment

MIB = 1024 * 1024
# (GPUInfo field, nvidia-smi --query-gpu field, rule, tolerance for dynamic fields)
FIELDS: list[tuple[str, str, str, float]] = [
    ("name", "name", "exact", 0),
    ("uuid", "uuid", "exact", 0),
    ("pci_bus_id", "pci.bus_id", "exact_ci", 0),
    ("serial", "serial", "exact", 0),
    ("compute_capability", "compute_cap", "exact", 0),
    ("memory_total_bytes", "memory.total", "mib", 0),
    ("power_limit_watts", "enforced.power.limit", "float", 0.5),
    ("max_sm_clock_mhz", "clocks.max.sm", "float", 0),
    ("max_memory_clock_mhz", "clocks.max.mem", "float", 0),
    ("persistence_mode", "persistence_mode", "enabled", 0),
    ("compute_mode", "compute_mode", "lower", 0),
    ("temperature_celsius", "temperature.gpu", "dynamic", 2),
    ("utilization_gpu_percent", "utilization.gpu", "dynamic", 10),
    ("utilization_memory_percent", "utilization.memory", "dynamic", 10),
    ("power_usage_watts", "power.draw", "dynamic", 5),
    ("sm_clock_mhz", "clocks.sm", "dynamic", 50),
    ("memory_clock_mhz", "clocks.mem", "dynamic", 50),
    ("memory_used_bytes", "memory.used", "dynamic_mib", 64),
    ("memory_free_bytes", "memory.free", "dynamic_mib", 64),
]
UNAVAILABLE = {"[n/a]", "n/a", "[not supported]", "not supported", "[unknown error]", ""}


def query() -> list[dict[str, str | None]] | None:
    keys = ["index"] + [smi for _, smi, _, _ in FIELDS]
    text = nvidia_smi(f"--query-gpu={','.join(keys)}", "--format=csv,noheader,nounits")
    if text is None:
        return None
    rows = []
    for line in text.strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        rows.append(
            {
                k: (None if v.lower() in UNAVAILABLE else v)
                for k, v in zip(keys, values, strict=True)
            }
        )
    return rows


def as_number(value: str | None, rule: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number * MIB if rule in ("mib", "dynamic_mib") else number


def compare(tool: Any, before: str | None, after: str | None, rule: str, tol: float) -> str:
    if tool is None:
        return PASS if before is None and after is None else FAIL
    if before is None and after is None:
        return FAIL if tool == 0 else INCONCLUSIVE  # tool has a value smi cannot show
    if rule == "exact":
        return PASS if str(tool) == before else FAIL
    if rule == "exact_ci":
        return PASS if str(tool).lower() == str(before).lower() else FAIL
    if rule == "lower":
        return PASS if str(tool).lower() == str(before).lower() else FAIL
    if rule == "enabled":
        return PASS if bool(tool) == (str(before).lower() == "enabled") else FAIL
    if rule in ("mib", "float"):
        expected = as_number(before, rule)
        return PASS if expected is not None and abs(float(tool) - expected) <= tol else FAIL
    lo_hi = [x for x in (as_number(before, rule), as_number(after, rule)) if x is not None]
    widen = tol * MIB if rule == "dynamic_mib" else tol
    inside = min(lo_hi) - widen <= float(tool) <= max(lo_hi) + widen
    return PASS if inside else INCONCLUSIVE


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase5a/nvml")
    checks = Checks("nvml_vs_smi")

    listing = nvidia_smi("-L")
    header = nvidia_smi()
    before = query()
    env = detect_environment(include_frameworks=False)  # the NVML path under test
    after = query()
    if before is None or after is None or listing is None or header is None:
        checks.add(
            "smi",
            FAIL,
            "nvidia-smi failed: not installed, or this driver rejects a queried field "
            "(see nvidia-smi --help-query-gpu); nothing was compared",
        )
        checks.finish(out_dir, {"environment": env.model_dump(mode="json")})

    checks.expect(
        "status",
        env.detection_status.value == "ok",
        "gpu_benchlab NVML detection succeeded",
        detection_status=env.detection_status.value,
        error=env.detection_error,
    )
    smi_count = len([line for line in listing.splitlines() if line.startswith("GPU ")])
    checks.expect(
        "count",
        len(env.gpus) == smi_count,
        "GPU count matches nvidia-smi -L",
        tool=len(env.gpus),
        smi=smi_count,
    )
    driver = before[0]["driver_version"] if before else None
    checks.expect(
        "driver",
        env.nvml.driver_version == driver,
        "driver version matches",
        tool=env.nvml.driver_version,
        smi=driver,
    )
    cuda_line = next((w for w in header.split("|") if "CUDA Version" in w), "")
    checks.expect(
        "cuda_driver_version",
        env.nvml.cuda_driver_version is not None
        and f"CUDA Version: {env.nvml.cuda_driver_version}" in cuda_line,
        "max CUDA version supported by the driver matches the nvidia-smi header",
        tool=env.nvml.cuda_driver_version,
        smi=cuda_line.strip(),
    )

    for gpu in env.gpus:
        b = next((r for r in before if r["index"] == str(gpu.index)), None)
        a = next((r for r in after if r["index"] == str(gpu.index)), None)
        if b is None or a is None:
            checks.add(f"gpu{gpu.index}", FAIL, "index missing from nvidia-smi output")
            continue
        for field, smi, rule, tol in FIELDS:
            tool = getattr(gpu, field)
            status = compare(tool, b[smi], a[smi], rule, tol)
            checks.add(
                f"gpu{gpu.index}.{field}",
                status,
                f"{field} vs nvidia-smi {smi} ({rule})",
                tool=tool,
                smi_before=b[smi],
                smi_after=a[smi],
                tolerance=tol,
            )

    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                uuid = str(getattr(props, "uuid", ""))
                match = next((g for g in env.gpus if g.uuid and uuid and uuid in g.uuid), None)
                checks.expect(
                    f"torch{i}.sm_count",
                    match is not None and match.multiprocessor_count == props.multi_processor_count,
                    "GPUInfo.multiprocessor_count equals torch's SM count (matched by UUID)",
                    torch_uuid=uuid,
                    torch_sm_count=props.multi_processor_count,
                    nvml_gpu=match.index if match else None,
                    nvml_multiprocessor_count=match.multiprocessor_count if match else None,
                )
        else:
            checks.add("sm_count", INCONCLUSIVE, "torch has no CUDA device; SM count unchecked")
    except ImportError:
        checks.add("sm_count", INCONCLUSIVE, "torch not installed; SM count unchecked")

    checks.add("raw", INFO, "raw readings stored", smi_listing=listing, smi_header=header)
    checks.finish(
        out_dir,
        {"environment": env.model_dump(mode="json"), "smi_before": before, "smi_after": after},
    )


if __name__ == "__main__":
    main()
