"""Shared helpers for the Phase 5A GPU validation scripts (docs/runbooks/gpu-validation.md).

These scripts are validation PROCEDURES, not benchmark features. Each one records
every pre-registered check as PASS / FAIL / INCONCLUSIVE / INFO, writes all raw
values to a JSON evidence file, and exits 1 if any check FAILED. None of them
changes a system setting.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

PASS, FAIL, INCONCLUSIVE, INFO = "PASS", "FAIL", "INCONCLUSIVE", "INFO"
EXIT_NO_CUDA = 2


class Checks:
    """Ordered record of pre-registered checks and the evidence behind each."""

    def __init__(self, script: str) -> None:
        self.script = script
        self.items: list[dict[str, Any]] = []

    def add(self, check_id: str, status: str, description: str, /, **evidence: Any) -> None:
        """Record one check. The first three are positional-only, so any evidence key
        (including "status") is safe to pass."""
        self.items.append(
            {"id": check_id, "status": status, "description": description, "evidence": evidence}
        )

    def expect(self, check_id: str, condition: bool, description: str, /, **evidence: Any) -> None:
        self.add(check_id, PASS if condition else FAIL, description, **evidence)

    def finish(self, out_dir: Path, extra: dict[str, Any] | None = None) -> NoReturn:
        """Write <script>.json, print one line per check, exit 1 on any FAIL."""
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "script": self.script,
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "checks": self.items,
            **(extra or {}),
        }
        path = out_dir / f"{self.script}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        for item in self.items:
            print(f"[{item['status']:<12}] {item['id']}: {item['description']}")
        failed = [i["id"] for i in self.items if i["status"] == FAIL]
        print(f"\nevidence: {path}")
        print(f"FAILED: {failed}" if failed else "no check failed")
        raise SystemExit(1 if failed else 0)


def require_cuda_torch() -> Any:
    """Import torch and refuse (exit 2) unless a CUDA device is usable. Never falls back."""
    try:
        import torch
    except ImportError:
        raise SystemExit(
            "torch is not installed: this step needs a CUDA build of PyTorch."
        ) from None
    if torch.version.cuda is None or not torch.cuda.is_available():
        print(
            f"No usable CUDA device (torch {torch.__version__}, cuda build {torch.version.cuda}). "
            "This step must run on an NVIDIA GPU machine; nothing was measured.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_NO_CUDA)
    return torch


def nvidia_smi(*args: str) -> str | None:
    """Run nvidia-smi with a fixed argv (no shell). None if it is absent or fails."""
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, shell=False, no user input
            ["nvidia-smi", *args],  # noqa: S607 - the driver's tool, resolved from PATH
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def compute_app_pids() -> set[int] | None:
    """PIDs nvidia-smi lists as using the GPU. None if unavailable."""
    text = nvidia_smi("--query-compute-apps=pid", "--format=csv,noheader,nounits")
    if text is None:
        return None
    return {int(line) for line in text.split() if line.strip().isdigit()}


def loaded_libraries(*needles: str) -> dict[str, list[str]] | None:
    """Shared libraries mapped into THIS process whose path contains each needle.

    Linux only (/proc/self/maps). This is direct evidence of which CUDA/cuDNN/cuBLAS
    files a runtime actually loaded, independent of what it reports about itself.
    """
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return None
    paths = {line.split()[-1] for line in maps.read_text().splitlines() if "/" in line}
    return {n: sorted(p for p in paths if n in Path(p).name) for n in needles}
