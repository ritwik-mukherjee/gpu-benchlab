"""Host (OS, CPU, memory, Python) detection.

Nothing here is NVIDIA-specific. It exists because a latency measurement is not
reproducible without knowing the machine that produced it: CPU model affects
preprocessing and host-side launch overhead, and Python version affects the
interpreter overhead inside the measurement loop.
"""

from __future__ import annotations

import platform
import socket
import sys

from gpu_benchlab.hardware.types import HostInfo

__all__ = ["probe_host"]


def _cpu_model() -> str | None:
    """Best-effort human-readable CPU name.

    `platform.processor()` is informative on Windows and macOS but frequently
    returns a bare architecture string on Linux, so /proc/cpuinfo is preferred
    there when available.
    """
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass

    proc = platform.processor()
    return proc or None


def _cpu_counts() -> tuple[int | None, int | None]:
    """Return (physical_cores, logical_cores)."""
    physical: int | None = None
    logical: int | None = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
    except ImportError:
        import os

        logical = os.cpu_count()
    return physical, logical


def _total_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        return None


def probe_host() -> HostInfo:
    """Collect host information. Never raises."""
    physical, logical = _cpu_counts()
    return HostInfo(
        hostname=socket.gethostname(),
        os_name=platform.system(),
        os_version=platform.version(),
        os_release=platform.release() or None,
        architecture=platform.machine(),
        cpu_model=_cpu_model(),
        cpu_physical_cores=physical,
        cpu_logical_cores=logical,
        total_memory_bytes=_total_memory_bytes(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        python_executable=sys.executable,
    )
