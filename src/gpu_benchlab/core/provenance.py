"""Git provenance capture.

A result that cannot be tied to the code that produced it is not reproducible.
This records the commit and, importantly, whether the working tree was dirty —
a result produced from uncommitted changes cannot be reproduced from the commit
alone, and saying so is more useful than silently recording the commit as if it
were the whole truth.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from gpu_benchlab import __version__
from gpu_benchlab.core.schema import Provenance

__all__ = ["capture_provenance"]

_GIT_TIMEOUT_SECONDS = 5


def _git(args: list[str], cwd: Path) -> str | None:
    """Run a git command, returning None if git or the repo is unavailable.

    Uses an explicit argument list -- never a shell string -- so no part of this
    can be influenced by shell interpolation (CLAUDE.md §11).
    """
    try:
        # S603/S607: argv is a fixed list with shell=False, and no element derives
        # from user input, so there is no injection surface. "git" is resolved from
        # PATH deliberately -- a developer tool must use the developer's git.
        completed = subprocess.run(  # noqa: S603
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def capture_provenance(repo_root: Path | None = None) -> Provenance:
    """Record the package version and git state. Never raises."""
    root = repo_root or Path(__file__).resolve().parents[3]

    commit = _git(["rev-parse", "HEAD"], root)
    dirty: bool | None = None
    if commit is not None:
        status = _git(["status", "--porcelain"], root)
        dirty = bool(status) if status is not None else None

    return Provenance(benchlab_version=__version__, git_commit=commit, git_dirty=dirty)
