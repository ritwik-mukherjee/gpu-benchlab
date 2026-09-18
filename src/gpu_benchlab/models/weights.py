"""Download, cache and verify pinned weights files.

Uses only the standard library, so fetching and verification work without torch.

Every file is verified against the publisher's hash prefix, and its full SHA-256
is returned so it can be recorded on the result. A result is then traceable to
exact bytes, not just a model name.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from gpu_benchlab.core.errors import BackendError, UnavailableError
from gpu_benchlab.models.registry import WeightsSpec

__all__ = ["CACHE_ENV_VAR", "ResolvedWeights", "default_cache_dir", "ensure_weights", "sha256_of"]

CACHE_ENV_VAR = "GPU_BENCHLAB_CACHE"
_CHUNK = 1 << 20
_DOWNLOAD_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class ResolvedWeights:
    path: Path
    sha256: str
    downloaded: bool
    """True if this call downloaded the file; False if it was already cached."""


def default_cache_dir() -> Path:
    """``$GPU_BENCHLAB_CACHE/weights`` if set, else ``~/.cache/gpu-benchlab/weights``."""
    root = os.environ.get(CACHE_ENV_VAR)
    base = Path(root) if root else Path.home() / ".cache" / "gpu-benchlab"
    return base / "weights"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, spec: WeightsSpec) -> str:
    """Return the full SHA-256, raising if it does not match the pinned prefix."""
    digest = sha256_of(path)
    if not digest.startswith(spec.sha256_prefix):
        raise BackendError(
            f"Weights file {path} has SHA-256 {digest}, which does not match the pinned "
            f"prefix {spec.sha256_prefix!r}. The file is corrupt or not the pinned "
            "weights; refusing to benchmark it."
        )
    return digest


def ensure_weights(
    spec: WeightsSpec, *, cache_dir: Path | None = None, allow_download: bool = True
) -> ResolvedWeights:
    """Return a verified local copy of a pinned weights file.

    Raises:
        UnavailableError: the file is not cached and cannot be downloaded (no
            network, or downloads disabled). The weights are a missing
            precondition, not a failed run.
        BackendError: a file was obtained but its hash does not match the pin.
    """
    if not spec.url.startswith("https://"):
        # The registry is the only source of URLs; this guards against a bad entry.
        raise BackendError(f"Refusing non-HTTPS weights URL: {spec.url}")

    directory = cache_dir or default_cache_dir()
    target = directory / spec.filename

    if target.is_file():
        return ResolvedWeights(path=target, sha256=_verify(target, spec), downloaded=False)

    if not allow_download:
        raise UnavailableError(
            f"Weights {spec.filename} are not cached in {directory} and downloading is "
            "disabled. Run `gpu-bench models fetch` or enable downloads."
        )

    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=directory, suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        # S310: the URL comes only from the pinned registry and is asserted to be
        # https above, so no user-controlled scheme reaches urlopen.
        with (
            os.fdopen(fd, "wb") as out,
            urllib.request.urlopen(spec.url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as resp,  # noqa: S310
        ):
            for chunk in iter(lambda: resp.read(_CHUNK), b""):
                out.write(chunk)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise UnavailableError(f"Could not download pinned weights from {spec.url}: {exc}") from exc

    try:
        digest = _verify(tmp_path, spec)
    except BackendError:
        tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.replace(target)
    return ResolvedWeights(path=target, sha256=digest, downloaded=True)
