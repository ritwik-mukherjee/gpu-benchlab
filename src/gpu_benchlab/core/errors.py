"""Exception taxonomy for the benchmark core.

Errors are first-class benchmark outcomes (PRD §33), so these exceptions exist to
be *caught by the engine and recorded as results*, not to escape to the user. The
taxonomy matters because the resulting status differs:

* :class:`UnsupportedConfigurationError` -> ``status = unsupported`` (valid tool,
  invalid combination for this hardware/backend/model)
* :class:`OutOfMemoryError` -> ``status = failed`` (valid combination, ran out of
  memory — this is a *measurement*, not a crash)
* :class:`BackendError` -> ``status = failed``
* :class:`InvalidSampleError` -> ``status = failed`` (the timing data itself is
  untrustworthy, so no statistics may be reported from it)

:class:`ConfigurationError` is the exception: it is raised before any measurement
is attempted and is surfaced to the user, because a malformed config is a user
error rather than a benchmark outcome.
"""

from __future__ import annotations

__all__ = [
    "BackendError",
    "BenchLabError",
    "ConfigurationError",
    "InvalidSampleError",
    "OutOfMemoryError",
    "UnsupportedConfigurationError",
]


class BenchLabError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(BenchLabError):
    """A benchmark configuration is malformed or self-contradictory.

    Raised before execution begins. This is a user error, not a benchmark result.
    """


class UnsupportedConfigurationError(BenchLabError):
    """The configuration is well-formed but cannot run on this hardware/backend.

    Example: FP8 on a GPU whose compute capability is below 8.9. This is recorded
    as ``status = unsupported`` with a reason, never silently skipped.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class BackendError(BenchLabError):
    """A backend failed during load, build or execution."""


class OutOfMemoryError(BackendError):
    """The device ran out of memory.

    Deliberately distinct from a generic :class:`BackendError`: an OOM at a given
    batch size is a legitimate, informative measurement about that configuration
    (PRD §12), and must be recorded rather than treated as a tool crash.
    """


class InvalidSampleError(BenchLabError):
    """Timing samples are not usable as measurements.

    Raised for empty sample sets, NaN, infinity, or negative durations. These are
    never silently dropped or coerced: discarding a NaN changes every statistic
    computed from the remaining samples, and a tool that does so quietly reports a
    number that no measurement supports.
    """
