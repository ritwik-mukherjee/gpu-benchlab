"""Inference backends.

Each module implements the contract in :mod:`gpu_benchlab.core.backend`.
Real runtimes arrive in Phases 3-5; :mod:`~gpu_benchlab.backends.fake` exists
only to test the framework itself and never produces a measurement.
"""

from __future__ import annotations

from gpu_benchlab.backends.fake import FakeBackend, FakeTimer

__all__ = ["FakeBackend", "FakeTimer"]
