"""Result persistence.

Layout per ADR 0001 — one directory per experiment::

    results/<experiment-id>/
        metadata.json   configuration, environment, backend, provenance, status
        raw.json        every individual sample
        summary.json    derived statistics and throughput
        result.json     the complete result (canonical; the others are views)

``result.json`` is canonical. The three split files exist because they are what
people actually open: a reviewer wants `summary.json`, an analyst wants
`raw.json`, and a reproducer wants `metadata.json`. They are written from the
same object, so they cannot disagree.

Simulated results are stored under a ``sim-`` prefixed directory. That makes the
distinction visible in a file listing, before anyone opens anything.
"""

from __future__ import annotations

import json
from pathlib import Path

from gpu_benchlab.core.schema import BenchmarkResult

__all__ = ["DEFAULT_RESULTS_DIR", "ResultStore", "result_directory_name"]

DEFAULT_RESULTS_DIR = Path("results")

SIMULATED_PREFIX = "sim-"


def result_directory_name(result: BenchmarkResult) -> str:
    """Directory name for a result, prefixed when simulated."""
    prefix = SIMULATED_PREFIX if result.is_simulated else ""
    return f"{prefix}{result.experiment_id}"


class ResultStore:
    """Writes and reads benchmark results as JSON on disk."""

    def __init__(self, root: str | Path = DEFAULT_RESULTS_DIR) -> None:
        self.root = Path(root)

    # -- write -------------------------------------------------------------------------

    def save(self, result: BenchmarkResult) -> Path:
        """Persist a result. Returns the directory it was written to."""
        directory = self.root / result_directory_name(result)
        directory.mkdir(parents=True, exist_ok=True)

        _write(directory / "result.json", result.model_dump(mode="json"))

        _write(
            directory / "metadata.json",
            {
                "schema_version": result.schema_version,
                "experiment_id": result.experiment_id,
                "group_id": result.group_id,
                "repeat_index": result.repeat_index,
                "timestamp_utc": result.timestamp_utc,
                "status": result.status.value,
                "is_simulated": result.is_simulated,
                "configuration": result.configuration.model_dump(mode="json"),
                "backend": result.backend.model_dump(mode="json"),
                "environment": result.environment.model_dump(mode="json"),
                "timing_mechanism": result.timing_mechanism.value,
                "measures_steady_state": result.measures_steady_state,
                "provenance": result.provenance.model_dump(mode="json"),
                "notes": result.notes,
            },
        )

        _write(
            directory / "raw.json",
            {
                "schema_version": result.schema_version,
                "experiment_id": result.experiment_id,
                "is_simulated": result.is_simulated,
                "timing_mechanism": result.timing_mechanism.value,
                "units": "milliseconds",
                **result.raw_samples.model_dump(mode="json"),
            },
        )

        _write(
            directory / "summary.json",
            {
                "schema_version": result.schema_version,
                "experiment_id": result.experiment_id,
                "status": result.status.value,
                "is_simulated": result.is_simulated,
                "backend": result.backend.name,
                "precision": result.configuration.precision.value,
                "batch_size": result.configuration.batch_size,
                "timing_mechanism": result.timing_mechanism.value,
                "measures_steady_state": result.measures_steady_state,
                "phases": result.phases.model_dump(mode="json"),
                "latency_ms": (result.latency.model_dump(mode="json") if result.latency else None),
                "throughput": [t.model_dump(mode="json") for t in result.throughput],
                "errors": [e.model_dump(mode="json") for e in result.errors],
                "notes": result.notes,
            },
        )

        (directory / "logs").mkdir(exist_ok=True)
        return directory

    # -- read --------------------------------------------------------------------------

    def load(self, experiment_id: str) -> BenchmarkResult:
        """Load a result by id, with or without the simulated prefix."""
        for name in (experiment_id, f"{SIMULATED_PREFIX}{experiment_id}"):
            path = self.root / name / "result.json"
            if path.is_file():
                return BenchmarkResult.model_validate_json(path.read_text(encoding="utf-8"))
        raise FileNotFoundError(
            f"No result found for experiment id {experiment_id!r} in {self.root}"
        )

    def list_results(self) -> list[BenchmarkResult]:
        """Load every result under the root, newest first.

        Unreadable directories are skipped rather than aborting the listing --
        one corrupt result must not make the rest unreadable.
        """
        if not self.root.is_dir():
            return []

        results: list[BenchmarkResult] = []
        for path in sorted(self.root.glob("*/result.json")):
            try:
                results.append(
                    BenchmarkResult.model_validate_json(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError):
                continue
        return sorted(results, key=lambda r: r.timestamp_utc, reverse=True)


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
