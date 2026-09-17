# ADR 0001 — Store results as JSON files on disk

- **Status:** Accepted
- **Date:** 2026-09-18
- **Phase:** 0

## Context

Benchmark results must be durable, inspectable, comparable and reproducible. The
candidates considered were JSON/JSONL files, SQLite, Parquet and CSV.

Volume is genuinely small at this stage: a 100-iteration run produces 100 floats.
Even a large matrix (5 models × 4 backends × 4 precisions × 6 batch sizes × 3 repeats)
is ~1,440 results — a few MB of JSON.

## Decision

One directory per experiment:

```
results/<experiment-id>/
    metadata.json   configuration + environment + git commit
    raw.json        every individual latency sample
    summary.json    derived statistics
    logs/
```

JSON is the **canonical** representation. CSV is an export format only.

## Rationale

- **Inspectable without tooling.** A sceptic can open the file and check the numbers.
  This matters more than query speed for a project whose thesis is "don't trust
  claims, check the data".
- **Diffable and reviewable** in git, unlike a binary database.
- **No migration story needed on day one.** Schema evolution is handled by
  `schema_version` plus additive fields.
- **Portable** across the machines where benchmarks actually run.

## Consequences

- Cross-experiment queries require loading many files. Acceptable at current volume.
- **When it stops being acceptable:** a SQLite index over the JSON files will be
  added — JSON stays canonical, SQLite becomes a derived cache that can be deleted
  and rebuilt. Trigger: result counts in the tens of thousands, or dashboard load
  times becoming noticeable.
- Parquet becomes worth revisiting only if raw sample volume grows by orders of
  magnitude (e.g. per-token traces for long LLM generations).

## Alternatives rejected

- **SQLite from the start** — adds a schema migration burden and makes results
  opaque to casual inspection, for query performance not yet needed.
- **CSV as canonical** — cannot represent nested environment metadata without
  flattening it into something lossy and unreadable.
