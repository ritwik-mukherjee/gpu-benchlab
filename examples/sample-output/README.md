# Sample result — SIMULATED, not a measurement

> **These files contain no performance data.**
>
> They were produced by `backend: fake`, a seeded random latency model that exists
> to test the framework on a machine with no GPU. Every file here is stamped
> `"is_simulated": true` and `"timing_mechanism": "scripted"`. Nothing in this
> directory describes any GPU, model or inference runtime.

They are committed so that the result schema can be reviewed without running
anything. Reproduce with:

```bash
gpu-bench run --config examples/simulated-smoke-test.yaml --seed 42
```

`experiment_id` and `timestamp_utc` have been replaced with fixed placeholders so
the committed sample does not churn on every regeneration. Everything else is
exactly what the tool wrote.

## The four files

| File | Contents | Who reads it |
|---|---|---|
| `result.json` | The complete result. **Canonical** — the others are views of it. | Tooling |
| `metadata.json` | Config, full environment, backend, provenance, status. | Someone reproducing the run |
| `raw.json` | Every individual sample, warmup included, with units named. | Someone re-deriving the statistics |
| `summary.json` | Phases, latency statistics, throughput, errors. | Someone reviewing the outcome |

All four independently carry `is_simulated`, so no single file can be read in
isolation and mistaken for real data.

## What to look at

**Phase separation** — `summary.json → phases` keeps model load, engine build,
input preparation, warmup and measurement apart. An engine build can never leak
into inference latency.

**Raw samples are preserved** — `raw.json → latency_ms` has all 200 measured
samples, and `warmup_latency_ms` keeps the 10 discarded warmup iterations so
warmup convergence can be analysed rather than assumed. In this sample the warmup
ramp decays from ~54 ms to ~14.6 ms, which is the shape the simulated model was
given.

**Statistics are derived, not asserted** — every value under `latency_ms` is
recomputable from `raw.json`. `percentile_method` and `stddev_ddof` are recorded
because tools disagree on both.

**Confidence is stated** — `low_confidence_percentiles` lists `p99`, because 200
samples cannot support a stable p99 (see `docs/methodology.md` §6). The value is
still reported; it is just not presented as trustworthy.

**Throughput names its unit and its formula** — and the observed-throughput figure
is deliberately absent here, because under a scripted timer the per-iteration
durations are fabricated while the loop's wall time is real. Dividing one by the
other would produce a number nothing supports, so none is emitted.
