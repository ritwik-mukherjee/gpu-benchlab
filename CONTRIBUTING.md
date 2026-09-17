# Contributing

Thanks for your interest. This project has one rule that matters more than all the
others, and it is not negotiable.

## The rule

**Never fabricate a measurement.**

Every performance number in this repository — in code, docs, README, a chart, a
commit message or a PR description — must be traceable to an actual benchmark run
whose raw samples are stored. If you cannot point at the result file, the number
does not go in.

If something cannot be measured, say so. `unsupported`, `unavailable`, `failed` and
`skipped` are all valid, useful outcomes. A missing number is honest; a plausible
invented one is not.

The full engineering rules are in [CLAUDE.md](CLAUDE.md). Please read it before your
first PR — it applies to human and AI contributors equally.

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Verify:

```bash
pytest
gpu-bench hardware
```

The test suite passes on a machine with no GPU. If it does not, that is a bug.

## Before opening a PR

```bash
ruff check .
ruff format .
mypy
pytest
```

All four must be clean. CI runs them on Ubuntu and Windows, Python 3.10 and 3.12.

## What a good PR looks like

1. **Say what you changed and why.**
2. **Say what you tested it on.** Include the GPU, driver and CUDA version if the
   change touches GPU behaviour. `gpu-bench hardware --json` output is ideal.
3. **Say what you could not test.** This is not a weakness in a PR; it is the single
   most useful thing you can tell a reviewer. If you wrote a TensorRT code path but
   have no TensorRT, say so explicitly and we will mark it unverified in
   [docs/limitations.md](docs/limitations.md).

## Contributing benchmark results

Results from real hardware are genuinely valuable — the project is only as good as
the range of GPUs it has data for.

To contribute results, include the complete `results/<experiment-id>/` directory,
raw samples included. Results without raw samples cannot be accepted, because they
cannot be independently checked or recomputed.

Please do not tune a configuration until it produces a flattering number and submit
only that run. Submit what you measured.

## Adding a backend

Backends live in `src/gpu_benchlab/backends/`. A backend declares its own capability
constraints and implements the common benchmark interface — but it is *not* required
to contort itself into a shape that does not fit. If your runtime measures something
the vision-model abstraction has no concept of (per-token latency, for instance),
model it honestly rather than flattening it.

See [docs/architecture.md](docs/architecture.md) §6.

## Commits

Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
Small and logical beats one large one.

## Reporting a measurement bug

Measurement-validity bugs are the highest-severity class of bug in this project —
higher than crashes. A tool that crashes is annoying; a tool that quietly reports a
wrong latency is actively harmful.

If you believe a number is wrong, please open an issue with the result directory and
your reasoning. Those get looked at first.
