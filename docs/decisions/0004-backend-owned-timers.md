# ADR 0004 — The backend owns its timer; the engine never synchronizes

- **Status:** Accepted
- **Date:** 2026-09-18
- **Phase:** 2

## Context

CUDA kernel launches are asynchronous, so a host clock stopped immediately after
`execute()` measures launch overhead rather than execution. Something must force
the host to wait for the device before the clock stops.

The obvious design puts that in the engine: the measurement loop calls
`torch.cuda.synchronize()` around each iteration. It is simple, it is uniform,
and it is what most benchmarking scripts do.

## Decision

The engine never synchronizes. Each backend returns its own `Timer` from
`Backend.make_timer()`, and that timer encapsulates whatever synchronization or
device-side instrumentation is correct for that runtime. `TimingMechanism` is
recorded on every result.

The default, `WallClockTimer(synchronize=backend.synchronize)`, uses
`time.perf_counter_ns()` and calls the backend's hook before stopping. A backend
with a better mechanism overrides `make_timer()`.

## Rationale

Putting synchronization in the engine would be wrong four ways:

1. **It is not always the correct call.** ONNX Runtime synchronizes internally
   before `run()` returns — an extra device-wide sync would add cost that is not
   part of the workload. TensorRT wants a *stream* synchronization, not a device
   one. A CPU backend needs none.
2. **It is not always the best mechanism.** Where CUDA events are available they
   measure device time directly, excluding host-side launch and synchronization
   overhead. No placement of a host clock achieves that.
3. **It would put a CUDA import in the engine**, which must load on a machine with
   no GPU — the current primary development environment.
4. **It would make the mechanism invisible.** A result measured with CUDA events
   and one measured with a synchronized host clock are not directly comparable.
   Making the timer a backend concern forces the mechanism into the schema.

## Consequences

- Backends carry more responsibility, and a backend that implements timing badly
  produces bad numbers. Mitigated by the mechanism being recorded and by the
  default being correct for any backend that implements `synchronize()`.
- Comparing results across timing mechanisms requires an explicit decision. The
  comparison engine (Phase 8) must refuse or clearly annotate such comparisons.
- `TimingMechanism.CUDA_EVENT` is declared now but **not implemented**, so the
  schema does not change when Phase 3 adds it.
- **Unverified:** the contract has only been exercised against synchronous work
  (`time.sleep`) and the scripted timer. Whether the hook is placed correctly for
  real asynchronous CUDA execution cannot be established without a GPU. This is
  recorded in `docs/limitations.md` §2b.

## Alternatives rejected

- **Engine-level `torch.cuda.synchronize()`** — wrong for three of the four
  planned backends, and couples the core to PyTorch.
- **Backend returns its own duration from `execute()`** — would let a backend
  report a number with no visibility into how it was produced, and would remove
  the engine's ability to enforce the phase structure.
