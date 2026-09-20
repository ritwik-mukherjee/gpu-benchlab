# Phase 5B operator notes — controlled rerun under unified inputs

Run 2026-09-20 on the same NVIDIA L4 instance as Phase 5A, at commit a02c94f with a
clean working tree (`git_dirty: false` on every result).

## What changed from Phase 5A, and what did not

Changed: **only the input path.** Both backends now take the benchmark input from
`core.inputs.synthetic_input`, so their input tensors are bit-identical before any
runtime-specific conversion. Verified on this GPU before the matrix ran
(`input-identity/input_identity_gpu.json`): PyTorch's device tensor and ORT's
device-resident OrtValue both equal the canonical array exactly, at batch 1 and 8.

Unchanged and deliberately so: the config files, warmup (100) and measured (1000)
iteration counts, the timing boundary and mechanisms (PyTorch CUDA events with a
synchronized host secondary series; ORT host clock around `run_with_iobinding`),
device-resident I/O, IEEE FP32 (`fp32_precision="ieee"` / `use_tf32=0`), NCHW,
5 repeats per cell in separate processes, alternating backend order between repeats,
and the telemetry method (`nvidia-smi` at 100 ms in a separate process, same fields).

## Pre-flight verification (all passed, before the matrix)

* input identity on the GPU, batch 1 and 8 — `input-identity/`
* ORT executes on the CUDA EP: 122/122 nodes, IOBinding device-resident, CUDA
  libraries loaded, PID visible to nvidia-smi — `ort-verify/`
* ORT's end-of-Run synchronization (S1) with its sensitivity control (S2)
* no silent CPU session with CUDA hidden (N1), for the ORT backend
* PyTorch: zero containment violations across all 10 runs, i.e. the synchronized host
  time is >= the CUDA-event time in every measured iteration

## Stability gate (pre-registered in Phase 5A, applied unchanged)

| Condition | Result |
|---|---|
| all runs `ok` | PASS (20/20) |
| no ANOMALY note | PASS (none) |
| within-run drift <= 2% | PASS (max 1.53%) |
| PyTorch containment violations = 0 | PASS |
| no thermal slowdown | PASS (no HwThermal/SwThermal flags) |
| run-to-run spread <= 5% | **FAIL in one cell: PyTorch batch 1 = 5.87%** (ORT b1 4.75%, ORT b8 2.93%, PyTorch b8 3.59%) |

**PyTorch batch 1 therefore does not meet the pre-registered stability criterion.** It is
reported as measured, not dropped and not re-run until it passed.

What the telemetry says about that cell: the SM clock was 2040 MHz in every sample of
every run and no `SwPowerCap` flag ever appeared (58.7-64.3 W against a 72 W limit), so
neither clock throttling nor the power cap accounts for the spread. Temperature rose
52 -> 78 C across repeats, but the per-run medians are not monotonic in temperature
(r3 = 5.851 ms at 67-71 C; r4 = 5.566 ms at 71-75 C).

That leaves the cause unestablished. A plausible hypothesis, NOT tested here: at batch 1
with the GPU below its power limit, this configuration may be bound by host-side kernel
launch rather than device execution, which would expose it to scheduling jitter on a
4 vCPU instance also running the telemetry sampler. Testing it needs an experiment this
run did not perform - for example comparing event time against host launch time per
iteration, pinning the sampler to another core, or CUDA graphs.

## Anomalies

None. No run was discarded, and no sample was removed.
