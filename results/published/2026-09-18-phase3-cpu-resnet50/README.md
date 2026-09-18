# 2026-09-18 — Phase 3 CPU ResNet-50 runs (evidence)

**These are real measurements of the host CPU of the development laptop. They are
NOT GPU performance and contain no NVIDIA data.** They are published as evidence
for the Phase 3 methodology findings, not as benchmark results to quote.

Machine: Intel Core i7-8565U (4C/8T, 15 W ULV), Windows 11 Pro 26200, 15.8 GiB RAM,
no NVIDIA GPU. PyTorch 2.14.0+cpu, torchvision 0.29.0+cpu, 4 intra-op threads.
Code: commit `1f153b9`. Model: ResNet-50, IEEE FP32, batch 1, 10 warmup + 100
measured iterations, `wall_clock` timing (PyTorch CPU ops are synchronous).

Every run directory is exactly what `gpu-bench run` wrote, copied unmodified.

## Contents

| Directory | Run | Weights | Power | Notes |
|---|---|---|---|---|
| `single/cpu-1bbb675f42d2` | P0 | pinned `IMAGENET1K_V2` (SHA-256 `11ad3fa6…`) | AC (checked via `Win32_Battery` immediately before) | Clean run. Clean git tree. |
| `ab/pinned/cpu-16597046731a` | P1 | pinned | AC → battery mid-run | Power switched ≈ 10 s in (≈ iteration 60); **system suspend inside iteration 79.** |
| `ab/random/cpu-0c0adf71660e` | R1 | seeded random init | battery | |
| `ab/pinned/cpu-b908255038f0` | P2 | pinned | battery | Multi-second outliers after resume. |
| `ab/random/cpu-b3bdf9a181f0` | R2 | seeded random init | battery | |

A/B runs executed in the order P1, R1, P2, R2. They carry `git_dirty: true` because
the random-weights config (`analysis/configs/resnet50-pytorch-cpu-fp32-random.yaml`)
was an uncommitted file at the time; the source code was unchanged at `1f153b9`, and
each run's full configuration is stored verbatim in its `result.json`.

These results predate the `power_plugged` field (environment schema 1.1), so power
source is not recorded inside them; it was established from the OS log below.

## Windows System log, 2026-09-18 (UTC)

Extracted with `Get-WinEvent -FilterHashtable @{LogName='System'; ...}`, filtered to
Kernel-Power / Power-Troubleshooter / Kernel-General:

```
05:00:20  Kernel-Power 105   Power source change            (AcOnline = false)
05:00:25  Kernel-Power 42    The system is entering sleep.
05:00:27  Kernel-Power 107   The system has resumed from sleep.   (timestamp predates the clock correction below)
05:10:54  Kernel-General 1   System time changed to 05:10:54.50 from 05:00:27.47
05:10:54  Kernel-Power 130/131  Firmware S3 times
05:10:56  Power-Troubleshooter 1  The system has returned from a low power state.
```

P1's start is reconstructed from its stored durations: it ended at 05:11:06 after
651.5 s of measurement, 0.77 s of warmup and 3.0 s of model load, so it began at about
05:00:10. The 05:00:20 power change therefore fell ≈ 10 s into P1, around measured
iteration 60; P1's first large outlier is iteration 62 (1,205 ms). The timing is
consistent, not proof of cause.

The suspend lasted 05:00:27 → 05:10:54 = 627 s. Run P1's iteration 79 recorded
627,219.9 ms. The timer (`time.perf_counter_ns`, QueryPerformanceCounter on Windows)
kept counting through S3 sleep.

## What these runs do and do not show

- **P0** shows within-run non-stationarity on this laptop CPU: iterations 0–39
  average ≈ 88 ms and iterations 40–99 average ≈ 111–118 ms. Cause **unconfirmed**
  (no clock or power telemetry was recorded).
- **The A/B comparison is inconclusive.** P1 switched from AC to battery mid-run and
  spans a system suspend; R1, P2 and R2 ran entirely on battery after the resume; P2
  has post-resume outliers. The pinned-vs-random difference in
  mean per-run p50 (+31.1 ms) is smaller than the pinned arm's own run-to-run range
  (104.7 ms). No conclusion about whether weight values affect CPU latency is justified.

Reproduce the analysis: `python analysis/ab_weights_cpu.py results/published/2026-09-18-phase3-cpu-resnet50/ab`
