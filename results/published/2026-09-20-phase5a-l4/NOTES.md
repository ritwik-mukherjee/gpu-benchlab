
## 2026-09-20 — deviation recorded BEFORE the §11 controlled runs

The runbook says: if §6 finds a settle iteration k > 50, set warmup to 2k in all
controlled configs. That rule is inapplicable as written, and I am not tuning it
after the fact; I am recording why and what I did instead.

§6 measured settle indices of 0–1270 for PyTorch and 380–None for ORT, while 95–100%
of all block medians lay within ±2% of steady state and drift was ≤0.5%. The statistic
is therefore dominated by ±2% iteration jitter: one late outlier block resets it, and
"None" has no 2k.

What the segment medians actually show (results/phase5a-raw/analysis/warmup.json):
  PyTorch  first 10 warmup iterations +0.6% to +2.0% of steady state -> 10 suffice
  ORT      first 10 are 7.1-8.6% FASTER than steady state, converging from below;
           by iterations 10-50 they are within ~1.7%. Cause is measured, not guessed:
           ORT saturates the L4 72 W power cap (47 samples flagged SwPowerCap) and the
           SM clock drops 2040 -> ~1680-1815 MHz. PyTorch never reaches the cap
           (58.8 W max) and holds 2040 MHz.

Decision: keep warmup_iterations = 100 for the controlled runs. It exceeds the
convergence point observed for both backends by a wide margin, and it was the
pre-registered default. No config was edited after seeing the data.

A better-specified stationarity rule is Phase 5B work; it must be registered before
new data is collected.
