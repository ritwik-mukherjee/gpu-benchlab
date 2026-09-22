# GPU validation runbook (Phase 5A)

> **This is a procedure, not a result.** It was written on 2026-09-19 on a machine with
> no NVIDIA GPU. **Nothing in it has run on NVIDIA hardware**, and it contains no
> measurements. What following it produces goes to `results/`, never into this file.

**Purpose:** before TensorRT, check that the existing infrastructure behaves correctly on
the first real NVIDIA machine. This is validation, not performance work.

The procedures live in [`analysis/gpu_validation/`](../../analysis/gpu_validation/).
Each script checks its pre-registered criteria (listed in its docstring) and marks
every check `PASS` / `FAIL` / `INCONCLUSIVE` / `INFO`. It writes all raw values to a
JSON file and exits 1 on any `FAIL`. **On the development machine these scripts have
only been run where no GPU is needed:**
- the GPU scripts refuse to run (exit 2, or `FAIL` "no CUDA EP / no nvidia-smi");
- `trajectory.py` was run on stored CPU runs, and its maths hand-checked;
- `telemetry_summary.py` was run on a synthetic CSV, as a format test only;
- all seven configs end `unavailable` in `validate` with 0 samples.

## 0. Status of each area

| Area | Implemented | Validated on CPU | Must be run on NVIDIA hardware | § |
|---|---|---|---|---|
| NVML detection | Success path, field extraction | Absence path only (no driver) | Success path, every field vs `nvidia-smi` | 3 |
| PyTorch CUDA backend | Device checks, IEEE-FP32 flags, sanity pass, no fallback | The CUDA request is refused (`unavailable` in `validate`) | Device identity, placement, sanity pass | 4 |
| `CudaEventTimer` | Event pair on the current stream; host time kept as a secondary series | Against a fake event API only | Real asynchronous GPU work | 5 |
| Warmup evidence | Every warmup sample is stored | CPU trajectories (non-stationary) | GPU trajectory | 6 |
| Correctness criterion | `compare_outputs`, FP16 negative control | PyTorch CPU vs ORT CPU: 9/9 | CPU vs CUDA; IEEE FP32 and TF32 | 7 |
| ORT CUDA EP | Provider options, provider check after creation, placement probe, IOBinding | Refusal with the real onnxruntime-gpu when CUDA libraries are missing | EP loads, placement, IOBinding, sync at the end of `Run` | 8 |
| ORT CUDA correctness | `gpu-bench onnx verify --device cuda:N` | CPU EP only | CUDA EP | 9 |
| Telemetry | One NVML snapshot per result; no sampling during a run | — | Sampling during a run (external `nvidia-smi`) | 10 |
| Controlled benchmark | Engine, schema, raw samples | — | Everything | 11 |

**Rules for the session:**
- **A `FAIL` stops the runbook** at that section. Do not continue on a failed prerequisite.
- **An `INCONCLUSIVE` needs a written explanation** in `$E/NOTES.md` before you continue.
- **Thresholds are pre-registered.** Never edit one after seeing GPU data. If a threshold
  proves wrong, report the result under the original and propose the change separately.
- **Tolerances are never loosened** to make correctness pass.
- **Never delete or edit raw evidence**, including anomalous samples.
- **Do not start TensorRT.**

## Before the session: open decisions

These are known now, from the code, and should be settled before paying for GPU time.

| # | Issue | Effect | Recommendation |
|---|---|---|---|
| D1 | The PyTorch backend makes inputs with `torch.randn`; ORT uses numpy PCG64 (`core/inputs.synthetic_input`). | §11's configs differ in input values. | Make the PyTorch backend use `synthetic_input`: a small change, testable on CPU. Otherwise §11 must report the difference as uncontrolled. |
| D2 | `GPUInfo.multiprocessor_count` comes from `nvmlDeviceGetNumGpuCores`, which NVML describes as the device's "core count". | The field may hold CUDA cores rather than SMs. | Leave it as is. §3 compares it with PyTorch's SM count; fix it after that evidence. |
| D3 | `docs/environment.md` suggested the `cu128` index, where torch stops at 2.11.0 (checked 2026-09-19). Phases 3–4 used 2.14.0. | Following it would change the torch version. | This runbook uses `cu130` (torch 2.14.0). `environment.md` has been corrected. |
| D4 | `gpu-bench run` imports torch during environment detection, before ORT creates its session. torch loads its own CUDA/cuDNN libraries. | ORT's CUDA EP might load only *because* torch was imported first. | §8 runs its check with and without torch imported first. |
| D5 | No test is marked `@pytest.mark.gpu`; every CUDA test uses fakes. | A green test suite on the GPU box is **not** GPU validation. | The evidence comes from this runbook. |

## Machine requirements

- **OS and GPU:** Linux x86_64 with exactly one visible NVIDIA GPU, MIG disabled, and no other compute processes.
- **Driver ≥ R580.** CUDA 13.0 needs R580 on Linux and Windows ([CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)). torch 2.14.0+cu130 and onnxruntime-gpu 1.30 (CUDA 13.0, cuDNN 9; [ORT CUDA EP docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)) both need it.
- **Compute capability:**
  - **≥ 7.5** at minimum. The CUDA 13.0 release notes drop Maxwell, Pascal and Volta.
  - **≥ 8.0** for the TF32 steps. Otherwise TF32 is recorded as not applicable.
- **Software:** Python 3.12, `uv`, `git`.
- **Do not upgrade the driver or lock clocks as part of this runbook.** Both change system settings. If the driver is too old, choose another machine image.

Session setup, used by every section:

```bash
git clone <your private remote or phase4.bundle> gpu-benchlab && cd gpu-benchlab
git checkout <commit>                      # the exact commit under test
export E=results/phase5a-raw               # git-ignored working evidence set
export CUDA_DEVICE_ORDER=PCI_BUS_ID        # CUDA enumerates GPUs in NVML's (PCI) order
mkdir -p $E/{runs,telemetry,logs,analysis}
```

## 1. Environment information

**Implemented:** `gpu-bench hardware --json` and `analysis/gpu_validation/collect_env.sh`,
which captures everything below read-only, one file per command, with exit codes.

```bash
bash analysis/gpu_validation/collect_env.sh $E/env-before   # BEFORE installing anything
# ... §2 installation ...
bash analysis/gpu_validation/collect_env.sh $E/env-after
```

| Item | Primary source | Cross-check |
|---|---|---|
| GPU model, count | `nvidia-smi -L`, NVML name | `torch.cuda.get_device_name` |
| VRAM | `memory.total`, NVML | `torch…get_device_properties().total_memory` |
| Compute capability | `compute_cap`, NVML | `torch.cuda.get_device_capability` |
| Driver | `nvidia-smi` header, NVML `driver_version` | `nvidia-smi -q` |
| CUDA the driver supports (max) | `nvidia-smi` header "CUDA Version", NVML | — |
| CUDA runtime actually used | `torch.version.cuda`; ORT: the `libcudart` it loads (§8 check O3) | wheel list in `pip-freeze.txt` |
| cuDNN | `torch.backends.cudnn.version()`; ORT: the `libcudnn` it loads (§8 O3) | `pip-freeze.txt` |
| Clocks (max), power limit, persistence, ECC, MIG, PCIe | `nvidia-smi-static.txt`, `nvidia-smi -q` | NVML (§3) |
| Temperature, utilization (idle) | `nvidia-smi -q`, NVML | §3 |
| OS, kernel, virtualization | `os-release`, `uname -a`, `systemd-detect-virt` | — |
| Python, PyTorch, ONNX Runtime (+ build info) | `python -VV`, `frameworks.json` | `gpu-bench doctor` |
| TensorRT | `gpu-bench hardware` (availability **only**, nothing is installed) | — |

**Don't assume CUDA is installed because a driver exists.**
- The driver reports only the *highest* CUDA version it supports. The CUDA runtime actually used comes from the pip wheels.
- A missing `nvcc` is expected and gets recorded; it is not an error.

**Success:** both captures are complete, and every missing item is explained in `NOTES.md`.

## 2. Installation

Gates, checked against `env-before` **before** installing anything:

| Gate | Check | If it fails |
|---|---|---|
| G1 | Driver ≥ 580 and the `nvidia-smi` header's CUDA Version ≥ 13.0 | Stop and use another image. torch cu126 would run, but ORT 1.30's CUDA EP would not, so §8, §9 and §11 would be impossible. |
| G2 | `compute_cap` ≥ 7.5 | Stop. |
| G3 | One GPU, MIG disabled, `nvidia-smi` lists no processes | Stop, or record why it's acceptable. |

Install. Dry-run each step first and read the plan before committing to it.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
# PyTorch: the same versions as Phases 3-4, built for CUDA 13.0 (2.14.0 is not on cu128).
uv pip install --dry-run torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install           torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
# ONNX Runtime GPU and export tooling. NOT onnxruntime-gpu[cuda,cudnn]: torch's CUDA 13
# wheels already provide cuDNN 9 and cuBLAS, so the venv keeps a single copy of each.
uv pip install --dry-run -e ".[onnx,onnx-export]"
uv pip install           -e ".[onnx,onnx-export]"
```

Checks after installing. Record the output in `NOTES.md`.

| Check | How | Required |
|---|---|---|
| I1 | The dry-run did not change torch or torchvision | yes |
| I2 | `uv pip list \| grep -i onnxruntime` shows only `onnxruntime-gpu 1.30.0`. The CPU package `onnxruntime` conflicts with it. | yes |
| I3 | `uv pip list \| grep -i cudnn` shows one cuDNN 9 package | yes |
| I4 | `frameworks.json`: torch `2.14.0…`, `cuda_build` 13.0.x, `is_available: true`, and this GPU's `sm_XY` (or a compatible `compute_XY`) in `arch_list` | yes |
| I5 | `gpu-bench doctor`: torch CUDA usable = yes | yes |
| I6 | `CUDAExecutionProvider` is in ORT's available providers. **This is necessary, not sufficient:** Phase 4 saw it listed with no working EP. §8 proves execution. | yes |

Self-test of the harness. This is not GPU validation (see D5):

```bash
ruff check . && ruff format --check . && mypy && pytest -q -rs 2>&1 | tee $E/logs/pytest.log
```

Expected: everything passes, and the tests for the no-CUDA path skip, each printing its skip reason. Any other skip or failure is a finding to record; don't weaken the tests to make them pass.

## 3. NVML validation

**Implemented:** the NVML success path (`hardware/nvml.py`). Until now it has only been tested against a fake NVML module.

```bash
python analysis/gpu_validation/nvml_vs_smi.py $E/nvml-idle            # GPU idle
# Again while a §6 run is executing in another shell (dynamic fields under load):
python analysis/gpu_validation/nvml_vs_smi.py $E/nvml-load
```

The script reads `nvidia-smi` just before and just after gpu_benchlab's detection, and compares 19 fields per GPU, plus GPU count, driver version and the maximum CUDA version the driver supports.

**Evidence:** `nvml_vs_smi.json`, holding both readings and the full `EnvironmentReport`.

**Success:**
- **Static fields match exactly** after unit conversion: name, UUID, PCI bus ID, serial, compute capability, total memory, enforced power limit, max clocks, persistence mode, compute mode.
- **Unavailable values stay null.** It is a `FAIL` if the tool reports null where `nvidia-smi` has a value, or 0 where `nvidia-smi` says `[N/A]`.
- **Dynamic fields** (temperature, utilization, power draw, clocks, memory used/free) lie between the two `nvidia-smi` readings, widened by the tolerance in the script. Anything outside is `INCONCLUSIVE` and needs a written explanation. Candidates:
  - NVML memory v1 vs v2 accounting for reserved memory;
  - power draw averaged vs instantaneous;
  - clocks changing between readings.
- **SM count** (`torch{i}.sm_count`) equals PyTorch's `multi_processor_count`. A mismatch confirms D2 and is recorded as a bug to fix. The check itself is not changed.

## 4. PyTorch CUDA validation

**Implemented:** `backends/pytorch.py`. It checks the CUDA build, availability, index and capability → precision support, and it sets conv and matmul `fp32_precision="ieee"`. It runs a sanity pass (shape, dtype, finite outputs) and never falls back to CPU.

```bash
gpu-bench models fetch resnet50                                 # pinned IMAGENET1K_V2, SHA-256 verified
python analysis/gpu_validation/pytorch_cuda_check.py $E/pytorch 2>&1 | tee $E/logs/pytorch_cuda_check.log
```

This drives the **real** backend lifecycle (validate → load → prepare → execute) for ResNet-50, batch 1, IEEE FP32, on `cuda:0`, then inspects what the backend built.

**Success:** P1–P11 and N1 `PASS`. P12 may be `INCONCLUSIVE` in a container.
- P1 `torch.cuda.is_available()`.
- P2 The backend's device is the NVML GPU with the same UUID.
- P3 The recorded capability equals both PyTorch's and NVML's.
- P4 Every parameter and buffer is on `cuda:0`.
- P5 Parameters are float32.
- P6 The input is on `cuda:0`.
- P7 The output is on `cuda:0`: shape (1, 1000), float32, finite.
- P8 `fp32_precision` is `ieee` for `cudnn.conv` and `cuda.matmul`, and recorded.
- P9 The sanity pass is recorded.
- P10 Allocated CUDA memory covers the weights.
- P11 The timer's stream is the execution stream.
- P12 This process appears in `nvidia-smi --query-compute-apps`.
- N1 With `CUDA_VISIBLE_DEVICES=""`, `gpu-bench run` stores `unavailable`/`failed` with **0 samples**. It never runs on CPU.

Headline numbers are not collected until §4 and §5 pass.

## 5. CUDA-event validation

**Implemented:** `CudaEventTimer` in `backends/pytorch.py`. It records a primary series of event times and a secondary series of host times after a synchronize.

Where synchronization happens. The engine itself never synchronizes (`core/timing.py`).

| Backend | Point | Call |
|---|---|---|
| PyTorch | `timer.start()` | `torch.cuda.synchronize(device)` drains the previous iteration → host `t0` → `start_event.record(stream)` |
| PyTorch | `execute` | `model(inputs)` queues kernels and returns early |
| PyTorch | `timer.stop()` | `end_event.record(stream)` → `end_event.synchronize()` (the host blocks until this iteration is done) → host `t1` |
| PyTorch | result | primary = `start_event.elapsed_time(end_event)`; secondary = `t1 − t0` |
| PyTorch | `prepare` (untimed) | `synchronize()` after the sanity pass |
| ORT | `timer.start()` / `stop()` | host clock only; no sync hook |
| ORT | `execute` | `run_with_iobinding` returns once ORT has synchronized its stream. This is assumed, and verified in §8 check S1. |

**The experiment** is part of `pytorch_cuda_check.py`, run in §4. Its workload is `torch.cuda._sleep`, a private PyTorch helper that keeps the GPU busy for a fixed number of clock cycles at near-zero launch cost. A chain of matmuls is used if it is absent. The amount is calibrated to ≈5 ms, then repeated at 1×, 2× and 4× the work, 30 trials each. Every trial records three times:
- **launch time:** from `h0` until the launch call returns, with no sync;
- **host time:** from `h0` until the host clock stops after a synchronize;
- **event time.**

**Success (pre-registered):**

| Check | Criterion | What it proves |
|---|---|---|
| E1 | median launch time < 5 % of the event time (4×) | the GPU work really is asynchronous |
| E2 | event time for 2×/1× ∈ [1.9, 2.1] and 4×/1× ∈ [3.8, 4.2] | event time follows GPU work |
| E3 | host time ≥ event time in **every** trial | the host interval contains the device interval |
| E4 | coefficient of variation (CV) of the 4× event time < 2 % | the measurement is stable |
| E5 | start event, then a 20 ms host sleep, then work: event time ≥ 20 ms | idle stream time between events is counted (the documented meaning) |
| E6 | `CudaEventTimer`'s median is within 2 % of the E1 reference, and every secondary sample ≥ its primary | **the real timer class** is correct |
| E7 | an unsynchronized `WallClockTimer` reads < 5 % of the event time | control: the experiment would catch a timer that measures only the launch |

**If E6 fails, the timer is wrong.** Stop, fix it (with a test that reproduces the failure), and re-run §4–§5 before continuing.

## 6. Warmup and stability

**Implemented:** the engine stores every warmup sample (`raw.json: warmup_latency_ms`), CUDA secondary samples, and time spent in each phase (`result.json: phases`).

Where initialization cost *should* appear. These are expectations to check, not facts:
- **CUDA initialization:** at the first CUDA call. That may be the capability query in `validate`, which is untimed, or `.to(device)` in `load` (`phases.model_load_ms`). The current phase timings can't separate CUDA context creation from weight upload; that is a known limitation.
- **cuDNN setup and `cudnn.benchmark` autotuning:** expected at the first forward pass with this input shape. That is the untimed sanity pass in `prepare` (`phases.prepare_inputs_ms`), not warmup.
  - `PyTorchOptions.cudnn_benchmark`'s description says "during warmup", which contradicts this.
  - §6 settles which is right: compare `prepare_inputs_ms` and the first warmup samples between the `cudnnbench` and `nocudnnbench` runs.
- **ORT:** session creation is recorded in `phases.engine_build_ms`. The EXHAUSTIVE convolution-algorithm search is expected on the first `Run` (the sanity run in `prepare`); check it the same way.

Runs: three configs × 3 repeats, 300 warmup + 1000 measured iterations each. Each runs as its own process, with telemetry (§10).

```bash
TELEMETRY_FIELDS=timestamp,index,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,enforced.power.limit,clocks.sm,clocks.mem,clocks.max.sm,clocks_event_reasons.active
# (older drivers: clocks_throttle_reasons.active -- check env-after/nvidia-smi-help-query-gpu.txt)
run_logged() {   # usage: run_logged <config.yaml> <label>
  nvidia-smi --query-gpu=$TELEMETRY_FIELDS --format=csv,nounits -lms 100 > "$E/telemetry/$2.csv" &
  local smi=$!; sleep 2
  gpu-bench run -c "$1" --results-dir "$E/runs/$2" 2>&1 | tee "$E/logs/$2.log"
  sleep 2; kill "$smi"
}
gpu-bench onnx export resnet50 2>&1 | tee $E/logs/onnx-export.log   # needed by the ORT config
for rep in 1 2 3; do
  for c in warmup-pytorch-cuda-fp32-b1-cudnnbench warmup-pytorch-cuda-fp32-b1-nocudnnbench warmup-onnxruntime-cuda-fp32-b1; do
    run_logged analysis/gpu_validation/configs/$c.yaml $c-r$rep
  done
done
python analysis/gpu_validation/trajectory.py --json $E/analysis/warmup.json $E/runs/warmup-*/*/
```

**Deciding whether 10 warmups are enough** (definitions in `trajectory.py`, fixed in advance):
- m* is the median of the last 500 measured samples.
- The trajectory (warmup + measured) is split into 10-iteration blocks.
- The **settle iteration k** is the first block from which *every* later block median stays within ±2 % of m*.
- **10 warmups are enough for a configuration only if k ≤ 10 in all 3 repeats.** Otherwise record k and use it in §11.
- Also record:
  - `first_timed_over_m_star`;
  - drift (second half vs first half; > 2 % means UNSTABLE);
  - `prepare_inputs_ms` with and without cuDNN autotuning;
  - containment (host ≥ event for every iteration).

A run that never settles, or that drifts, is a finding. Investigate it with telemetry (clocks, event reasons, temperature) before §11. Don't tune the definition until the run passes.

## 7. PyTorch CPU vs CUDA correctness

**Implemented:** the criterion (`core/correctness.py`), deterministic inputs (`core/inputs.py`) and the FP16 negative control. On CUDA, none of these have run.

```bash
python analysis/gpu_validation/correctness_gpu.py $E/correctness 2>&1 | tee $E/logs/correctness_gpu.log
```

**Setup:**
- The criterion is the Phase 4 one, **unchanged**: `|cand − ref| ≤ 1e-4·max|ref| + 1e-4·|ref|` elementwise, plus identical top-1 (ADR 0007).
- Inputs: numpy PCG64 float32, batch sizes 1/4/8 × seeds 0/1/2.
- Reference: PyTorch FP32 on CPU, with pinned weights. The weights' SHA-256 must equal the ONNX artifact's (W1).
- PyTorch CUDA runs with `cudnn.benchmark` on, as the benchmark does.

**Success:**
- **W1** The weights are identical.
- **C1** CPU vs CUDA in IEEE mode: 9/9 pass.
- **NC** The FP16 control on CUDA is **rejected**.

**TF32** (only on SM ≥ 8.0; nothing is claimed until it is observed):
- **T1** records whether TF32 output differs bitwise from IEEE output.
- **C2** records whether TF32 passes the FP32 tolerance.
- They are interpreted as follows, all pre-registered:

| Observation | Meaning |
|---|---|
| T1 differs, C2 fails | The tolerance detects TF32 on this GPU. This supports the Phase 4 FP16 proxy. |
| T1 differs, C2 passes | TF32 engaged but stays within the FP32 tolerance for ResNet-50. The tolerance cannot tell TF32 from IEEE here, so FP32 claims rest on the recorded flags alone. |
| T1 identical | TF32 had no observable effect. No TF32 claim is made. |
| C1 fails | **Stop.** Don't loosen the tolerance. As a diagnostic only, re-run with `cudnn.benchmark` off, and record both runs. |

R1 (same-process repeatability) is recorded for information only.

## 8. ONNX Runtime CUDA validation

**Implemented:** `backends/ort_backend.py`. It requests exactly one EP, passes `use_tf32="0"`, checks `get_providers()` after creation (the result is `unavailable` if ORT substituted another EP), measures node placement via profiling, and uses IOBinding with input and output on the device. The script below checks all of this **independently**, with the raw ORT API and identical settings.

```bash
python analysis/gpu_validation/ort_cuda_check.py $E/ort --batch 8 --import-torch-first
python analysis/gpu_validation/ort_cuda_check.py $E/ort --batch 8          # ORT alone (D4)
# if S2 is INCONCLUSIVE: repeat both with --batch 32
gpu-bench run -c examples/resnet50-onnxruntime-cuda-fp32.yaml --results-dir $E/runs/ort-example
python -c "import json,glob; s=json.load(open(glob.glob('$E/runs/ort-example/*/result.json')[0]))['backend']['settings']; print({k:v for k,v in s.items() if k.startswith(('active_','node_placement','io_binding','available_'))})"
```

**Success (torch-first run):**

| Check | Criterion |
|---|---|
| O1 | The requested provider is the **first** active one: `get_providers()[0] == CUDAExecutionProvider` |
| O2 | Effective `use_tf32 == "0"`. ORT's default is 1. |
| O3 | `libonnxruntime_providers_cuda`, `libcudnn` and `libcublas` are loaded into the process (`/proc/self/maps`). Their paths are recorded. |
| O4 | Profiling shows every executed node on the CUDA EP. Any other EP's nodes are listed, and the profile JSON is kept. |
| O5 | The IOBinding input `OrtValue` and the bound output are both on `cuda` |
| O6 | Output is (B, 1000), float32, finite |
| O7 | This process appears in `nvidia-smi` compute apps. `INCONCLUSIVE` is allowed in containers. |
| S1 | `median(run + device sync) − median(run) ≤ max(0.05 ms, 2 %)`, i.e. `run_with_iobinding` returns only after the GPU has finished. This is what makes the backend's host clock valid. |
| S2 | The control has power: with `disable_synchronize_execution_providers=1`, the median `run` is < 80 % of the default. Otherwise re-run at batch 32. |
| N1 | With `CUDA_VISIBLE_DEVICES=""`, the result is `unavailable`/`failed` with 0 samples. It is never a silent CPU session. |
| gpu-bench | `active_providers` starts with the CUDA EP; `node_placement.CUDAExecutionProvider` equals O4's count, with no other `node_placement.*`; `io_binding: true`; `active_provider_option.use_tf32: "0"` |

**The ORT-alone run is diagnostic.**
- If it passes, ORT can load CUDA on its own.
- If O1 or O3 fails there while the torch-first run passes, record it as a **finding**: `gpu-bench` currently works only because torch loaded the libraries first. Recommend explicit library loading (`onnxruntime.preload_dlls()`) for Phase 5B. It does not block §9.

## 9. PyTorch CUDA vs ONNX Runtime CUDA correctness

```bash
gpu-bench onnx verify resnet50 --device cuda:0 --results-dir $E/correctness-verify
python analysis/rederive_correctness.py $E/correctness-verify/correctness-*/     # independent re-derivation
```

Also produced by `correctness_gpu.py` (§7):
- **C3:** PyTorch CUDA (IEEE) vs ORT CUDA (`use_tf32=0`);
- **C4:** CPU reference vs ORT CUDA;
- **C5:** ORT with `use_tf32=1`, observed only.

**Success:**
- **Correctness:** C3 and C4 pass 9/9, and the `onnx verify` report passes (9/9, FP16 control rejected). Output shape, dtype, max absolute error, max relative error, worst violation ratio and top-1/top-5 agreement are all recorded.
- **Re-derivation** reports 0 disagreements.
- **Same inputs and weights:** input generator and weights SHA-256 are identical across all runtimes.
- **Artifact:** record its SHA-256 and compare it with the one exported on the development machine (`results/published/2026-09-19-phase4-onnx/`). Identical bytes across machines is informative, not required.

**Comparison with Phase 4 CPU** (`results/published/2026-09-19-phase4-onnx/correctness-1d8f04cba5b5/`): report the max error and worst violation ratio from both, side by side. No verdict depends on how they compare.

## 10. GPU telemetry

**Implemented:** one NVML snapshot per result, taken at run start. **Not implemented:** sampling during a run (Phase 7). The benchmark loop must not poll, so the sampler here is `nvidia-smi` in a separate process (`run_logged`, §6), sampling every 100 ms.

```bash
for f in $E/telemetry/*.csv; do
  python analysis/gpu_validation/telemetry_summary.py "$f" 100 > "$E/analysis/$(basename "$f" .csv).telemetry.json"
done
```

Fields recorded:
- utilization (GPU and memory controller) and `pstate`;
- memory used and total;
- temperature;
- power draw and enforced limit;
- SM and memory clocks, with max SM clock;
- clock event reasons, decoded with the installed `pynvml` constants.

**The sampling is adequate for run-level context if** (pre-registered):
- the median achieved interval is ≤ 2× the requested one;
- there are ≥ 10 samples per run;
- no field is entirely `[N/A]`.

**What telemetry can't show:**
- **Per-iteration effects** when an iteration is shorter than the sampling interval.
- **How busy the GPU is.** NVML utilization is the fraction of time *any* kernel was running during its own sample window, not SM occupancy. So "100 %" does not mean the GPU was saturated.

**Worth flagging:**
- **Thermal clock slowdowns** mark a run UNSTABLE.
- **A power cap** is recorded as a run condition, but doesn't invalidate the run on its own.
- **Memory growth** across a run, and SM clock below max during measurement, are also noted.

## 11. Small controlled benchmark

Only after §3–§10 have no `FAIL`, with every `INCONCLUSIVE` explained. This is a **validation experiment**; it is not a ranking.

**Common configuration** (`analysis/gpu_validation/configs/controlled-*.yaml`):

| Dimension | Value |
|---|---|
| Model / weights | ResNet-50, IMAGENET1K_V2 (SHA-256 verified); ORT artifact opset 20, dynamic batch (SHA-256 recorded) |
| Inputs | Seed 0, float32 standard normal. **The generator differs unless D1 is fixed.** |
| Batch / shape | 1 and 8 × 3 × 224 × 224 |
| Precision | IEEE FP32: PyTorch `fp32_precision="ieee"`, ORT `use_tf32="0"` |
| Autotuning | PyTorch `cudnn_benchmark: true`; ORT `cudnn_conv_algo_search: EXHAUSTIVE`. Both happen in the untimed sanity pass. |
| Memory format | NCHW (`channels_last: false`) |
| Data movement | Input and output stay on the device for both: PyTorch tensors on `cuda:0`, ORT IOBinding. No host↔device copy is timed. |
| Warmup / measured | 100 / 1000. If §6 found any settle k > 50, set warmup to 2k in **all four** configs and commit that change before the first run. |
| Repetitions | 5 per (backend, batch), each in its own process, in alternating order (ABBA) |
| Threads | Defaults, recorded (`num_threads`, ORT intra/inter-op) |
| Clocks | Not locked (a system setting). Recorded through telemetry. |
| Git | The tree must be clean (`git_dirty: false`) |

```bash
for rep in 1 2 3 4 5; do
  for b in 1 8; do
    if [ $((rep % 2)) -eq 1 ]; then order="pytorch onnxruntime"; else order="onnxruntime pytorch"; fi
    for be in $order; do
      run_logged analysis/gpu_validation/configs/controlled-$be-cuda-fp32-b$b.yaml ctl-$be-b$b-r$rep
    done
  done
done
python analysis/gpu_validation/trajectory.py --json $E/analysis/controlled.json $E/runs/ctl-*/*/
```

**The two backends' timing methods differ.**
- **PyTorch's primary series** is CUDA-event time: device time only.
- **ORT's series** is host wall-clock time around `run_with_iobinding` (made valid by §8 S1).
- **For a like-for-like comparison, use host-side time for both:** PyTorch's secondary series against ORT's primary one. Report PyTorch's event time separately, and never compare it with ORT's host time.

**The numbers are trustworthy only if every one of these holds for each (backend, batch)** (pre-registered):
1. All 5 runs are `ok`, with no `ANOMALY` note.
2. The spread of per-run medians is ≤ 5 %.
3. No run drifts by more than 2 %.
4. For PyTorch, 0 containment violations.
5. §6's settle iteration is ≤ the configured warmup.
6. No thermal slowdown during measurement.

If any condition fails, report the numbers as **unstable** and investigate before interpreting them.

**How to report:** for each (backend, batch), give:
- the 5 per-run medians, their median and their spread;
- p90/p99 per run, not pooled across runs;
- the timing series used;
- telemetry during the run (clocks, power, temperature, event reasons);
- hardware, driver, CUDA/cuDNN runtime, software versions, the configuration, the git commit, and the limitations.

**What can be concluded:** on *this* GPU, driver and software stack, for ResNet-50 at these two batch sizes in IEEE FP32 with device-resident I/O, the stable per-run medians and their spread are as measured. A difference between backends means something only if it exceeds both backends' spread.

**What cannot be concluded:**
- that either backend is generally faster;
- anything about other batch sizes, GPUs, precisions, models or concurrency;
- anything about throughput under load, or production latency;
- anything about TensorRT;
- any comparison with a CPU result from any phase.

## 12. Evidence to preserve

Everything goes under `$E` during the session and is never edited afterwards:

```text
results/phase5a-raw/
  env-before/  env-after/        collect_env.sh: nvidia-smi (-q), pip freeze, git commit + status, environment.json
  nvml-idle/  nvml-load/         nvml_vs_smi.json
  pytorch/                       pytorch_cuda_check.json (+ negative-no-cuda/ result)
  correctness/                   correctness_gpu.json + outputs.npz
  correctness-verify/            gpu-bench onnx verify report.json + outputs.npz
  ort/b8-torch-first/ ...        ort_cuda_check.json + ORT profile JSON (+ negative-no-cuda/)
  runs/<label>/<experiment-id>/  result.json, raw.json, metadata.json, summary.json, logs/
  telemetry/<label>.csv          nvidia-smi samples for each run
  logs/                          console output (tee), pytest log, export log
  analysis/                      trajectory / telemetry summaries
  NOTES.md                       operator notes: every INCONCLUSIVE, deviation and decision, timestamped
```

**Checklist:**
- **Git state is recorded** (commit plus `git status`) in `env-*`, and every result carries `provenance`.
- **Software versions:** `pip-freeze.txt`, `frameworks.json`, and each result's `environment`.
- **Configs:** keep the exact YAML used, in `analysis/gpu_validation/configs/` at the recorded commit, and in each result's `configuration`.
- **Raw samples:** keep every `raw.json` (warmup included), and never drop anomalous samples.
- **Keep failures.** A `FAIL` or `unavailable` result is evidence too.
- **Publishing:**
  1. Copy the whole set, unmodified, to `results/published/<YYYY-MM-DD>-phase5a-<gpu>/` with a README stating what it is and is not, as Phase 4 did.
  2. Decide on identifiers *before* publishing. Hostname, GPU UUID and serial, PCI bus ID and cloud instance metadata appear in the environment captures. Either publish a file unmodified or leave it out; never edit it.
  3. Commit on the GPU machine and bring the commits back with `git bundle`.
- **Never commit** model weights, ONNX artifacts or anything secret.

## Definition of success

Phase 5A counts as **validated on NVIDIA hardware** only if:
- §1–§10 have no `FAIL`, every `INCONCLUSIVE` is explained in `NOTES.md`, and the evidence is preserved as §12 describes;
- the docs (`limitations.md` "validated on NVIDIA GPU" column, `environment.md`, methodology, ENGINEERING_LOG, CHANGELOG) are then updated from that evidence alone.

§11's numbers are trustworthy only if its stability conditions hold; otherwise they are reported as unstable. Anything that was not run stays **not yet validated**.
