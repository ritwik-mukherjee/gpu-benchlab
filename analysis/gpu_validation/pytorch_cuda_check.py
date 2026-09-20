"""Phase 5A runbook §4 + §5: the PyTorch CUDA backend and CudaEventTimer on real hardware.

§4 drives the REAL PyTorchBackend lifecycle (validate -> load -> prepare -> execute)
and inspects what it built: device identity, where the weights, input and output
live, the precision flags, the sanity pass, and the stream the timer records on.

§5 checks that CUDA-event timing measures GPU execution, not CPU launch time.
Workload: torch.cuda._sleep(cycles) -- a private PyTorch helper that spins the GPU
for a fixed number of clock cycles with negligible launch cost (a matmul chain is
used if it is absent). Per trial, all on the current stream:

    synchronize                         <- nothing outstanding
    h0 = host clock; start_event.record()
    launch work                         <- returns immediately (asynchronous)
    h_launch = host clock               <- launch cost only
    end_event.record(); synchronize     <- host blocks until the GPU is done
    h1 = host clock
    event_ms = start_event.elapsed_time(end_event)

Pre-registered checks (thresholds fixed before any GPU data):
  E1 async        median launch_ms < 5% of median event_ms (4x work)
  E2 linearity    event 2x/1x in [1.9, 2.1] and 4x/1x in [3.8, 4.2]
  E3 containment  host (h1-h0) >= event_ms in every trial (host interval contains device)
  E4 stability    coefficient of variation of event_ms (4x) < 2%
  E5 idle gap     start event, host sleeps 20 ms, then work: event_ms >= 20 ms
                  (documents that idle stream time between the events is counted)
  E6 real timer   CudaEventTimer primary within 2% of E1's event median (4x), and its
                  secondary host samples >= primary in every trial
  E7 control      an UNSYNCHRONIZED WallClockTimer reads < 5% of the event time: proof
                  the experiment would catch a timer that measures only the launch
  N1 no fallback  gpu-bench run with CUDA hidden (CUDA_VISIBLE_DEVICES="") ends
                  unavailable/failed with zero samples -- never a CPU run

Usage: python analysis/gpu_validation/pytorch_cuda_check.py <out_dir>
"""

from __future__ import annotations

import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from _common import INCONCLUSIVE, INFO, Checks, compute_app_pids, require_cuda_torch

CONFIG = {
    "name": "phase5a-pytorch-cuda-check",
    "backend": "pytorch",
    "device": "cuda:0",
    "precision": "fp32",
    "batch_size": 1,
    "model": {"name": "resnet50", "weights": "IMAGENET1K_V2"},
    "backend_options": {"cudnn_benchmark": True, "channels_last": False},
    "benchmark": {"warmup_iterations": 10, "measurement_iterations": 100},
}
TRIALS = 30


def section4(torch: Any, checks: Checks) -> dict[str, Any]:
    from gpu_benchlab.backends.pytorch import PyTorchBackend
    from gpu_benchlab.core.config import parse_config
    from gpu_benchlab.hardware.detect import detect_environment

    env = detect_environment(include_frameworks=False)
    cfg = parse_config(CONFIG)
    backend = PyTorchBackend(cfg)
    device = torch.device("cuda:0")
    try:
        backend.validate(cfg, env)
        backend.load()
        backend.build()
        inputs = backend.prepare(cfg)
        timer = backend.make_timer()
        with backend.execution_context():
            output = backend.execute(inputs)
        backend.synchronize()
        settings = dict(backend.descriptor.settings)
        model = backend._model  # read-only inspection of what the backend built
        tensors = [*model.parameters(), *model.buffers()]
        props = torch.cuda.get_device_properties(0)
        nvml_gpu = next(
            (g for g in env.gpus if g.uuid and str(getattr(props, "uuid", "?")) in g.uuid), None
        )

        checks.expect("P1", torch.cuda.is_available(), "torch.cuda.is_available()")
        checks.expect(
            "P2",
            nvml_gpu is not None and settings.get("cuda_device_name") == props.name,
            "backend's device is the NVML GPU with the same UUID",
            torch_name=props.name,
            torch_uuid=str(getattr(props, "uuid", None)),
            nvml=nvml_gpu.model_dump(mode="json") if nvml_gpu else None,
        )
        checks.expect(
            "P3",
            settings.get("cuda_capability") == f"{props.major}.{props.minor}"
            and (
                nvml_gpu is None or nvml_gpu.compute_capability == settings.get("cuda_capability")
            ),
            "recorded compute capability matches torch and NVML",
            recorded=settings.get("cuda_capability"),
        )
        checks.expect(
            "P4",
            all(t.device == device for t in tensors),
            "every parameter and buffer is on cuda:0",
            devices=sorted({str(t.device) for t in tensors}),
            count=len(tensors),
        )
        checks.expect(
            "P5",
            all(t.dtype == torch.float32 for t in model.parameters()),
            "parameters are float32",
        )
        checks.expect(
            "P6", inputs.device == device, "input is on cuda:0", device=str(inputs.device)
        )
        checks.expect(
            "P7",
            output.device == device
            and tuple(output.shape) == (1, 1000)
            and output.dtype == torch.float32
            and bool(torch.isfinite(output).all()),
            "output is on cuda:0, shape (1, 1000), float32, finite",
            device=str(output.device),
            shape=list(output.shape),
            dtype=str(output.dtype),
        )
        checks.expect(
            "P8",
            settings.get("fp32_precision.cudnn.conv") == "ieee"
            and settings.get("fp32_precision.cuda.matmul") == "ieee",
            "IEEE FP32 is applied and recorded (TF32 off)",
            **{k: v for k, v in settings.items() if "precision" in k or "tf32" in k},
        )
        checks.expect("P9", settings.get("sanity_check") == "passed", "sanity pass recorded")
        weight_bytes = sum(t.numel() * t.element_size() for t in tensors)
        checks.expect(
            "P10",
            torch.cuda.memory_allocated(device) >= weight_bytes,
            "CUDA memory allocated covers the weights",
            allocated=torch.cuda.memory_allocated(device),
            weights=weight_bytes,
        )
        stream = torch.cuda.current_stream(device)
        checks.expect(
            "P11",
            timer._stream == stream,  # the timer records on the stream kernels use
            "CudaEventTimer records on the current (execution) stream",
            timer_stream=timer._stream.cuda_stream,
            current_stream=stream.cuda_stream,
        )
        pids = compute_app_pids()
        if pids is None or os.getpid() not in pids:
            checks.add(
                "P12",
                INCONCLUSIVE,
                "this PID in nvidia-smi compute apps (not visible in "
                "containers with a separate PID namespace)",
                pids=sorted(pids or []),
            )
        else:
            checks.expect("P12", True, "this PID is listed in nvidia-smi compute apps")
        return {"settings": settings}
    finally:
        backend.close()


def make_work(torch: Any) -> tuple[str, Any]:
    if hasattr(torch.cuda, "_sleep"):
        return "torch.cuda._sleep", lambda n: torch.cuda._sleep(int(n))
    a = torch.randn(4096, 4096, device="cuda")

    def matmuls(n: float) -> None:
        for _ in range(int(n)):
            a @ a  # the GPU work is the point; the result is discarded

    return "matmul-4096", matmuls


def trial(torch: Any, launch: Any, amount: float) -> tuple[float, float, float]:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    h0 = time.perf_counter_ns()
    start.record()
    launch(amount)
    h_launch = time.perf_counter_ns()
    end.record()
    torch.cuda.synchronize()
    h1 = time.perf_counter_ns()
    return (h_launch - h0) / 1e6, (h1 - h0) / 1e6, float(start.elapsed_time(end))


def section5(torch: Any, checks: Checks) -> dict[str, Any]:
    from gpu_benchlab.backends.pytorch import CudaEventTimer
    from gpu_benchlab.core.timing import WallClockTimer

    workload, launch = make_work(torch)
    probe = 1_000_000 if workload == "torch.cuda._sleep" else 10
    for _ in range(5):
        trial(torch, launch, probe)  # warm the path; discarded
    per_unit = st.median(trial(torch, launch, probe)[2] for _ in range(10)) / probe
    base = 5.0 / per_unit  # work amount that takes about 5 ms
    raw: dict[str, Any] = {"workload": workload, "base_amount": base}
    for mult in (1, 2, 4):
        rows = [trial(torch, launch, base * mult) for _ in range(TRIALS)]
        raw[f"x{mult}"] = {
            "launch_ms": [r[0] for r in rows],
            "host_sync_ms": [r[1] for r in rows],
            "event_ms": [r[2] for r in rows],
        }
    ev = {m: st.median(raw[f"x{m}"]["event_ms"]) for m in (1, 2, 4)}
    x4 = raw["x4"]
    checks.expect(
        "E1",
        st.median(x4["launch_ms"]) < 0.05 * ev[4],
        "launch returns long before the GPU finishes (asynchronous)",
        launch_median_ms=st.median(x4["launch_ms"]),
        event_median_ms=ev[4],
    )
    checks.expect(
        "E2",
        1.9 <= ev[2] / ev[1] <= 2.1 and 3.8 <= ev[4] / ev[1] <= 4.2,
        "event time scales linearly with GPU work",
        ratio_2x=ev[2] / ev[1],
        ratio_4x=ev[4] / ev[1],
        medians_ms=ev,
    )
    violations = [
        (m, i)
        for m in (1, 2, 4)
        for i, (h, e) in enumerate(
            zip(raw[f"x{m}"]["host_sync_ms"], raw[f"x{m}"]["event_ms"], strict=True)
        )
        if h < e
    ]
    checks.expect(
        "E3",
        not violations,
        "synchronized host interval contains the event interval",
        violations=violations,
        median_host_minus_event_ms={
            m: st.median(
                h - e
                for h, e in zip(raw[f"x{m}"]["host_sync_ms"], raw[f"x{m}"]["event_ms"], strict=True)
            )
            for m in (1, 2, 4)
        },
    )
    cv = st.stdev(x4["event_ms"]) / st.mean(x4["event_ms"])
    checks.expect("E4", cv < 0.02, "event time is stable (CV < 2%)", cv=cv)

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    time.sleep(0.020)
    launch(base)
    end.record()
    torch.cuda.synchronize()
    gap_ms = float(start.elapsed_time(end))
    checks.expect(
        "E5",
        gap_ms >= 20.0,
        "idle stream time between the events is counted (documented "
        "semantics: device time between events, not the sum of kernel time)",
        event_ms=gap_ms,
        work_alone_ms=ev[1],
    )

    timer = CudaEventTimer(torch, torch.device("cuda:0"))
    primary = []
    for _ in range(TRIALS):
        timer.start()
        launch(base * 4)
        primary.append(timer.stop())
    drained = timer.drain_secondary()
    secondary = drained[1] if drained else []
    checks.expect(
        "E6",
        abs(st.median(primary) - ev[4]) <= 0.02 * ev[4]
        and len(secondary) == TRIALS
        and all(s >= p for s, p in zip(secondary, primary, strict=True)),
        "real CudaEventTimer agrees with the reference event time; secondary host >= primary",
        timer_median_ms=st.median(primary),
        reference_ms=ev[4],
    )
    raw["cuda_event_timer"] = {"primary_ms": primary, "secondary_host_ms": secondary}

    naive = WallClockTimer()  # no synchronization hook: the bug this experiment must catch
    naive_ms = []
    for _ in range(TRIALS):
        naive.start()
        launch(base * 4)
        naive_ms.append(naive.stop())
        torch.cuda.synchronize()  # outside the timed region: do not let work queue up
    checks.expect(
        "E7",
        st.median(naive_ms) < 0.05 * ev[4],
        "negative control: an unsynchronized host timer reads only the launch",
        naive_median_ms=st.median(naive_ms),
        event_median_ms=ev[4],
    )
    raw["unsynchronized_wall_clock_ms"] = naive_ms
    return raw


def negative(checks: Checks, out_dir: Path) -> None:
    config = out_dir / "cuda-hidden.json"
    config.write_text(json.dumps(CONFIG), encoding="utf-8")
    results = out_dir / "negative-no-cuda"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run(  # noqa: S603 - fixed argv built from this script's own paths
        [
            sys.executable,
            "-m",
            "gpu_benchlab.cli.main",
            "run",
            "-c",
            str(config),
            "--results-dir",
            str(results),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    stored = [json.loads(p.read_text(encoding="utf-8")) for p in results.glob("*/result.json")]
    # Every stored run must have refused, not just the newest: a leftover from an
    # earlier invocation that had measured something would be a finding, not noise.
    ok = bool(stored) and all(
        s["status"] in ("unavailable", "failed") and not s["raw_samples"]["latency_ms"]
        for s in stored
    )
    checks.expect(
        "N1",
        ok,
        "CUDA hidden: result is unavailable/failed with zero samples (no CPU fallback)",
        exit_code=done.returncode,
        stored_status=[s["status"] for s in stored],
        errors=[s.get("errors") for s in stored],
        stdout_tail=done.stdout[-2000:],
    )


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase5a/pytorch")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch = require_cuda_torch()
    checks = Checks("pytorch_cuda_check")
    s4 = section4(torch, checks)
    s5 = section5(torch, checks)
    negative(checks, out_dir)
    checks.add(
        "versions",
        INFO,
        "software",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
    )
    checks.finish(out_dir, {"section4": s4, "section5_raw": s5})


if __name__ == "__main__":
    main()
