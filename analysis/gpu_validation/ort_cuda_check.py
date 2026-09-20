"""Phase 5A runbook §8: prove ONNX Runtime's CUDA EP actually executes the model.

Phase 4 observed onnxruntime-gpu LISTING CUDAExecutionProvider and then silently
building a CPU session. So "the provider is listed" proves nothing. This script uses
the raw ORT API (independent of gpu_benchlab's backend code, same settings) and
collects evidence from several independent sources.

Run it twice (runbook §8): once as-is, and once with --import-torch-first. torch
preloads its own CUDA/cuDNN libraries, and `gpu-bench run` imports torch during
environment detection -- so whether the EP loads WITHOUT torch is a separate fact.

Pre-registered checks:
  O1 requested       CUDAExecutionProvider is first in session.get_providers()
  O2 options         effective use_tf32 == "0" (ORT's default is 1)
  O3 libraries       libonnxruntime_providers_cuda, libcudnn and libcublas are mapped
                     into this process (Linux /proc/self/maps)
  O4 placement       ORT profiling: every executed node ran on CUDAExecutionProvider
                     (node names of any other EP are listed; the profile is kept)
  O5 iobinding       input OrtValue and bound output live on "cuda"
  O6 output          (B, 1000) float32, finite
  O7 gpu process     this PID is listed by nvidia-smi --query-compute-apps
  S1 ORT waits       median(run + device sync) - median(run) <= max(0.05 ms, 2%):
                     run_with_iobinding already waited for the GPU, so a host clock
                     around it (the backend's timer) measures completed work
  S2 control power   with disable_synchronize_execution_providers=1, median(run) is
                     < 80% of the default -- else S1 could not have detected a missing
                     sync at this batch size: INCONCLUSIVE, rerun with --batch 32
  N1 no fallback     gpu-bench run with CUDA hidden -> unavailable/failed, 0 samples

Prerequisite: `gpu-bench onnx export resnet50` (this script never exports, because
exporting imports torch).

Usage: python analysis/gpu_validation/ort_cuda_check.py <out_dir> [--batch N] [--import-torch-first]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from _common import FAIL, INCONCLUSIVE, INFO, Checks, compute_app_pids, loaded_libraries

CUDA_EP = "CUDAExecutionProvider"
PROVIDER_OPTIONS = {"device_id": "0", "use_tf32": "0", "cudnn_conv_algo_search": "EXHAUSTIVE"}
ITERATIONS = 200


def device_synchronizer(torch: Any) -> tuple[str, Any] | None:
    """A device-wide synchronize that waits for every stream ORT may have used."""
    if torch is not None:
        return "torch.cuda.synchronize", torch.cuda.synchronize
    libs = loaded_libraries("libcudart") or {}
    for path in libs.get("libcudart", []):
        cudart = ctypes.CDLL(path)  # the runtime ORT itself loaded

        def sync(lib: Any = cudart) -> None:
            if lib.cudaDeviceSynchronize() != 0:
                raise RuntimeError("cudaDeviceSynchronize failed")

        return f"cudaDeviceSynchronize via {path}", sync
    return None


def timed(fn: Any, n: int) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--import-torch-first", action="store_true")
    args = parser.parse_args()
    tag = f"b{args.batch}-{'torch-first' if args.import_torch_first else 'no-torch'}"
    out_dir: Path = args.out_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = Checks("ort_cuda_check")

    torch: Any = None
    if args.import_torch_first:
        import torch as preloaded  # deliberately preloads torch's CUDA libraries

        torch = preloaded

    import numpy as np
    import onnxruntime as ort

    from gpu_benchlab.core.inputs import synthetic_input
    from gpu_benchlab.export.onnx_export import ExportConfig, ensure_artifact
    from gpu_benchlab.models.registry import get_model

    weights = get_model("resnet50").resolve_weights(None)
    artifact, manifest, _ = ensure_artifact(
        ExportConfig(model="resnet50", weights=weights), allow_export=False
    )
    checks.add(
        "artifact",
        INFO,
        "artifact",
        path=str(artifact),
        sha256=manifest.artifact.sha256,
        weights_sha256=manifest.model.weights_sha256,
        torch_in_process="torch" in sys.modules,
    )
    available = ort.get_available_providers()
    if CUDA_EP not in available:
        checks.add(
            "O0",
            FAIL,
            "CUDA EP not even listed (CPU-only onnxruntime package?)",
            available=available,
        )
        checks.finish(out_dir)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_profiling = True
    so.profile_file_prefix = str(out_dir / "ort-profile")
    session = ort.InferenceSession(str(artifact), so, providers=[(CUDA_EP, PROVIDER_OPTIONS)])
    active = session.get_providers()
    checks.expect(
        "O1",
        bool(active) and active[0] == CUDA_EP,
        "CUDA EP is the session's first active provider",
        active=active,
        available=available,
    )
    if CUDA_EP not in active:
        checks.add("stop", FAIL, "ORT substituted another EP; nothing further is meaningful")
        checks.finish(out_dir, {"libraries": loaded_libraries("onnxruntime", "cud", "cublas")})
    effective = session.get_provider_options().get(CUDA_EP, {})
    checks.expect(
        "O2",
        effective.get("use_tf32") == "0",
        "effective use_tf32 is 0 (IEEE FP32)",
        effective={k: effective.get(k) for k in PROVIDER_OPTIONS},
    )

    libs = loaded_libraries("libonnxruntime_providers_cuda", "libcudnn", "libcublas", "libcudart")
    if libs is None:
        checks.add("O3", INCONCLUSIVE, "no /proc/self/maps (not Linux); libraries unchecked")
    else:
        checks.expect(
            "O3",
            all(libs[k] for k in ("libonnxruntime_providers_cuda", "libcudnn", "libcublas")),
            "CUDA EP, cuDNN and cuBLAS libraries are loaded in this process",
            libraries=libs,
        )

    x = synthetic_input((args.batch, 3, 224, 224), seed=0)
    x_gpu = ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
    binding = session.io_binding()
    binding.bind_ortvalue_input(manifest.input.name, x_gpu)
    binding.bind_output(manifest.output.name, "cuda", 0)
    session.run_with_iobinding(binding)
    bound = binding.get_outputs()[0]
    checks.expect(
        "O5",
        x_gpu.device_name() == "cuda" and bound.device_name() == "cuda",
        "IOBinding input and output are device-resident",
        input_device=x_gpu.device_name(),
        output_device=bound.device_name(),
    )
    y = binding.copy_outputs_to_cpu()[0]
    checks.expect(
        "O6",
        y.shape == (args.batch, 1000) and y.dtype == np.float32 and bool(np.isfinite(y).all()),
        "output shape, dtype, finiteness",
        shape=list(y.shape),
        dtype=str(y.dtype),
    )

    profile = Path(session.end_profiling())
    events = json.loads(profile.read_text(encoding="utf-8"))
    nodes: dict[str, list[str]] = {}
    for e in events:
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
            ep = e.get("args", {}).get("provider", "?")
            nodes.setdefault(ep, []).append(e["name"].removesuffix("_kernel_time"))
    counts = {ep: len(v) for ep, v in nodes.items()}
    foreign = {ep: v for ep, v in nodes.items() if ep != CUDA_EP}
    checks.expect(
        "O4",
        bool(nodes.get(CUDA_EP)) and not foreign,
        "every profiled node executed on the CUDA EP",
        counts=counts,
        foreign_nodes=foreign,
        profile=str(profile),
    )

    pids = compute_app_pids()
    if pids is None or os.getpid() not in pids:
        checks.add(
            "O7",
            INCONCLUSIVE,
            "PID not visible to nvidia-smi (containers hide it)",
            pids=sorted(pids or []),
        )
    else:
        checks.expect("O7", True, "this PID is listed in nvidia-smi compute apps")

    # Timing semantics on a fresh session without profiling (profiling adds overhead).
    so.enable_profiling = False
    session = ort.InferenceSession(str(artifact), so, providers=[(CUDA_EP, PROVIDER_OPTIONS)])
    binding = session.io_binding()
    binding.bind_ortvalue_input(manifest.input.name, x_gpu)
    binding.bind_output(manifest.output.name, "cuda", 0)
    no_sync = ort.RunOptions()
    no_sync.add_run_config_entry("disable_synchronize_execution_providers", "1")
    syncer = device_synchronizer(torch)
    raw: dict[str, Any] = {"batch": args.batch}
    if syncer is None:
        checks.add(
            "S1", INCONCLUSIVE, "no device synchronize available; rerun with --import-torch-first"
        )
    else:
        name, sync = syncer

        def run() -> None:
            session.run_with_iobinding(binding)

        def run_then_sync() -> None:
            session.run_with_iobinding(binding)
            sync()

        def run_no_sync() -> None:
            session.run_with_iobinding(binding, no_sync)

        timed(run_then_sync, 50)  # warmup, discarded
        raw["run_ms"] = timed(run, ITERATIONS)
        raw["run_then_device_sync_ms"] = timed(run_then_sync, ITERATIONS)
        control = []
        for _ in range(ITERATIONS):
            control += timed(run_no_sync, 1)
            sync()  # outside the timed region
        raw["run_sync_disabled_ms"] = control
        a, b, c = (
            st.median(raw[k]) for k in ("run_ms", "run_then_device_sync_ms", "run_sync_disabled_ms")
        )
        checks.expect(
            "S1",
            b - a <= max(0.05, 0.02 * a),
            "run_with_iobinding returns only after the GPU finished",
            median_run_ms=a,
            median_run_then_sync_ms=b,
            synchronizer=name,
        )
        if c < 0.8 * a:
            checks.expect(
                "S2",
                True,
                "control: disabling ORT's sync is detectable",
                median_sync_disabled_ms=c,
                median_run_ms=a,
            )
        else:
            checks.add(
                "S2",
                INCONCLUSIVE,
                "control could not separate sync from no-sync at this batch; rerun with --batch 32",
                median_sync_disabled_ms=c,
                median_run_ms=a,
            )

    config = out_dir / "cuda-hidden.json"
    config.write_text(
        json.dumps(
            {
                "name": "phase5a-ort-cuda-hidden",
                "backend": "onnxruntime",
                "device": "cuda:0",
                "precision": "fp32",
                "batch_size": 1,
                "model": {"name": "resnet50", "weights": "IMAGENET1K_V2"},
                "backend_options": {"allow_export": False},
                "benchmark": {"warmup_iterations": 10, "measurement_iterations": 100},
            }
        ),
        encoding="utf-8",
    )
    results = out_dir / "negative-no-cuda"
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
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    stored = [json.loads(p.read_text(encoding="utf-8")) for p in results.glob("*/result.json")]
    checks.expect(
        "N1",
        len(stored) == 1
        and stored[0]["status"] in ("unavailable", "failed")
        and not stored[0]["raw_samples"]["latency_ms"],
        "CUDA hidden: unavailable/failed with zero samples (no silent CPU session)",
        exit_code=done.returncode,
        status=[s["status"] for s in stored],
        errors=[s.get("errors") for s in stored],
    )
    checks.add(
        "versions",
        INFO,
        "software",
        onnxruntime=ort.__version__,
        torch=getattr(torch, "__version__", None),
    )
    checks.finish(out_dir, {"timing_raw": raw, "libraries": libs})


if __name__ == "__main__":
    main()
