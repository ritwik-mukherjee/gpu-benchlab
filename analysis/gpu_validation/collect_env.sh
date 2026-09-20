#!/usr/bin/env bash
# Phase 5A runbook step 1: capture the GPU machine's environment.
# Run once BEFORE installing anything (out_dir .../env-before) and once after
# (.../env-after). Read-only: it changes no setting. A failing command is recorded
# with its exit code, never skipped silently.
#
# Usage: bash analysis/gpu_validation/collect_env.sh <out_dir>
set -u
out="${1:?usage: collect_env.sh <out_dir>}"
mkdir -p "$out"

run() {
  local name="$1"; shift
  { echo "\$ $*"; "$@"; echo "[exit $?]"; } > "$out/$name.txt" 2>&1
}

run date           date -u +%Y-%m-%dT%H:%M:%SZ
run uname          uname -a
run os-release     cat /etc/os-release
run virtualization systemd-detect-virt
run cpu            lscpu
run memory         free -b
run nvidia-smi     nvidia-smi
run nvidia-smi-q   nvidia-smi -q
run nvidia-smi-L   nvidia-smi -L
# If a field is rejected by this driver, the error is recorded: look it up in
# nvidia-smi-help-query-gpu.txt and record the substitution in the run notes.
run nvidia-smi-static nvidia-smi --format=csv --query-gpu=index,name,uuid,pci.bus_id,serial,compute_cap,memory.total,driver_version,vbios_version,power.limit,enforced.power.limit,power.max_limit,clocks.max.sm,clocks.max.mem,persistence_mode,compute_mode,mig.mode.current,ecc.mode.current,pcie.link.gen.max,pcie.link.width.max
run nvidia-smi-help-query-gpu nvidia-smi --help-query-gpu
run nvcc           nvcc --version
run python         python -VV
run uv             uv --version
run pip-freeze     uv pip freeze
run git-commit     git rev-parse HEAD
run git-status     git status --porcelain
run gpu-bench-hardware gpu-bench hardware
run gpu-bench-doctor   gpu-bench doctor
run environment-json   gpu-bench hardware --json -o "$out/environment.json"

# onnxruntime is imported BEFORE torch so torch cannot preload CUDA libraries for it.
python - > "$out/frameworks.json" 2> "$out/frameworks.stderr.txt" <<'EOF'
import json
out = {}
try:
    import onnxruntime as ort
    out["onnxruntime"] = {
        "version": ort.__version__,
        "device": ort.get_device(),
        "available_providers": ort.get_available_providers(),
        "build_info": ort.get_build_info() if hasattr(ort, "get_build_info") else None,
    }
except Exception as exc:
    out["onnxruntime"] = repr(exc)
try:
    import torch
    t = {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "is_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "arch_list": torch.cuda.get_arch_list() if torch.version.cuda else None,
    }
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        t[f"device{i}"] = {
            "name": p.name,
            "capability": f"{p.major}.{p.minor}",
            "total_memory_bytes": p.total_memory,
            "multi_processor_count": p.multi_processor_count,
            "uuid": str(getattr(p, "uuid", None)),
        }
    out["torch"] = t
except Exception as exc:
    out["torch"] = repr(exc)
print(json.dumps(out, indent=2))
EOF
echo "[exit $?]" >> "$out/frameworks.stderr.txt"
echo "environment captured in $out"
