"""`gpu-bench` command-line interface.

Phase 1 exposes environment detection only. Later phases add `run`, `suite`,
`compare`, `report` and `dashboard`.

Exit codes are meaningful so the CLI is usable in CI:
    0  detection succeeded and an NVIDIA GPU is present
    1  detection ran but no usable NVIDIA GPU was found
    2  detection itself failed unexpectedly
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gpu_benchlab import __version__
from gpu_benchlab.cli import models_cmd, onnx_cmd, run_cmd
from gpu_benchlab.hardware import DetectionStatus, EnvironmentReport, detect_environment

app = typer.Typer(
    name="gpu-bench",
    help="Measure and compare AI inference performance across NVIDIA acceleration stacks.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)

EXIT_OK = 0
EXIT_NO_GPU = 1
EXIT_DETECTION_FAILED = 2

# Human-readable guidance per failure mode. Kept next to the status enum it explains
# so that a new status cannot be added without someone noticing the missing advice.
_STATUS_GUIDANCE: dict[DetectionStatus, str] = {
    DetectionStatus.NO_NVIDIA_DEVICE: (
        "The NVIDIA driver is installed and responding, but it reports no GPUs. "
        "If a GPU is physically present, check that it is not hidden by "
        "CUDA_VISIBLE_DEVICES and that it appears in your OS device manager."
    ),
    DetectionStatus.DRIVER_UNAVAILABLE: (
        "No NVIDIA driver could be loaded. Install the NVIDIA driver for your GPU, "
        "or run this tool on a machine with an NVIDIA GPU. Benchmarks cannot run here."
    ),
    DetectionStatus.LIBRARY_UNAVAILABLE: (
        "The 'nvidia-ml-py' package is missing. Install it with: pip install nvidia-ml-py"
    ),
    DetectionStatus.ERROR: (
        "NVML returned an unexpected error. The full message is shown above; please "
        "include it when opening an issue."
    ),
}


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"gpu-bench {__version__}")
        raise typer.Exit(EXIT_OK)


@app.callback()
def _main(
    _version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """gpu-bench root command."""


# ---------------------------------------------------------------------------------------
# hardware
# ---------------------------------------------------------------------------------------


@app.command()
def hardware(
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the raw EnvironmentReport as JSON.")
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the JSON report to this path."),
    ] = None,
    skip_frameworks: Annotated[
        bool,
        typer.Option(
            "--skip-frameworks",
            help="Skip importing PyTorch/ONNX Runtime/TensorRT (much faster).",
        ),
    ] = False,
) -> None:
    """Detect GPUs, driver, CUDA and installed inference frameworks."""
    report = detect_environment(include_frameworks=not skip_frameworks)

    payload = report.model_dump_json(indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")

    if as_json:
        console.print_json(payload)
    else:
        _render_report(report)
        if output is not None:
            console.print(f"\n[dim]JSON report written to {output}[/dim]")

    raise typer.Exit(_exit_code(report))


def _exit_code(report: EnvironmentReport) -> int:
    if report.has_nvidia_gpu:
        return EXIT_OK
    if report.detection_status in (DetectionStatus.ERROR, DetectionStatus.LIBRARY_UNAVAILABLE):
        return EXIT_DETECTION_FAILED
    return EXIT_NO_GPU


# ---------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------


def _fmt_bytes(value: int | None) -> str:
    if value is None:
        return "[dim]not reported[/dim]"
    return f"{value / (1024**3):.2f} GiB"


def _fmt(value: object, suffix: str = "") -> str:
    if value is None:
        return "[dim]not reported[/dim]"
    return f"{value}{suffix}"


def _render_report(report: EnvironmentReport) -> None:
    _render_host(report)
    _render_gpus(report)
    _render_frameworks(report)
    _render_verdict(report)


def _render_host(report: EnvironmentReport) -> None:
    host = report.host
    table = Table(title="Host", title_justify="left", show_header=False, box=None, pad_edge=False)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()

    cores = "[dim]not reported[/dim]"
    if host.cpu_logical_cores is not None:
        physical = host.cpu_physical_cores
        cores = (
            f"{physical} physical / {host.cpu_logical_cores} logical"
            if physical
            else str(host.cpu_logical_cores)
        )

    table.add_row("Hostname", host.hostname)
    table.add_row("OS", f"{host.os_name} {host.os_release or ''} ({host.os_version})".strip())
    table.add_row("Architecture", host.architecture)
    table.add_row("CPU", _fmt(host.cpu_model))
    table.add_row("Cores", cores)
    table.add_row("System RAM", _fmt_bytes(host.total_memory_bytes))
    table.add_row(
        "Python", f"{host.python_implementation} {host.python_version} ({host.python_executable})"
    )
    console.print(table)
    console.print()


def _render_gpus(report: EnvironmentReport) -> None:
    nvml = report.nvml
    table = Table(
        title="NVIDIA driver", title_justify="left", show_header=False, box=None, pad_edge=False
    )
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    table.add_row("Detection status", _status_markup(report.detection_status))
    table.add_row("Driver version", _fmt(nvml.driver_version))
    table.add_row("NVML version", _fmt(nvml.nvml_version))
    table.add_row("CUDA (driver max)", _fmt(nvml.cuda_driver_version))
    console.print(table)
    console.print()

    if not report.gpus:
        return

    for gpu in report.gpus:
        gt = Table(
            title=f"GPU {gpu.index}: {gpu.name}",
            title_justify="left",
            show_header=False,
            box=None,
            pad_edge=False,
        )
        gt.add_column(style="cyan", no_wrap=True)
        gt.add_column()
        gt.add_row("Architecture", _fmt(gpu.architecture))
        gt.add_row("Compute capability", _fmt(gpu.compute_capability))
        gt.add_row("UUID", _fmt(gpu.uuid))
        gt.add_row("PCI bus", _fmt(gpu.pci_bus_id))
        gt.add_row("VRAM total", _fmt_bytes(gpu.memory_total_bytes))
        gt.add_row("VRAM free", _fmt_bytes(gpu.memory_free_bytes))
        gt.add_row("Utilisation (GPU)", _fmt(gpu.utilization_gpu_percent, " %"))
        gt.add_row("Temperature", _fmt(gpu.temperature_celsius, " °C"))
        gt.add_row("Power draw", _fmt(gpu.power_usage_watts, " W"))
        gt.add_row("Power limit", _fmt(gpu.power_limit_watts, " W"))
        gt.add_row("SM clock", _fmt(gpu.sm_clock_mhz, " MHz"))
        gt.add_row("Compute mode", _fmt(gpu.compute_mode))
        console.print(gt)
        console.print("[dim]Telemetry above is an instantaneous sample at detection time.[/dim]")
        console.print()

        if gpu.precision_support:
            pt = Table(title="Precision support", title_justify="left", box=None, pad_edge=False)
            pt.add_column("Precision", style="cyan", no_wrap=True)
            pt.add_column("Supported", no_wrap=True)
            pt.add_column("Tensor core", no_wrap=True)
            pt.add_column("Note")
            for ps in gpu.precision_support:
                pt.add_row(
                    ps.precision.value.upper(),
                    "[green]yes[/green]" if ps.supported else "[red]no[/red]",
                    "[green]yes[/green]" if ps.tensor_core else "[dim]no[/dim]",
                    ps.note,
                )
            console.print(pt)
            console.print()


def _status_markup(status: DetectionStatus) -> str:
    colour = {
        DetectionStatus.OK: "green",
        DetectionStatus.NO_NVIDIA_DEVICE: "yellow",
        DetectionStatus.DRIVER_UNAVAILABLE: "yellow",
        DetectionStatus.LIBRARY_UNAVAILABLE: "red",
        DetectionStatus.ERROR: "red",
    }[status]
    return f"[{colour}]{status.value}[/{colour}]"


def _render_frameworks(report: EnvironmentReport) -> None:
    if not report.frameworks:
        return
    table = Table(title="Inference frameworks", title_justify="left", box=None, pad_edge=False)
    table.add_column("Framework", style="cyan", no_wrap=True)
    table.add_column("Installed", no_wrap=True)
    table.add_column("Version", no_wrap=True)
    table.add_column("CUDA usable", no_wrap=True)
    table.add_column("Detail")

    for fw in report.frameworks:
        if not fw.available or fw.cuda_available is None:
            cuda_cell = "[dim]n/a[/dim]"
        else:
            cuda_cell = "[green]yes[/green]" if fw.cuda_available else "[red]no[/red]"

        detail = fw.detail or ""
        if fw.providers:
            detail = f"providers: {', '.join(fw.providers)}"

        table.add_row(
            fw.name,
            "[green]yes[/green]" if fw.available else "[dim]no[/dim]",
            fw.version or "[dim]-[/dim]",
            cuda_cell,
            detail,
        )
    console.print(table)
    console.print()


def _render_verdict(report: EnvironmentReport) -> None:
    """Print an unambiguous statement of whether real GPU benchmarks can run here."""
    if report.has_nvidia_gpu:
        names = ", ".join(g.name for g in report.gpus)
        console.print(
            Panel(
                f"[green]NVIDIA GPU detected:[/green] {names}\n"
                "GPU benchmarks can run on this machine.",
                title="Verdict",
                border_style="green",
            )
        )
        return

    lines = ["[yellow]No usable NVIDIA GPU detected on this machine.[/yellow]"]
    if report.detection_error:
        lines.append(f"\n[dim]{report.detection_error}[/dim]")
    guidance = _STATUS_GUIDANCE.get(report.detection_status)
    if guidance:
        lines.append(f"\n{guidance}")
    lines.append(
        "\n[bold]GPU benchmark results cannot be produced here.[/bold] "
        "This tool will report such configurations as 'unavailable' rather than "
        "estimating or fabricating numbers."
    )
    console.print(Panel("\n".join(lines), title="Verdict", border_style="yellow"))


# ---------------------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Summarise, in one line per check, whether this machine can run benchmarks."""
    report = detect_environment(include_frameworks=True)

    checks: list[tuple[str, bool, str]] = []
    checks.append(
        (
            "NVIDIA driver",
            report.detection_status
            not in (DetectionStatus.DRIVER_UNAVAILABLE, DetectionStatus.LIBRARY_UNAVAILABLE),
            report.nvml.driver_version or "not available",
        )
    )
    checks.append(
        (
            "NVIDIA GPU present",
            report.has_nvidia_gpu,
            f"{len(report.gpus)} device(s)" if report.gpus else "none",
        )
    )
    for fw in report.frameworks:
        usable = bool(fw.available and (fw.cuda_available or fw.name == "tensorrt"))
        checks.append((fw.name, usable, fw.version or "not installed"))

    table = Table(box=None, pad_edge=False, show_header=False)
    table.add_column(no_wrap=True)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="dim")
    for name, ok, detail in checks:
        table.add_row("[green]PASS[/green]" if ok else "[yellow]FAIL[/yellow]", name, detail)
    console.print(table)

    raise typer.Exit(_exit_code(report))


# ---------------------------------------------------------------------------------------
# run (Phase 2)
# ---------------------------------------------------------------------------------------

run_cmd.register(app)
app.add_typer(models_cmd.models_app, name="models")
app.add_typer(onnx_cmd.onnx_app, name="onnx")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    sys.exit(app())
