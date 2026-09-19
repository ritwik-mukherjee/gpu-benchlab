"""`gpu-bench onnx` — export registry models to ONNX and verify them against PyTorch."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from gpu_benchlab.core.errors import BenchLabError
from gpu_benchlab.core.storage import DEFAULT_RESULTS_DIR
from gpu_benchlab.export.onnx_export import DEFAULT_OPSET, ExportConfig

console = Console()

onnx_app = typer.Typer(
    help="Export models to ONNX and check ONNX Runtime against PyTorch.", no_args_is_help=True
)

EXIT_OK = 0
EXIT_FAILED = 1


def _config(model: str, weights: str | None, opset: int, static_batch: int | None) -> ExportConfig:
    return ExportConfig(
        model=model,
        weights=weights,
        opset=opset,
        dynamic_batch=static_batch is None,
        trace_batch_size=static_batch or 1,
    )


@onnx_app.command("export")
def export_command(
    model: Annotated[str, typer.Argument(help="Registry model, e.g. resnet50.")],
    weights: Annotated[str | None, typer.Option(help="Weights id (default: registry pin).")] = None,
    opset: Annotated[int, typer.Option(help="ONNX opset (pinned default).")] = DEFAULT_OPSET,
    static_batch: Annotated[
        int | None, typer.Option(help="Export with a fixed batch instead of a dynamic one.")
    ] = None,
) -> None:
    """Export a model to ONNX (overwrites) and print its manifest summary."""
    from gpu_benchlab.export.onnx_export import export_onnx, validate_artifact

    try:
        path, manifest = export_onnx(_config(model, weights, opset, static_batch))
        checks = validate_artifact(path, manifest)
    except BenchLabError as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {escape(str(exc))}")
        raise typer.Exit(EXIT_FAILED) from exc

    console.print(
        f"Exported [cyan]{manifest.artifact.filename}[/cyan] ({manifest.artifact.bytes:,} bytes)"
    )
    console.print(f"  sha256   {manifest.artifact.sha256}")
    console.print(f"  weights  {manifest.model.weights} ({manifest.model.weights_sha256})")
    console.print(f"  opset    {manifest.exporter.opset}   input {manifest.input.shape}")
    console.print(f"  location {path.parent}")
    for check in checks:
        console.print(f"  [green]PASS[/green] {check}")


@onnx_app.command("verify")
def verify_command(
    model: Annotated[str, typer.Argument(help="Registry model, e.g. resnet50.")],
    weights: Annotated[str | None, typer.Option(help="Weights id (default: registry pin).")] = None,
    opset: Annotated[int, typer.Option(help="ONNX opset.")] = DEFAULT_OPSET,
    device: Annotated[str, typer.Option(help="ONNX Runtime device: cpu or cuda:N.")] = "cpu",
    results_dir: Annotated[
        Path, typer.Option(help="Where to store the report.")
    ] = DEFAULT_RESULTS_DIR,
) -> None:
    """Compare ONNX Runtime outputs with PyTorch on identical inputs; store the evidence."""
    from gpu_benchlab.export.verify import save_report, verify_against_pytorch

    try:
        report, outputs = verify_against_pytorch(
            _config(model, weights, opset, None), device=device
        )
    except BenchLabError as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {escape(str(exc))}")
        raise typer.Exit(EXIT_FAILED) from exc
    directory = save_report(report, outputs, results_dir)

    table = Table(box=None, pad_edge=False)
    for col in ("batch", "seed", "max abs err", "mean abs err", "worst/tol", "top-1", "result"):
        table.add_column(col, justify="right")
    for case in report.cases:
        c = case.comparison
        table.add_row(
            str(case.batch_size),
            str(case.seed),
            f"{c.max_abs_error:.3e}" if c.max_abs_error is not None else "-",
            f"{c.mean_abs_error:.3e}" if c.mean_abs_error is not None else "-",
            f"{c.worst_violation_ratio:.4f}" if c.worst_violation_ratio is not None else "-",
            f"{c.top1_agreement:.3f}" if c.top1_agreement is not None else "-",
            "[green]pass[/green]"
            if c.passed
            else f"[red]FAIL[/red] {escape('; '.join(c.reasons))}",
        )
    console.print(table)
    nc = report.negative_control.comparison
    control = (
        "[green]rejected, as required[/green]"
        if report.negative_control_rejected
        else "[red]NOT rejected -- the tolerance cannot detect reduced precision[/red]"
    )
    worst = f"{nc.worst_violation_ratio:.1f}" if nc.worst_violation_ratio is not None else "-"
    console.print(
        f"Negative control (FP16 PyTorch vs FP32 reference): {control} (worst/tol {worst})"
    )
    console.print(f"tolerance: rtol={report.rtol:g}, atol={report.atol_scale:g}*max|ref|")
    verdict = "[green]PASSED[/green]" if report.passed else "[red]FAILED[/red]"
    console.print(
        f"\n{verdict}  {report.cases_passed}/{report.cases_total} cases.  Evidence: {directory}"
    )
    raise typer.Exit(EXIT_OK if report.passed else EXIT_FAILED)
