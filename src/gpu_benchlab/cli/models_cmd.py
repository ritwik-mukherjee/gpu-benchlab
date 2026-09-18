"""`gpu-bench models` — list registered models and fetch their pinned weights."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from gpu_benchlab.core.errors import BenchLabError, ConfigurationError
from gpu_benchlab.models.registry import RANDOM_WEIGHTS, get_model, list_models
from gpu_benchlab.models.weights import default_cache_dir, ensure_weights

console = Console()

models_app = typer.Typer(
    help="List benchmark models and fetch pinned weights.", no_args_is_help=True
)


@models_app.command("list")
def list_command() -> None:
    """Show registered models, their pinned weights and expected parameter counts."""
    table = Table(box=None, pad_edge=False)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Family", no_wrap=True)
    table.add_column("Parameters", justify="right", no_wrap=True)
    table.add_column("Input", no_wrap=True)
    table.add_column("Weights (default first)")
    for spec in list_models():
        ids = [spec.default_weights] + [w for w in spec.weights if w != spec.default_weights]
        table.add_row(
            spec.name,
            spec.family,
            f"{spec.expected_parameters:,}",
            "x".join(str(d) for d in spec.input_shape),
            ", ".join([*ids, RANDOM_WEIGHTS]),
        )
    console.print(table)
    console.print(f"\n[dim]Weights cache: {default_cache_dir()}[/dim]")


@models_app.command("fetch")
def fetch_command(
    name: Annotated[str, typer.Argument(help="Registered model name, e.g. resnet50.")],
    weights: Annotated[
        str | None, typer.Option("--weights", help="Weights id. Defaults to the pinned default.")
    ] = None,
) -> None:
    """Download a model's pinned weights into the cache and verify their SHA-256."""
    try:
        spec = get_model(name)
        weights_id = spec.resolve_weights(weights)
        if weights_id == RANDOM_WEIGHTS:
            raise ConfigurationError("'random' weights need no download.")
        resolved = ensure_weights(spec.weights[weights_id])
    except BenchLabError as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from exc

    state = "downloaded" if resolved.downloaded else "already cached"
    console.print(f"{spec.name} {weights_id}: {state}")
    console.print(f"  path   {resolved.path}")
    console.print(f"  sha256 {resolved.sha256}")
