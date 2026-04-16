"""forgekit CLI — `run`, `list-recipes`, `list-plugins`, `doctor`."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from forgekit import __version__
from forgekit.hw import detect as hw_detect
from forgekit.recipe import load_recipe
from forgekit.registry import STAGE_GROUPS, discover, get
from forgekit.stages import StageContext

app = typer.Typer(
    name="forgekit",
    help="Research-incubator pipeline for local LLM development.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


# Optional extras surfaced in `doctor`. Name → pip module import-check target.
_OPTIONAL_EXTRAS: dict[str, str] = {
    "unsloth": "unsloth",
    "trl": "trl",
    "torchtune": "torchtune",
    "onecompression": "onecompression",
    "auto-gptq": "auto_gptq",
    "autoawq": "awq",
    "hqq": "hqq",
    "bitsandbytes": "bitsandbytes",
    "aqlm": "aqlm",
    "lm-eval": "lm_eval",
    "gguf": "gguf",
    "vllm": "vllm",
    "mlx-lm": "mlx_lm",
}

_I_DLM_PLUGIN_NAMES = {"i_dlm", "i_dlm_isd"}

# Actionable hints keyed by detect.DetectionStatus. Kept as plain strings so
# the module stays importable on CPU-only hosts where torch is absent.
_DETECT_HINTS: dict[hw_detect.DetectionStatus, str] = {
    hw_detect.DetectionStatus.TORCH_MISSING: (
        "install a torch build for your hardware, e.g. `uv pip install torch` "
        "(GB10 / sm_121 needs the NVIDIA-provided nightly / preview wheel)"
    ),
    hw_detect.DetectionStatus.CUDA_UNAVAILABLE: (
        "check the NVIDIA driver and `CUDA_VISIBLE_DEVICES`; run "
        "`python -c 'import torch; print(torch.version.cuda, torch.cuda.device_count())'`"
    ),
    hw_detect.DetectionStatus.UNRECOGNIZED_CC: (
        "add the compute capability to `_CC_TO_ARCH` in forgekit/hw/detect.py "
        "(or open an issue with the CC tuple)"
    ),
}


@app.command()
def version() -> None:
    """Print the forgekit version."""
    console.print(f"forgekit {__version__}")


@app.command("run")
def run_cmd(
    recipe_path: Annotated[Path, typer.Argument(exists=True, readable=True, help="Path to a recipe YAML file.")],
    work_dir: Annotated[
        Path,
        typer.Option("--work-dir", "-w", help="Directory for intermediate artifacts."),
    ] = Path("outputs"),
) -> None:
    """Execute a recipe end-to-end."""
    recipe = load_recipe(recipe_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    hw = hw_detect.detect()
    console.print(f"[bold]Recipe:[/] {recipe.name}")
    console.print(f"[bold]Model:[/]  {recipe.model}")
    console.print(f"[bold]HW:[/]     {hw.arch} ({hw.vram_gb:.1f} GB)")

    ctx = StageContext(
        recipe_name=recipe.name,
        model_path=recipe.model,
        work_dir=work_dir,
        hw=hw,
    )

    ctx = _run_stage("data", recipe.data.kind, recipe.data.config, ctx, method="prepare")
    ctx = _run_stage("trainer", recipe.trainer.kind, recipe.trainer.config, ctx, method="train")
    for spec in recipe.compressors:
        ctx = _run_stage("compressor", spec.kind, spec.config, ctx, method="compress")
    if recipe.evaluator is not None:
        ctx = _run_stage(
            "evaluator", recipe.evaluator.kind, recipe.evaluator.config, ctx, method="evaluate"
        )
    if recipe.exporter is not None:
        ctx = _run_stage(
            "exporter", recipe.exporter.kind, recipe.exporter.config, ctx, method="export"
        )

    console.print("[green]✓ recipe complete[/]")


def _run_stage(
    stage_kind: str,
    name: str,
    config: dict[str, object],
    ctx: StageContext,
    method: str,
) -> StageContext:
    if stage_kind == "data":
        # `data` is not yet an entry_points group — data loading lives inside
        # trainer plugins for v0. Left as a no-op stub call site.
        console.print(f"[dim]data stage {name!r} (no-op in scaffold)[/]")
        return ctx
    console.print(f"→ {stage_kind}: [cyan]{name}[/]")
    plugin_cls = get(stage_kind, name)
    plugin = plugin_cls()
    ctx.stage_config = dict(config)
    try:
        result = getattr(plugin, method)(ctx)
    except NotImplementedError as exc:
        console.print(f"[red]{stage_kind} plugin {name!r} is not implemented yet:[/] {exc}")
        raise typer.Exit(code=2) from exc
    return result if isinstance(result, StageContext) else ctx


@app.command("list-recipes")
def list_recipes_cmd(
    recipes_dir: Annotated[
        Path,
        typer.Option("--dir", help="Directory to scan for *.yaml recipes."),
    ] = Path("recipes"),
) -> None:
    """List recipe YAMLs in the given directory."""
    if not recipes_dir.exists():
        console.print(f"[yellow]no recipes dir at {recipes_dir}[/]")
        raise typer.Exit(code=1)
    table = Table(title=f"recipes in {recipes_dir}")
    table.add_column("name")
    table.add_column("path")
    for yaml_path in sorted(recipes_dir.glob("*.y*ml")):
        try:
            spec = load_recipe(yaml_path)
            table.add_row(spec.name, str(yaml_path))
        except Exception as exc:
            table.add_row(f"[red]invalid[/] ({exc.__class__.__name__})", str(yaml_path))
    console.print(table)


@app.command("list-plugins")
def list_plugins_cmd() -> None:
    """List every registered plugin across all four stage groups."""
    for stage, group in STAGE_GROUPS.items():
        plugins = discover(group)
        table = Table(title=f"{stage} plugins ({group})")
        table.add_column("name")
        table.add_column("class")
        for name in sorted(plugins):
            cls = plugins[name]
            label = f"{cls.__module__}:{cls.__qualname__}"
            if name in _I_DLM_PLUGIN_NAMES and not _i_dlm_accepted():
                label = f"{label}  [yellow]⚠ license gate[/]"
            table.add_row(name, label)
        console.print(table)


@app.command()
def doctor() -> None:
    """Report detected hardware and which optional extras are importable."""
    diag = hw_detect.diagnose()
    hw = diag.profile
    console.print(f"[bold]forgekit[/] v{__version__}")
    console.print(f"arch:      {hw.arch}")
    console.print(f"vram:      {hw.vram_gb:.1f} GB")
    console.print(f"unified:   {hw.unified_memory}")
    if diag.status is not hw_detect.DetectionStatus.OK:
        console.print(f"[yellow]detect:[/]    {diag.status.value} — {diag.detail}")
        hint = _DETECT_HINTS.get(diag.status)
        if hint:
            console.print(f"[dim]hint:[/]      {hint}")
    console.print()

    table = Table(title="optional extras")
    table.add_column("extra")
    table.add_column("status")
    for extra, module in _OPTIONAL_EXTRAS.items():
        ok = importlib.util.find_spec(module) is not None
        table.add_row(extra, "[green]✓[/]" if ok else "[dim]✗[/]")
    console.print(table)

    if not _i_dlm_accepted():
        console.print(
            "[yellow]⚠ I-DLM plugins are gated. Set FORGEKIT_ACCEPT_I_DLM_LICENSE=1 "
            "after verifying upstream license terms.[/]"
        )


def _i_dlm_accepted() -> bool:
    return os.environ.get("FORGEKIT_ACCEPT_I_DLM_LICENSE") == "1"


if __name__ == "__main__":
    app()
