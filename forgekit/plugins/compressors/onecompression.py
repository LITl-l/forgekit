"""OneCompression — Fujitsu Research mixed-precision PTQ + AutoBit search.

Upstream: https://github.com/FujitsuResearch/OneCompression (MIT).
Install via ``forgekit[onecompression]``. This is forgekit's P0 differentiator
— "AutoBit" selects per-layer bit widths to hit a target average.

The plugin wraps the upstream quantization script via subprocess so forgekit
stays decoupled from its internal module layout. CLI flag names are the
reasonable defaults from the OneCompression README; ``extra_args`` lets
recipe authors override any of them if the upstream drifts.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from forgekit.stages import StageContext

Mode = Literal["autobit", "uniform", "mixed"]


class OneCompressionCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = "wikitext"
    name: str | None = "wikitext-2-raw-v1"
    split: str = "train"
    text_column: str = "text"
    num_samples: int = 128
    seq_len: int = 2048


class OneCompressionConfig(BaseModel):
    """Validated configuration for the OneCompression compressor."""

    model_config = ConfigDict(extra="forbid")

    mode: Mode = "autobit"
    target_avg_bits: float = 4.0
    min_bits: Literal[2, 3, 4] = 2
    max_bits: Literal[4, 8] = 8
    calibration: OneCompressionCalibration = Field(default_factory=OneCompressionCalibration)
    script_path: str | None = None
    python_executable: str | None = None
    output_subdir: str = "onecompression"
    merge_adapter: bool = True
    extra_args: list[str] = Field(default_factory=list)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_backend() -> None:
    missing = [
        m
        for m in ("onecompression", "transformers", "torch", "datasets")
        if not _module_available(m)
    ]
    if missing:
        raise RuntimeError(
            f"onecompression: required modules missing: {', '.join(missing)}. "
            "Install via `forgekit[onecompression]` "
            "(https://github.com/FujitsuResearch/OneCompression)."
        )


def _find_onecompression_script(explicit: str | None) -> str:
    """Locate the upstream OneCompression quantization entry point."""
    if explicit:
        if Path(explicit).is_file():
            return explicit
        raise RuntimeError(
            f"onecompression: script_path {explicit!r} does not exist."
        )

    for candidate in ("onecompression_quantize.py", "onecompression", "onecompression-quantize"):
        found = shutil.which(candidate)
        if found:
            return found

    if _module_available("onecompression"):
        onecompression = importlib.import_module("onecompression")

        module_file = getattr(onecompression, "__file__", None)
        if module_file:
            pkg_dir = Path(module_file).parent
            for rel in (
                "scripts/quantize.py",
                "scripts/main.py",
                "quantize.py",
                "main.py",
            ):
                candidate_path = pkg_dir / rel
                if candidate_path.is_file():
                    return str(candidate_path)

    raise RuntimeError(
        "onecompression: could not locate the upstream quantization script. "
        "Point `script_path` at the script bundled with the Fujitsu upstream."
    )


class OneCompressionCompressor:
    name: ClassVar[str] = "onecompression"

    def compress(self, ctx: StageContext) -> StageContext:
        cfg = OneCompressionConfig.model_validate(ctx.stage_config)
        _require_backend()
        script = _find_onecompression_script(cfg.script_path)

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        source_model = _resolve_source_model(ctx, merge_adapter=cfg.merge_adapter)
        python_exe = cfg.python_executable or sys.executable

        argv = [
            python_exe,
            script,
            "--model",
            source_model,
            "--mode",
            cfg.mode,
            "--target-avg-bits",
            str(cfg.target_avg_bits),
            "--min-bits",
            str(cfg.min_bits),
            "--max-bits",
            str(cfg.max_bits),
            "--calibration-dataset",
            cfg.calibration.path,
            "--calibration-split",
            cfg.calibration.split,
            "--calibration-column",
            cfg.calibration.text_column,
            "--num-samples",
            str(cfg.calibration.num_samples),
            "--seq-len",
            str(cfg.calibration.seq_len),
            "--output",
            str(out_dir),
        ]
        if cfg.calibration.name is not None:
            argv.extend(["--calibration-name", cfg.calibration.name])
        argv.extend(cfg.extra_args)

        subprocess.run(argv, check=True)

        ctx.artifacts["onecompression_model_path"] = str(out_dir)
        ctx.artifacts["onecompression_mode"] = cfg.mode
        ctx.artifacts["onecompression_avg_bits"] = cfg.target_avg_bits
        ctx.model_path = str(out_dir)
        return ctx


def _resolve_source_model(ctx: StageContext, *, merge_adapter: bool) -> str:
    """Merge a bare qlora adapter into its base so OneCompression has dense weights."""
    adapter_path = ctx.artifacts.get("qlora_adapter_path")
    base_model = ctx.artifacts.get("qlora_base_model")
    if not adapter_path or not base_model or not merge_adapter:
        return ctx.model_path

    merged_existing = ctx.artifacts.get("qlora_merged_path")
    if merged_existing:
        return str(merged_existing)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    merged_dir = ctx.work_dir / "qlora_merged"
    if not merged_dir.exists():
        merged_dir.mkdir(parents=True)
        base = AutoModelForCausalLM.from_pretrained(base_model)
        merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
        merged.save_pretrained(str(merged_dir))
    ctx.artifacts["qlora_merged_path"] = str(merged_dir)
    return str(merged_dir)
