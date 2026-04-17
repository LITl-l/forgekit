"""AQLM — Additive Quantization of Language Models (Egiazarian et al. 2024).

Upstream: https://github.com/Vahe1994/AQLM (Apache-2.0).
Install via ``forgekit[aqlm]``.

AQLM ships a quantization script plus an inference-time runtime (``aqlm``
pypi package). The plugin wraps the upstream script via subprocess — this
keeps forgekit independent of upstream's internal module layout and lets
users drop in newer versions without core churn.
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


class AQLMCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = "allenai/c4"
    name: str | None = "en"
    split: str = "train"
    text_column: str = "text"
    num_samples: int = 1024
    seq_len: int = 4096


class AQLMConfig(BaseModel):
    """Validated configuration for the AQLM compressor."""

    model_config = ConfigDict(extra="forbid")

    num_codebooks: Literal[1, 2, 4, 8, 16] = 1
    nbits_per_codebook: Literal[8, 16] = 16
    in_group_size: int = 8
    out_group_size: int = 1
    calibration: AQLMCalibration = Field(default_factory=AQLMCalibration)
    script_path: str | None = None
    python_executable: str | None = None
    output_subdir: str = "aqlm"
    merge_adapter: bool = True
    extra_args: list[str] = Field(default_factory=list)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_backend() -> None:
    missing = [
        m for m in ("aqlm", "transformers", "torch", "datasets") if not _module_available(m)
    ]
    if missing:
        raise RuntimeError(
            f"aqlm: required modules missing: {', '.join(missing)}. "
            "Install via `forgekit[aqlm]` (see https://github.com/Vahe1994/AQLM)."
        )


def _find_aqlm_script(explicit: str | None) -> str:
    """Locate the upstream AQLM `main.py` / `aqlm_main.py`."""
    if explicit:
        if Path(explicit).is_file():
            return explicit
        raise RuntimeError(f"aqlm: script_path {explicit!r} does not exist.")

    for candidate in ("aqlm_main.py", "aqlm-main", "aqlm"):
        found = shutil.which(candidate)
        if found:
            return found

    if _module_available("aqlm"):
        aqlm = importlib.import_module("aqlm")

        module_file = getattr(aqlm, "__file__", None)
        if module_file:
            pkg_dir = Path(module_file).parent
            for name in ("main.py", "quantize.py", "scripts/main.py"):
                candidate_path = pkg_dir / name
                if candidate_path.is_file():
                    return str(candidate_path)

    raise RuntimeError(
        "aqlm: could not locate the upstream quantization script. "
        "Clone https://github.com/Vahe1994/AQLM and pass its `main.py` via "
        "`script_path` in the recipe config."
    )


class AQLMCompressor:
    name: ClassVar[str] = "aqlm"

    def compress(self, ctx: StageContext) -> StageContext:
        cfg = AQLMConfig.model_validate(ctx.stage_config)
        _require_backend()
        script = _find_aqlm_script(cfg.script_path)

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        source_model = _resolve_source_model(ctx, merge_adapter=cfg.merge_adapter)
        python_exe = cfg.python_executable or sys.executable

        argv = [
            python_exe,
            script,
            source_model,
            cfg.calibration.path,
            "--num_codebooks",
            str(cfg.num_codebooks),
            "--nbits_per_codebook",
            str(cfg.nbits_per_codebook),
            "--in_group_size",
            str(cfg.in_group_size),
            "--out_group_size",
            str(cfg.out_group_size),
            "--save",
            str(out_dir),
            "--calibration_dataset_split",
            cfg.calibration.split,
            "--calibration_column",
            cfg.calibration.text_column,
            "--num_samples",
            str(cfg.calibration.num_samples),
            "--seq_len",
            str(cfg.calibration.seq_len),
        ]
        if cfg.calibration.name is not None:
            argv.extend(["--calibration_dataset_name", cfg.calibration.name])
        argv.extend(cfg.extra_args)

        subprocess.run(argv, check=True)

        ctx.artifacts["aqlm_model_path"] = str(out_dir)
        ctx.artifacts["aqlm_num_codebooks"] = cfg.num_codebooks
        ctx.artifacts["aqlm_nbits_per_codebook"] = cfg.nbits_per_codebook
        ctx.model_path = str(out_dir)
        return ctx


def _resolve_source_model(ctx: StageContext, *, merge_adapter: bool) -> str:
    """Merge a bare qlora adapter into the base so AQLM has dense weights."""
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
