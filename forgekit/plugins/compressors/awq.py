"""AWQ — Lin et al. 2023 (https://arxiv.org/abs/2306.00978).

Backend: ``autoawq``. Install via ``forgekit[awq]``.

Wraps `AutoAWQForCausalLM.quantize` with a small calibration slice from
``mit-han-lab/pile-val-backup`` (AWQ's reference set). The saved checkpoint
reloads via `AutoAWQForCausalLM.from_quantized` at inference time.
"""

from __future__ import annotations

import importlib.util
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from forgekit.stages import StageContext

Version = Literal["gemm", "gemv", "gemv_fast"]


class AWQCalibrationDataset(BaseModel):
    """Small text dataset used for AWQ activation statistics."""

    model_config = ConfigDict(extra="forbid")

    path: str = "mit-han-lab/pile-val-backup"
    name: str | None = None
    split: str = "validation"
    text_column: str = "text"
    num_samples: int = 128
    seq_len: int = 512


class AWQConfig(BaseModel):
    """Validated configuration for the AWQ compressor."""

    model_config = ConfigDict(extra="forbid")

    bits: Literal[4] = 4
    group_size: int = 128
    zero_point: bool = True
    version: Version = "gemm"
    calibration: AWQCalibrationDataset = Field(default_factory=AWQCalibrationDataset)
    output_subdir: str = "awq"
    merge_adapter: bool = True


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_backend() -> None:
    missing = [m for m in ("awq", "transformers", "torch") if not _module_available(m)]
    if missing:
        raise RuntimeError(
            f"awq: required modules missing: {', '.join(missing)}. "
            "Install via `forgekit[awq]`."
        )


class AWQCompressor:
    name: ClassVar[str] = "awq"

    def compress(self, ctx: StageContext) -> StageContext:
        cfg = AWQConfig.model_validate(ctx.stage_config)
        _require_backend()

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        source_model = _resolve_source_model(ctx, merge_adapter=cfg.merge_adapter)

        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source_model)
        model = AutoAWQForCausalLM.from_pretrained(source_model)

        quant_config = {
            "w_bit": cfg.bits,
            "q_group_size": cfg.group_size,
            "zero_point": cfg.zero_point,
            "version": cfg.version.upper(),
        }
        calib_data = _build_calibration_strings(cfg)
        model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
        model.save_quantized(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        ctx.artifacts["awq_model_path"] = str(out_dir)
        ctx.artifacts["awq_bits"] = cfg.bits
        ctx.artifacts["awq_group_size"] = cfg.group_size
        ctx.model_path = str(out_dir)
        return ctx


def _resolve_source_model(ctx: StageContext, *, merge_adapter: bool) -> str:
    """Merge a bare qlora adapter into its base so AWQ has dense weights to quantize."""
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


def _build_calibration_strings(cfg: AWQConfig) -> list[str]:
    """Tokenize-free calibration slice — AWQ's `calib_data` expects raw strings."""
    from datasets import load_dataset

    ds_kwargs: dict[str, object] = {"split": cfg.calibration.split}
    ds = (
        load_dataset(cfg.calibration.path, cfg.calibration.name, **ds_kwargs)
        if cfg.calibration.name is not None
        else load_dataset(cfg.calibration.path, **ds_kwargs)
    )
    col = cfg.calibration.text_column
    if col not in ds.column_names:
        raise ValueError(
            f"awq: calibration text_column {col!r} not in columns {ds.column_names}"
        )

    out: list[str] = []
    char_budget = cfg.calibration.seq_len * 4  # rough token→char heuristic
    for row in ds:
        text = row[col]
        if not text or not isinstance(text, str):
            continue
        out.append(text[:char_budget])
        if len(out) >= cfg.calibration.num_samples:
            break
    if not out:
        raise RuntimeError("awq: could not build any calibration strings from dataset.")
    return out
