"""MLX exporter — convert an HF checkpoint to Apple MLX format.

Upstream: ``mlx-lm`` (https://github.com/ml-explore/mlx-lm, MIT).
Install via ``forgekit[mlx]``.

Thin wrapper around ``mlx_lm.convert`` that materializes an MLX-native
checkpoint under ``ctx.work_dir/<output_subdir>``, optionally quantizing
during conversion.
"""

from __future__ import annotations

import importlib.util
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from forgekit.stages import StageContext

DType = Literal["float16", "bfloat16", "float32"]


class MLXConfig(BaseModel):
    """Validated configuration for the MLX exporter."""

    model_config = ConfigDict(extra="forbid")

    quantize: bool = False
    q_bits: Literal[2, 4, 8] = 4
    q_group_size: int = 64
    dtype: DType = "float16"
    output_subdir: str = "mlx"
    merge_adapter: bool = True


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_backend() -> None:
    missing = [m for m in ("mlx_lm", "mlx") if not _module_available(m)]
    if missing:
        raise RuntimeError(
            f"mlx: required modules missing: {', '.join(missing)}. "
            "Install via `forgekit[mlx]`."
        )


class MLXExporter:
    name: ClassVar[str] = "mlx"

    def export(self, ctx: StageContext) -> StageContext:
        cfg = MLXConfig.model_validate(ctx.stage_config)
        _require_backend()

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        source_model = _resolve_source_model(ctx, merge_adapter=cfg.merge_adapter)

        from mlx_lm import convert

        convert(
            hf_path=source_model,
            mlx_path=str(out_dir),
            quantize=cfg.quantize,
            q_bits=cfg.q_bits,
            q_group_size=cfg.q_group_size,
            dtype=cfg.dtype,
        )

        ctx.artifacts["mlx_path"] = str(out_dir)
        ctx.artifacts["mlx_quantized"] = cfg.quantize
        ctx.artifacts["mlx_bits"] = cfg.q_bits if cfg.quantize else None
        ctx.model_path = str(out_dir)
        return ctx


def _resolve_source_model(ctx: StageContext, *, merge_adapter: bool) -> str:
    """Merge a bare qlora adapter so MLX sees dense weights to convert."""
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
