"""HQQ — Half-Quadratic Quantization (Badri & Shaji 2023).

Upstream: https://mobiusml.github.io/hqq_blog/ — ``hqq`` (Apache-2.0).
Install via ``forgekit[hqq]``.

Wraps ``hqq``'s `AutoHQQHFModel.quantize_model`, which rewrites linear
layers in-place with HQQ's group-quantized kernels. The compressed model is
saved via the HQQ serialization format so it can round-trip through
``AutoHQQHFModel.from_quantized`` at inference time.
"""

from __future__ import annotations

import importlib.util
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from forgekit.stages import StageContext

Axis = Literal[0, 1]


class HQQConfig(BaseModel):
    """Validated configuration for the HQQ compressor."""

    model_config = ConfigDict(extra="forbid")

    bits: Literal[1, 2, 3, 4, 8] = 4
    group_size: int = 64
    quant_zero: bool = True
    quant_scale: bool = False
    axis: Axis = 1
    compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device: str | None = None
    output_subdir: str = "hqq"
    merge_adapter: bool = True


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_backend() -> None:
    missing = [m for m in ("hqq", "transformers", "torch") if not _module_available(m)]
    if missing:
        raise RuntimeError(
            f"hqq: required modules missing: {', '.join(missing)}. "
            "Install via `forgekit[hqq]`."
        )


class HQQCompressor:
    name: ClassVar[str] = "hqq"

    def compress(self, ctx: StageContext) -> StageContext:
        cfg = HQQConfig.model_validate(ctx.stage_config)
        _require_backend()

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        source_model = _resolve_source_model(ctx, merge_adapter=cfg.merge_adapter)

        import torch
        from hqq.core.quantize import BaseQuantizeConfig
        from hqq.models.hf.base import AutoHQQHFModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            cfg.compute_dtype
        ]
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoModelForCausalLM.from_pretrained(source_model, torch_dtype=dtype)

        q_config = BaseQuantizeConfig(
            nbits=cfg.bits,
            group_size=cfg.group_size,
            quant_zero=cfg.quant_zero,
            quant_scale=cfg.quant_scale,
            axis=cfg.axis,
        )
        AutoHQQHFModel.quantize_model(
            model,
            quant_config=q_config,
            compute_dtype=dtype,
            device=device,
        )
        AutoHQQHFModel.save_quantized(model, str(out_dir))
        AutoTokenizer.from_pretrained(source_model).save_pretrained(str(out_dir))

        ctx.artifacts["hqq_model_path"] = str(out_dir)
        ctx.artifacts["hqq_bits"] = cfg.bits
        ctx.artifacts["hqq_group_size"] = cfg.group_size
        ctx.model_path = str(out_dir)
        return ctx


def _resolve_source_model(ctx: StageContext, *, merge_adapter: bool) -> str:
    """Merge a bare qlora adapter into its base so HQQ has dense weights to quantize."""
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
