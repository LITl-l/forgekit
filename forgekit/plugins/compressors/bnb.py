"""bitsandbytes — 4-bit NF4 / 8-bit weight-only quantization.

Backend: ``bitsandbytes`` + ``transformers``. Install via ``forgekit[bnb]`` /
``forgekit[trl]`` (either extra pulls bitsandbytes in).

The model is loaded under a ``BitsAndBytesConfig`` and re-saved so that
``AutoModelForCausalLM.from_pretrained`` can pick the quantization back up at
inference time (transformers ≥ 4.38 persists the quantization_config to
``config.json`` for us).
"""

from __future__ import annotations

import importlib.util
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from forgekit.stages import StageContext

QuantType = Literal["nf4", "fp4"]
ComputeDtype = Literal["bfloat16", "float16", "float32"]


class BnBConfig(BaseModel):
    """Validated configuration for the bitsandbytes compressor."""

    model_config = ConfigDict(extra="forbid")

    bits: Literal[4, 8] = 4
    quant_type: QuantType = "nf4"
    double_quant: bool = True
    compute_dtype: ComputeDtype = "bfloat16"
    llm_int8_threshold: float = 6.0
    output_subdir: str = "bnb"
    merge_adapter: bool = True


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_backend() -> None:
    missing = [m for m in ("bitsandbytes", "transformers", "torch") if not _module_available(m)]
    if missing:
        raise RuntimeError(
            f"bnb: required modules missing: {', '.join(missing)}. "
            "Install via `forgekit[bnb]` (or any extra that pulls bitsandbytes, e.g. `trl`)."
        )


class BnBCompressor:
    name: ClassVar[str] = "bnb"

    def compress(self, ctx: StageContext) -> StageContext:
        cfg = BnBConfig.model_validate(ctx.stage_config)
        _require_backend()

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        source_model = _resolve_source_model(ctx, merge_adapter=cfg.merge_adapter)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            cfg.compute_dtype
        ]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.bits == 4,
            load_in_8bit=cfg.bits == 8,
            bnb_4bit_quant_type=cfg.quant_type,
            bnb_4bit_use_double_quant=cfg.double_quant,
            bnb_4bit_compute_dtype=dtype,
            llm_int8_threshold=cfg.llm_int8_threshold,
        )

        model = AutoModelForCausalLM.from_pretrained(
            source_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.save_pretrained(str(out_dir))
        AutoTokenizer.from_pretrained(source_model).save_pretrained(str(out_dir))

        ctx.artifacts["bnb_model_path"] = str(out_dir)
        ctx.artifacts["bnb_bits"] = cfg.bits
        ctx.artifacts["bnb_quant_type"] = cfg.quant_type
        ctx.model_path = str(out_dir)
        return ctx


def _resolve_source_model(ctx: StageContext, *, merge_adapter: bool) -> str:
    """Merge a bare qlora adapter into the base (cached) so bnb can quantize it."""
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
