"""GPTQ — Frantar et al. 2022, https://arxiv.org/abs/2210.17323.

Backend: ``auto-gptq`` (Apache-2.0). Install via ``forgekit[gptq]``.

Takes a HF model path (possibly the merged output of a prior trainer stage),
runs GPTQ calibration against a small text dataset, and writes a quantized
checkpoint to ``ctx.work_dir/<output_subdir>``.
"""

from __future__ import annotations

import importlib.util
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from forgekit.stages import StageContext


class GPTQCalibrationDataset(BaseModel):
    """Small text dataset used for GPTQ activation statistics."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        default="wikitext",
        description="HF dataset path. Default 'wikitext' with name='wikitext-2-raw-v1'.",
    )
    name: str | None = "wikitext-2-raw-v1"
    split: str = "train"
    text_column: str = "text"
    num_samples: int = 128
    seq_len: int = 2048


class GPTQConfig(BaseModel):
    """Validated configuration for the GPTQ compressor."""

    model_config = ConfigDict(extra="forbid")

    bits: Literal[2, 3, 4, 8] = 4
    group_size: int = 128
    desc_act: bool = False
    sym: bool = True
    damp_percent: float = 0.01
    calibration: GPTQCalibrationDataset = Field(default_factory=GPTQCalibrationDataset)
    output_subdir: str = "gptq"
    merge_adapter: bool = True


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_auto_gptq() -> None:
    if not _module_available("auto_gptq"):
        raise RuntimeError(
            "gptq: `auto-gptq` is not installed. Install via `forgekit[gptq]`."
        )


class GPTQCompressor:
    name: ClassVar[str] = "gptq"

    def compress(self, ctx: StageContext) -> StageContext:
        cfg = GPTQConfig.model_validate(ctx.stage_config)
        _require_auto_gptq()

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        source_model = _resolve_source_model(ctx, merge_adapter=cfg.merge_adapter)

        examples = _build_calibration_examples(source_model, cfg)

        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        quantize_config = BaseQuantizeConfig(
            bits=cfg.bits,
            group_size=cfg.group_size,
            desc_act=cfg.desc_act,
            sym=cfg.sym,
            damp_percent=cfg.damp_percent,
        )
        model = AutoGPTQForCausalLM.from_pretrained(
            source_model, quantize_config=quantize_config
        )
        model.quantize(examples)
        model.save_quantized(str(out_dir))

        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(source_model).save_pretrained(str(out_dir))

        ctx.artifacts["gptq_model_path"] = str(out_dir)
        ctx.artifacts["gptq_bits"] = cfg.bits
        ctx.artifacts["gptq_group_size"] = cfg.group_size
        ctx.model_path = str(out_dir)
        return ctx


def _resolve_source_model(ctx: StageContext, *, merge_adapter: bool) -> str:
    """If the prior stage produced a LoRA adapter, optionally merge it into the base.

    When ``merge_adapter`` is False, returns ``ctx.model_path`` unchanged (GPTQ will
    fail on a bare adapter directory — this path exists for users who pre-merge).
    """
    adapter_path = ctx.artifacts.get("qlora_adapter_path")
    base_model = ctx.artifacts.get("qlora_base_model")
    if not adapter_path or not base_model or not merge_adapter:
        return ctx.model_path

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


def _build_calibration_examples(model_path: str, cfg: GPTQConfig) -> list[dict[str, Any]]:
    """Tokenize a small slice of a text dataset into GPTQ-style calibration batches."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds_kwargs: dict[str, Any] = {"split": cfg.calibration.split}
    if cfg.calibration.name is not None:
        ds = load_dataset(cfg.calibration.path, cfg.calibration.name, **ds_kwargs)
    else:
        ds = load_dataset(cfg.calibration.path, **ds_kwargs)

    col = cfg.calibration.text_column
    if col not in ds.column_names:
        raise ValueError(
            f"gptq: calibration text_column {col!r} not in columns {ds.column_names}"
        )

    examples: list[dict[str, Any]] = []
    for row in ds:
        text = row[col]
        if not text or not isinstance(text, str):
            continue
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.calibration.seq_len,
        )
        if enc["input_ids"].shape[1] < 16:
            continue
        examples.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
        if len(examples) >= cfg.calibration.num_samples:
            break

    if not examples:
        raise RuntimeError("gptq: could not build any calibration examples from dataset.")
    return examples
