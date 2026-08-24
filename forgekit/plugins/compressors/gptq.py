"""GPTQ — Frantar et al. 2022, https://arxiv.org/abs/2210.17323.

Backend: ``gptqmodel`` (Apache-2.0). Install via ``forgekit[gptq]``.

Backend migration (2026-08)
---------------------------
This plugin previously targeted ``auto-gptq``. That project was **archived on
2025-04-11** and has been removed from ``transformers``; installing it against a
current torch/transformers stack no longer resolves. ``gptqmodel`` is the
maintained successor from the same lineage and is being upstreamed into
transformers / optimum / peft.

It is *not* import-compatible, which is why this is a rewrite rather than a pin
bump. Three things changed:

* ``AutoGPTQForCausalLM.from_pretrained(path, quantize_config=...)``
  → ``GPTQModel.load(path, quant_config)``
* ``model.save_quantized(dir)`` → ``model.save(dir)``
* **Calibration data is now a list of raw strings.** The old API took
  pre-tokenized ``{"input_ids", "attention_mask"}`` tensor dicts; ``gptqmodel``
  tokenizes internally. Passing the old shape fails at quantize time.

Takes a HF model path (possibly the merged output of a prior trainer stage),
runs GPTQ calibration against a small text dataset, and writes a quantized
checkpoint to ``ctx.work_dir/<output_subdir>``.
"""

from __future__ import annotations

import importlib.util
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from forgekit.stages import StageContext

# Rough bytes-per-token ratio for English text, used to trim calibration rows to
# roughly `seq_len` tokens without paying to tokenize them first (gptqmodel
# tokenizes internally, so an exact count here would be wasted work).
_CHARS_PER_TOKEN = 4

# Calibration rows shorter than this contribute almost no activation signal.
_MIN_CALIBRATION_CHARS = 128


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
    batch_size: int = Field(
        default=1,
        description=(
            "Calibration batch size. Raise it to use more VRAM and quantize "
            "faster; 1 is the safe default on 8-12 GB cards."
        ),
    )
    calibration: GPTQCalibrationDataset = Field(default_factory=GPTQCalibrationDataset)
    output_subdir: str = "gptq"
    merge_adapter: bool = True


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_gptqmodel() -> None:
    if not _module_available("gptqmodel"):
        raise RuntimeError(
            "gptq: `gptqmodel` is not installed. Install via `forgekit[gptq]`. "
            "(forgekit migrated off `auto-gptq`, which was archived 2025-04-11.)"
        )


class GPTQCompressor:
    name: ClassVar[str] = "gptq"

    def compress(self, ctx: StageContext) -> StageContext:
        cfg = GPTQConfig.model_validate(ctx.stage_config)
        _require_gptqmodel()

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        source_model = _resolve_source_model(ctx, merge_adapter=cfg.merge_adapter)
        calibration = _build_calibration_texts(cfg)

        from gptqmodel import GPTQConfig as GPTQModelConfig
        from gptqmodel import GPTQModel

        quant_config = GPTQModelConfig(
            bits=cfg.bits,
            group_size=cfg.group_size,
            desc_act=cfg.desc_act,
            sym=cfg.sym,
            damp_percent=cfg.damp_percent,
        )
        model = GPTQModel.load(source_model, quant_config)
        model.quantize(calibration, batch_size=cfg.batch_size)
        model.save(str(out_dir))

        # gptqmodel writes the tokenizer alongside the weights when it loaded
        # one, but a merged-adapter directory can arrive without it. Saving
        # explicitly keeps the output loadable by vLLM / the exporters.
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(source_model).save_pretrained(str(out_dir))

        ctx.artifacts["gptq_model_path"] = str(out_dir)
        ctx.artifacts["gptq_bits"] = cfg.bits
        ctx.artifacts["gptq_group_size"] = cfg.group_size
        ctx.artifacts["quantized_model_path"] = str(out_dir)
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


def _build_calibration_texts(cfg: GPTQConfig) -> list[str]:
    """Collect a small slice of a text dataset as raw calibration strings.

    ``gptqmodel`` tokenizes internally, so this returns plain text — see the
    module docstring for why this differs from the ``auto-gptq`` shape.
    """
    from datasets import load_dataset

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

    max_chars = cfg.calibration.seq_len * _CHARS_PER_TOKEN
    texts: list[str] = []
    for row in ds:
        text = row[col]
        if not isinstance(text, str) or len(text) < _MIN_CALIBRATION_CHARS:
            continue
        texts.append(text[:max_chars])
        if len(texts) >= cfg.calibration.num_samples:
            break

    if not texts:
        raise RuntimeError(
            "gptq: could not build any calibration examples from dataset "
            f"{cfg.calibration.path!r} (need rows of at least "
            f"{_MIN_CALIBRATION_CHARS} characters in column {col!r})."
        )
    return texts
