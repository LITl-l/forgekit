"""AWQ — Lin et al. 2023 (https://arxiv.org/abs/2306.00978).

Backend: ``llmcompressor`` (Apache-2.0). Install via ``forgekit[awq]``.

Backend migration (2026-08)
---------------------------
This plugin previously targeted ``autoawq``. That project is **officially
deprecated and unmaintained**; its own README directs users to the vLLM
project's ``llm-compressor``, which absorbed the algorithm. ``transformers``
tracks the same move.

The rewrite changes the output format as well as the API. ``llm-compressor``
emits a **compressed-tensors** checkpoint rather than an AutoAWQ one, which is
what vLLM consumes natively — so the ``vllm`` exporter downstream gets a
directly loadable directory instead of a format vLLM has to special-case.

Two config fields did not survive the move:

* ``version`` (``gemm`` / ``gemv`` / ``gemv_fast``) selected an AutoAWQ CUDA
  kernel at quantization time. compressed-tensors defers kernel choice to the
  serving runtime, so there is nothing to pick here.
* ``bits`` stays 4 — AWQ's search is defined for 4-bit — but symmetry is now
  expressed through the quantization *scheme* rather than a ``zero_point`` flag.
"""

from __future__ import annotations

import importlib.util
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from forgekit.stages import StageContext


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
    symmetric: bool = Field(
        default=False,
        description=(
            "False (default) selects the asymmetric W4A16_ASYM scheme, which is "
            "what the AWQ paper evaluates. True selects symmetric W4A16."
        ),
    )
    ignore: list[str] = Field(
        default_factory=lambda: ["lm_head"],
        description="Module names left in full precision. lm_head is standard.",
    )
    calibration: AWQCalibrationDataset = Field(default_factory=AWQCalibrationDataset)
    output_subdir: str = "awq"
    merge_adapter: bool = True

    @property
    def scheme(self) -> str:
        """compressed-tensors scheme name for this bit/symmetry combination."""
        return "W4A16" if self.symmetric else "W4A16_ASYM"


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_backend() -> None:
    missing = [
        m
        for m in ("llmcompressor", "transformers", "torch", "datasets")
        if not _module_available(m)
    ]
    if missing:
        raise RuntimeError(
            f"awq: required modules missing: {', '.join(missing)}. "
            "Install via `forgekit[awq]`. (forgekit migrated off `autoawq`, "
            "which is deprecated in favour of `llmcompressor`.)"
        )


class AWQCompressor:
    name: ClassVar[str] = "awq"

    def compress(self, ctx: StageContext) -> StageContext:
        cfg = AWQConfig.model_validate(ctx.stage_config)
        _require_backend()

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        source_model = _resolve_source_model(ctx, merge_adapter=cfg.merge_adapter)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source_model)
        model = AutoModelForCausalLM.from_pretrained(source_model, torch_dtype="auto")

        dataset = _build_calibration_dataset(tokenizer, cfg)

        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier
        from llmcompressor.modifiers.quantization import QuantizationModifier

        recipe = [
            AWQModifier(),
            QuantizationModifier(
                targets=["Linear"], scheme=cfg.scheme, ignore=list(cfg.ignore)
            ),
        ]
        oneshot(
            model=model,
            dataset=dataset,
            recipe=recipe,
            max_seq_length=cfg.calibration.seq_len,
            num_calibration_samples=cfg.calibration.num_samples,
        )

        model.save_pretrained(str(out_dir), save_compressed=True)
        tokenizer.save_pretrained(str(out_dir))

        ctx.artifacts["awq_model_path"] = str(out_dir)
        ctx.artifacts["awq_bits"] = cfg.bits
        ctx.artifacts["awq_scheme"] = cfg.scheme
        # compressed-tensors output loads directly in vLLM; the exporter reads this.
        ctx.artifacts["quantized_model_path"] = str(out_dir)
        ctx.artifacts["quantized_format"] = "compressed-tensors"
        ctx.model_path = str(out_dir)
        return ctx


def _resolve_source_model(ctx: StageContext, *, merge_adapter: bool) -> str:
    """Merge a preceding LoRA adapter into its base, once, if one was produced."""
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


def _build_calibration_dataset(tokenizer: Any, cfg: AWQConfig) -> Any:
    """Load and tokenize the calibration slice.

    ``llmcompressor.oneshot`` expects a *pre-tokenized* dataset — unlike
    ``gptqmodel``, which takes raw strings. Getting this wrong surfaces as an
    opaque collator error deep inside the modifier, so it is done explicitly.
    """
    from datasets import load_dataset

    spec = cfg.calibration
    split = f"{spec.split}[:{spec.num_samples}]"
    ds = (
        load_dataset(spec.path, spec.name, split=split)
        if spec.name is not None
        else load_dataset(spec.path, split=split)
    )

    if spec.text_column not in ds.column_names:
        raise ValueError(
            f"awq: calibration text_column {spec.text_column!r} not in columns "
            f"{ds.column_names}"
        )

    def _tokenize(sample: dict[str, Any]) -> Any:
        return tokenizer(
            sample[spec.text_column],
            padding=False,
            max_length=spec.seq_len,
            truncation=True,
            add_special_tokens=True,
        )

    return ds.map(_tokenize, remove_columns=ds.column_names)
