"""SFT — supervised fine-tuning (full-precision, LoRA, or 8-bit + LoRA).

Backends: ``trl`` (``SFTTrainer``), ``unsloth``. Install via ``forgekit[trl]``
or ``forgekit[unsloth]``.

The trainer hands off to TRL's ``SFTTrainer`` (or Unsloth's fast path if
requested). Unlike the QLoRA plugin, SFT defaults to dense weights — opt in
to ``use_lora`` for adapter training or ``load_in_8bit`` for bnb 8-bit runtime.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from forgekit.stages import StageContext

Backend = Literal["trl", "unsloth", "auto"]


class SFTDataset(BaseModel):
    """Inline dataset reference for v0 (data stage is a no-op in the scaffold)."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="HF dataset path, e.g. 'tatsu-lab/alpaca'.")
    split: str = "train"
    text_column: str = "text"
    max_samples: int | None = None


class SFTConfig(BaseModel):
    """Validated configuration for the SFT trainer."""

    model_config = ConfigDict(extra="forbid")

    backend: Backend = "auto"
    dataset: SFTDataset
    lr: float = 2e-5
    steps: int = 500
    micro_batch_size: int | None = None
    grad_accum: int = 1
    seq_len: int | None = None
    load_in_8bit: bool = False
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] | None = None
    warmup_steps: int = 5
    weight_decay: float = 0.0
    seed: int = 42
    output_subdir: str = "sft"


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _resolve_backend(requested: Backend) -> Literal["trl", "unsloth"]:
    if requested == "auto":
        if _module_available("unsloth"):
            return "unsloth"
        if _module_available("trl"):
            return "trl"
        raise RuntimeError(
            "sft: no backend available. Install `forgekit[unsloth]` or `forgekit[trl]`."
        )
    if requested == "unsloth" and not _module_available("unsloth"):
        raise RuntimeError("sft: backend='unsloth' requested but `unsloth` is not installed.")
    if requested == "trl" and not _module_available("trl"):
        raise RuntimeError("sft: backend='trl' requested but `trl` is not installed.")
    return requested


class SFTTrainer:
    name: ClassVar[str] = "sft"

    def train(self, ctx: StageContext) -> StageContext:
        cfg = SFTConfig.model_validate(ctx.stage_config)
        backend = _resolve_backend(cfg.backend)

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        micro_bsz = cfg.micro_batch_size or ctx.hw.suggested_micro_batch
        seq_len = cfg.seq_len or ctx.hw.suggested_seq_len

        if backend == "unsloth":
            output_path = _train_unsloth(ctx, cfg, out_dir, micro_bsz, seq_len)
        else:
            output_path = _train_trl(ctx, cfg, out_dir, micro_bsz, seq_len)

        ctx.artifacts["sft_output_path"] = str(output_path)
        ctx.artifacts["sft_backend"] = backend
        if cfg.use_lora:
            ctx.artifacts["qlora_adapter_path"] = str(output_path)
            ctx.artifacts["qlora_base_model"] = ctx.model_path
        ctx.model_path = str(output_path)
        return ctx


def _load_dataset(cfg: SFTConfig) -> Any:
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset.path, split=cfg.dataset.split)
    if cfg.dataset.max_samples is not None:
        ds = ds.select(range(min(cfg.dataset.max_samples, len(ds))))
    if cfg.dataset.text_column not in ds.column_names:
        raise ValueError(
            f"sft: text_column {cfg.dataset.text_column!r} not in dataset columns "
            f"{ds.column_names}"
        )
    return ds


def _train_unsloth(
    ctx: StageContext,
    cfg: SFTConfig,
    out_dir: Path,
    micro_bsz: int,
    seq_len: int,
) -> Path:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ctx.model_path,
        max_seq_length=seq_len,
        load_in_4bit=False,
        load_in_8bit=cfg.load_in_8bit,
    )
    if cfg.use_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules
            or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
        )

    from transformers import TrainingArguments
    from trl import SFTTrainer as TRLSFTTrainer

    trainer = TRLSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=_load_dataset(cfg),
        dataset_text_field=cfg.dataset.text_column,
        max_seq_length=seq_len,
        args=TrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            per_device_train_batch_size=micro_bsz,
            gradient_accumulation_steps=cfg.grad_accum,
            max_steps=cfg.steps,
            learning_rate=cfg.lr,
            warmup_steps=cfg.warmup_steps,
            weight_decay=cfg.weight_decay,
            logging_steps=10,
            seed=cfg.seed,
            bf16=True,
            report_to=[],
        ),
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir


def _train_trl(
    ctx: StageContext,
    cfg: SFTConfig,
    out_dir: Path,
    micro_bsz: int,
    seq_len: int,
) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer as TRLSFTTrainer

    load_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if cfg.load_in_8bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(ctx.model_path, **load_kwargs)

    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if cfg.load_in_8bit:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=cfg.target_modules
                or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )

    tokenizer = AutoTokenizer.from_pretrained(ctx.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = TRLSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=_load_dataset(cfg),
        dataset_text_field=cfg.dataset.text_column,
        max_seq_length=seq_len,
        args=TrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            per_device_train_batch_size=micro_bsz,
            gradient_accumulation_steps=cfg.grad_accum,
            max_steps=cfg.steps,
            learning_rate=cfg.lr,
            warmup_steps=cfg.warmup_steps,
            weight_decay=cfg.weight_decay,
            logging_steps=10,
            seed=cfg.seed,
            bf16=True,
            report_to=[],
        ),
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
