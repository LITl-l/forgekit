"""Full-weight fine-tuning (no adapter).

Backends: ``transformers`` (``Trainer``) and ``torchtune`` (reserved for a
follow-up PR). Install via ``forgekit[trl]`` — the trl extra pulls in
transformers, datasets, and peft which cover this trainer's runtime needs.

Default bf16 with gradient checkpointing. FSDP opt-in via ``fsdp='full_shard'``
(or any other `transformers.TrainingArguments` FSDP string).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from forgekit.stages import StageContext

Backend = Literal["transformers", "torchtune", "auto"]
Optim = Literal["adamw_torch", "adamw_torch_fused", "adafactor"]


class FullFinetuneDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="HF dataset path.")
    split: str = "train"
    text_column: str = "text"
    max_samples: int | None = None


class FullFinetuneConfig(BaseModel):
    """Validated configuration for the full-finetune trainer."""

    model_config = ConfigDict(extra="forbid")

    backend: Backend = "auto"
    dataset: FullFinetuneDataset
    lr: float = 2e-5
    steps: int = 1000
    epochs: float | None = None
    micro_batch_size: int | None = None
    grad_accum: int = 1
    seq_len: int | None = None
    warmup_steps: int = 100
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True
    optim: Optim = "adamw_torch"
    seed: int = 42
    bf16: bool = True
    fsdp: str | None = None
    output_subdir: str = "full_finetune"


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _resolve_backend(requested: Backend) -> Literal["transformers", "torchtune"]:
    if requested == "auto":
        if _module_available("torchtune"):
            return "torchtune"
        if _module_available("transformers"):
            return "transformers"
        raise RuntimeError(
            "full_finetune: no backend available. Install `forgekit[trl]` "
            "(transformers) or `forgekit[torchtune]`."
        )
    if requested == "torchtune" and not _module_available("torchtune"):
        raise RuntimeError(
            "full_finetune: backend='torchtune' requested but `torchtune` is not installed."
        )
    if requested == "transformers" and not _module_available("transformers"):
        raise RuntimeError(
            "full_finetune: backend='transformers' requested but `transformers` is not installed."
        )
    return requested


class FullFinetuneTrainer:
    name: ClassVar[str] = "full_finetune"

    def train(self, ctx: StageContext) -> StageContext:
        cfg = FullFinetuneConfig.model_validate(ctx.stage_config)
        backend = _resolve_backend(cfg.backend)

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        micro_bsz = cfg.micro_batch_size or ctx.hw.suggested_micro_batch
        seq_len = cfg.seq_len or ctx.hw.suggested_seq_len

        if backend == "torchtune":
            raise NotImplementedError(
                "full_finetune: torchtune backend is reserved for a follow-up PR. "
                "Use backend='transformers' (or 'auto' with torchtune absent)."
            )

        output_path = _train_transformers(ctx, cfg, out_dir, micro_bsz, seq_len)

        ctx.artifacts["full_finetune_output_path"] = str(output_path)
        ctx.artifacts["full_finetune_backend"] = backend
        ctx.model_path = str(output_path)
        return ctx


def _train_transformers(
    ctx: StageContext,
    cfg: FullFinetuneConfig,
    out_dir: Path,
    micro_bsz: int,
    seq_len: int,
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(ctx.model_path, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(ctx.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(cfg.dataset.path, split=cfg.dataset.split)
    if cfg.dataset.max_samples is not None:
        ds = ds.select(range(min(cfg.dataset.max_samples, len(ds))))
    if cfg.dataset.text_column not in ds.column_names:
        raise ValueError(
            f"full_finetune: text_column {cfg.dataset.text_column!r} not in "
            f"columns {ds.column_names}"
        )

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return dict(
            tokenizer(
                batch[cfg.dataset.text_column],
                truncation=True,
                max_length=seq_len,
            )
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    kwargs: dict[str, Any] = dict(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=micro_bsz,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim=cfg.optim,
        bf16=cfg.bf16,
        report_to=[],
        seed=cfg.seed,
        logging_steps=10,
    )
    if cfg.epochs is not None:
        kwargs["num_train_epochs"] = cfg.epochs
    else:
        kwargs["max_steps"] = cfg.steps
    if cfg.fsdp:
        kwargs["fsdp"] = cfg.fsdp

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**kwargs),
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
