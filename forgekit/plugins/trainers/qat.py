"""QAT — Quantization-Aware Training via torchao fake-quant + HF Trainer.

Backends: ``torchao`` (primary) and ``torchtune`` (reserved for a follow-up).
Install via ``forgekit[qat]`` once that extra is pinned — today torchao ships
on PyPI and is the simplest path.

Applies ``quantize_(model, IntXWeightOnlyConfig)`` so torchao inserts fake
quantization observers, then fine-tunes with HF ``Trainer``. The resulting
checkpoint contains fused quantized weights ready for int4/int8 inference.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from forgekit.stages import StageContext

Backend = Literal["torchao", "torchtune", "auto"]


class QATDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="HF dataset path.")
    split: str = "train"
    text_column: str = "text"
    max_samples: int | None = None


class QATConfig(BaseModel):
    """Validated configuration for the QAT trainer."""

    model_config = ConfigDict(extra="forbid")

    backend: Backend = "auto"
    dataset: QATDataset
    target_bits: Literal[4, 8] = 4
    group_size: int = 32
    lr: float = 2e-5
    steps: int = 1000
    micro_batch_size: int | None = None
    grad_accum: int = 1
    seq_len: int | None = None
    warmup_steps: int = 100
    weight_decay: float = 0.0
    seed: int = 42
    bf16: bool = True
    output_subdir: str = "qat"


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _resolve_backend(requested: Backend) -> Literal["torchao", "torchtune"]:
    if requested == "auto":
        if _module_available("torchao"):
            return "torchao"
        if _module_available("torchtune"):
            return "torchtune"
        raise RuntimeError(
            "qat: no backend available. Install `forgekit[qat]` (torchao) or "
            "`forgekit[torchtune]`."
        )
    if requested == "torchao" and not _module_available("torchao"):
        raise RuntimeError("qat: backend='torchao' requested but `torchao` is not installed.")
    if requested == "torchtune" and not _module_available("torchtune"):
        raise RuntimeError(
            "qat: backend='torchtune' requested but `torchtune` is not installed."
        )
    return requested


class QATTrainer:
    name: ClassVar[str] = "qat"

    def train(self, ctx: StageContext) -> StageContext:
        cfg = QATConfig.model_validate(ctx.stage_config)
        backend = _resolve_backend(cfg.backend)

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        micro_bsz = cfg.micro_batch_size or ctx.hw.suggested_micro_batch
        seq_len = cfg.seq_len or ctx.hw.suggested_seq_len

        if backend == "torchtune":
            raise NotImplementedError(
                "qat: torchtune backend is reserved for a follow-up PR. "
                "Use backend='torchao' (or 'auto' with torchao installed)."
            )

        output_path = _train_torchao(ctx, cfg, out_dir, micro_bsz, seq_len)

        ctx.artifacts["qat_output_path"] = str(output_path)
        ctx.artifacts["qat_backend"] = backend
        ctx.artifacts["qat_bits"] = cfg.target_bits
        ctx.model_path = str(output_path)
        return ctx


def _train_torchao(
    ctx: StageContext,
    cfg: QATConfig,
    out_dir: Path,
    micro_bsz: int,
    seq_len: int,
) -> Path:
    import torch
    from datasets import load_dataset
    _torchao_q = importlib.import_module("torchao.quantization")
    Int4WeightOnlyConfig = _torchao_q.Int4WeightOnlyConfig
    Int8WeightOnlyConfig = _torchao_q.Int8WeightOnlyConfig
    quantize_ = _torchao_q.quantize_
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(ctx.model_path, torch_dtype=dtype)

    quant_cfg: Any = (
        Int4WeightOnlyConfig(group_size=cfg.group_size)
        if cfg.target_bits == 4
        else Int8WeightOnlyConfig()
    )
    quantize_(model, quant_cfg)

    tokenizer = AutoTokenizer.from_pretrained(ctx.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(cfg.dataset.path, split=cfg.dataset.split)
    if cfg.dataset.max_samples is not None:
        ds = ds.select(range(min(cfg.dataset.max_samples, len(ds))))
    if cfg.dataset.text_column not in ds.column_names:
        raise ValueError(
            f"qat: text_column {cfg.dataset.text_column!r} not in columns {ds.column_names}"
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

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            per_device_train_batch_size=micro_bsz,
            gradient_accumulation_steps=cfg.grad_accum,
            max_steps=cfg.steps,
            learning_rate=cfg.lr,
            warmup_steps=cfg.warmup_steps,
            weight_decay=cfg.weight_decay,
            bf16=cfg.bf16,
            report_to=[],
            seed=cfg.seed,
            logging_steps=10,
        ),
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
