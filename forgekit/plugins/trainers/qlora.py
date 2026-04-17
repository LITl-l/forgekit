"""QLoRA trainer — LoRA over 4-bit NF4 base weights.

Backends: ``unsloth`` (default ≤24 GB), ``trl`` (``SFTTrainer`` + PEFT + bitsandbytes).
Reference: Dettmers et al. 2023, https://arxiv.org/abs/2305.14314.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from forgekit.stages import StageContext

Backend = Literal["unsloth", "trl", "auto"]


class QLoRADataset(BaseModel):
    """Inline dataset reference for v0 (data stage is a no-op in the scaffold).

    Three loading shapes are supported, in priority order:

    1. ``path`` points at a directory containing ``dataset_info.json`` — loaded
       via ``datasets.load_from_disk`` (i.e. a dataset previously produced by
       ``Dataset.save_to_disk``).
    2. ``path`` + optional ``data_files`` — forwarded to ``load_dataset``. Use
       this for HF Hub ids, builder names (``json`` / ``parquet`` / ``csv`` /
       ``text``), or local files whose extension HF can sniff.
    3. ``path`` alone as a bare local file — ``load_dataset(path)`` sniffs the
       extension (``.jsonl`` → json loader, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        description=(
            "HF dataset id ('tatsu-lab/alpaca'), HF builder name ('json'), "
            "or a local file / directory path."
        )
    )
    name: str | None = Field(
        default=None,
        description="HF config name, e.g. 'wikitext-2-raw-v1'.",
    )
    data_files: str | list[str] | dict[str, str | list[str]] | None = Field(
        default=None,
        description=(
            "Local files to load. Accepts a single path, a list of paths / "
            "globs, or a split→paths mapping. Ignored when path is a "
            "save_to_disk directory."
        ),
    )
    split: str = "train"
    text_column: str = "text"
    max_samples: int | None = None


class QLoRAConfig(BaseModel):
    """Validated configuration for the QLoRA trainer."""

    model_config = ConfigDict(extra="forbid")

    backend: Backend = "auto"
    dataset: QLoRADataset
    lr: float = 2e-4
    steps: int = 200
    micro_batch_size: int | None = None
    grad_accum: int = 1
    seq_len: int | None = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] | None = None
    warmup_steps: int = 5
    weight_decay: float = 0.0
    seed: int = 42
    output_subdir: str = "qlora"


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _resolve_backend(requested: Backend) -> Literal["unsloth", "trl"]:
    """Pick a concrete backend. 'auto' prefers unsloth if importable, else trl."""
    if requested == "auto":
        if _module_available("unsloth"):
            return "unsloth"
        if _module_available("trl"):
            return "trl"
        raise RuntimeError(
            "qlora: no backend available. Install `forgekit[unsloth]` or `forgekit[trl]`."
        )
    if requested == "unsloth" and not _module_available("unsloth"):
        raise RuntimeError("qlora: backend='unsloth' requested but `unsloth` is not installed.")
    if requested == "trl" and not _module_available("trl"):
        raise RuntimeError("qlora: backend='trl' requested but `trl` is not installed.")
    return requested


class QLoRATrainer:
    name: ClassVar[str] = "qlora"

    def train(self, ctx: StageContext) -> StageContext:
        cfg = QLoRAConfig.model_validate(ctx.stage_config)
        backend = _resolve_backend(cfg.backend)

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        micro_bsz = cfg.micro_batch_size or ctx.hw.suggested_micro_batch
        seq_len = cfg.seq_len or ctx.hw.suggested_seq_len

        if backend == "unsloth":
            adapter_path = _train_unsloth(ctx, cfg, out_dir, micro_bsz, seq_len)
        else:
            adapter_path = _train_trl(ctx, cfg, out_dir, micro_bsz, seq_len)

        ctx.artifacts["qlora_adapter_path"] = str(adapter_path)
        ctx.artifacts["qlora_base_model"] = ctx.model_path
        ctx.artifacts["qlora_backend"] = backend
        ctx.model_path = str(adapter_path)
        return ctx


def _load_dataset(cfg: QLoRAConfig) -> Any:
    """Lazy HF `datasets` loader. Returns a `Dataset` with `cfg.dataset.text_column`."""
    from datasets import load_dataset, load_from_disk

    spec = cfg.dataset
    local_path = Path(spec.path)
    if local_path.is_dir() and (local_path / "dataset_info.json").exists():
        ds = load_from_disk(str(local_path))
        if hasattr(ds, "keys") and spec.split in ds:
            ds = ds[spec.split]
    else:
        kwargs: dict[str, Any] = {"split": spec.split}
        if spec.name is not None:
            kwargs["name"] = spec.name
        if spec.data_files is not None:
            kwargs["data_files"] = spec.data_files
        ds = load_dataset(spec.path, **kwargs)

    if spec.max_samples is not None:
        ds = ds.select(range(min(spec.max_samples, len(ds))))
    if spec.text_column not in ds.column_names:
        raise ValueError(
            f"qlora: text_column {spec.text_column!r} not in dataset columns "
            f"{ds.column_names}"
        )
    return ds


def _train_unsloth(
    ctx: StageContext,
    cfg: QLoRAConfig,
    out_dir: Path,
    micro_bsz: int,
    seq_len: int,
) -> Path:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ctx.model_path,
        max_seq_length=seq_len,
        load_in_4bit=True,
    )
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

    ds = _load_dataset(cfg)

    from transformers import TrainingArguments
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
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
    cfg: QLoRAConfig,
    out_dir: Path,
    micro_bsz: int,
    seq_len: int,
) -> Path:
    import torch
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        ctx.model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(ctx.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.target_modules
        or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)

    ds = _load_dataset(cfg)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
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
