"""DPO — Direct Preference Optimization (Rafailov et al. 2023).

Backend: ``trl`` (``DPOTrainer``). Install via ``forgekit[trl]``.

Trains on a preference dataset with ``prompt`` / ``chosen`` / ``rejected``
columns. Optional 4-bit NF4 quantization + LoRA for QLoRA-style DPO on
memory-constrained setups.
"""

from __future__ import annotations

import importlib.util
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from forgekit.stages import StageContext

LossType = Literal["sigmoid", "hinge", "ipo", "kto_pair"]


class DPODataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="HF dataset path, e.g. 'argilla/ultrafeedback-binarized'.")
    split: str = "train"
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    max_samples: int | None = None


class DPOConfigModel(BaseModel):
    """Validated configuration for the DPO trainer."""

    model_config = ConfigDict(extra="forbid")

    dataset: DPODataset
    lr: float = 5e-7
    steps: int = 500
    beta: float = 0.1
    loss_type: LossType = "sigmoid"
    micro_batch_size: int | None = None
    grad_accum: int = 1
    max_length: int | None = None
    max_prompt_length: int | None = None
    load_in_4bit: bool = False
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] | None = None
    warmup_steps: int = 5
    weight_decay: float = 0.0
    seed: int = 42
    output_subdir: str = "dpo"


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_backend() -> None:
    missing = [
        m for m in ("trl", "transformers", "peft", "torch") if not _module_available(m)
    ]
    if missing:
        raise RuntimeError(
            f"dpo: required modules missing: {', '.join(missing)}. "
            "Install via `forgekit[trl]`."
        )


class DPOTrainer:
    name: ClassVar[str] = "dpo"

    def train(self, ctx: StageContext) -> StageContext:
        cfg = DPOConfigModel.model_validate(ctx.stage_config)
        _require_backend()

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        micro_bsz = cfg.micro_batch_size or ctx.hw.suggested_micro_batch

        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig as TRLDPOConfig
        from trl import DPOTrainer as TRLDPOTrainer

        load_kwargs: dict[str, Any] = {}
        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(ctx.model_path, **load_kwargs)
        if cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        if cfg.use_lora:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=cfg.lora_r,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=cfg.target_modules
                    or [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                ),
            )

        tokenizer = AutoTokenizer.from_pretrained(ctx.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ds = load_dataset(cfg.dataset.path, split=cfg.dataset.split)
        if cfg.dataset.max_samples is not None:
            ds = ds.select(range(min(cfg.dataset.max_samples, len(ds))))
        for col in (cfg.dataset.prompt_column, cfg.dataset.chosen_column, cfg.dataset.rejected_column):
            if col not in ds.column_names:
                raise ValueError(
                    f"dpo: column {col!r} missing from dataset columns {ds.column_names}"
                )

        args = TRLDPOConfig(
            output_dir=str(out_dir / "checkpoints"),
            per_device_train_batch_size=micro_bsz,
            gradient_accumulation_steps=cfg.grad_accum,
            max_steps=cfg.steps,
            learning_rate=cfg.lr,
            warmup_steps=cfg.warmup_steps,
            weight_decay=cfg.weight_decay,
            beta=cfg.beta,
            loss_type=cfg.loss_type,
            max_length=cfg.max_length,
            max_prompt_length=cfg.max_prompt_length,
            logging_steps=10,
            seed=cfg.seed,
            bf16=True,
            report_to=[],
        )

        trainer = TRLDPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=ds,
        )
        trainer.train()
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        ctx.artifacts["dpo_output_path"] = str(out_dir)
        ctx.artifacts["dpo_backend"] = "trl"
        if cfg.use_lora:
            ctx.artifacts["qlora_adapter_path"] = str(out_dir)
            ctx.artifacts["qlora_base_model"] = ctx.model_path
        ctx.model_path = str(out_dir)
        return ctx
