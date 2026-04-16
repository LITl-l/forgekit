"""GRPO — Group Relative Policy Optimization (Shao et al. 2024 — DeepSeek).

Backend: ``trl`` (``GRPOTrainer``). Install via ``forgekit[trl]``.

Reward is supplied by dotted path (``module:callable``) so it can be specified
entirely from the recipe YAML without touching python source. The reward
callable receives ``(prompts, completions, **kwargs)`` and returns a
``list[float]`` of per-sample rewards, matching TRL's GRPO contract.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from forgekit.stages import StageContext


class GRPODataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="HF dataset path with a prompt column.")
    split: str = "train"
    prompt_column: str = "prompt"
    max_samples: int | None = None


class GRPOConfigModel(BaseModel):
    """Validated configuration for the GRPO trainer."""

    model_config = ConfigDict(extra="forbid")

    dataset: GRPODataset
    reward_function: str = Field(
        description="Dotted path to a reward callable, e.g. 'my_module:length_reward'."
    )
    lr: float = 1e-6
    steps: int = 500
    num_generations: int = 8
    max_prompt_length: int = 512
    max_completion_length: int = 256
    beta: float = 0.04
    temperature: float = 0.9
    micro_batch_size: int | None = None
    grad_accum: int = 1
    load_in_4bit: bool = False
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] | None = None
    warmup_steps: int = 5
    weight_decay: float = 0.0
    seed: int = 42
    output_subdir: str = "grpo"


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_backend() -> None:
    missing = [
        m for m in ("trl", "transformers", "peft", "torch") if not _module_available(m)
    ]
    if missing:
        raise RuntimeError(
            f"grpo: required modules missing: {', '.join(missing)}. "
            "Install via `forgekit[trl]`."
        )


def _load_reward_function(dotted: str) -> Callable[..., list[float]]:
    """Parse ``'module.sub:callable'`` and return the callable."""
    if ":" not in dotted:
        raise ValueError(
            f"grpo: reward_function must be 'module:name', got {dotted!r}."
        )
    module_path, attr = dotted.split(":", 1)
    module_path = module_path.strip()
    attr = attr.strip()
    if not module_path or not attr:
        raise ValueError(
            f"grpo: reward_function must be 'module:name', got {dotted!r}."
        )
    module = importlib.import_module(module_path)
    try:
        fn = getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(
            f"grpo: reward callable {attr!r} not found in module {module_path!r}."
        ) from exc
    if not callable(fn):
        raise ValueError(
            f"grpo: reward {dotted!r} resolved but is not callable."
        )
    return fn  # type: ignore[no-any-return]


class GRPOTrainer:
    name: ClassVar[str] = "grpo"

    def train(self, ctx: StageContext) -> StageContext:
        cfg = GRPOConfigModel.model_validate(ctx.stage_config)
        _require_backend()

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        reward_fn = _load_reward_function(cfg.reward_function)
        micro_bsz = cfg.micro_batch_size or ctx.hw.suggested_micro_batch

        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig as TRLGRPOConfig
        from trl import GRPOTrainer as TRLGRPOTrainer

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
        if cfg.dataset.prompt_column not in ds.column_names:
            raise ValueError(
                f"grpo: prompt column {cfg.dataset.prompt_column!r} not in dataset "
                f"columns {ds.column_names}"
            )

        args = TRLGRPOConfig(
            output_dir=str(out_dir / "checkpoints"),
            per_device_train_batch_size=micro_bsz,
            gradient_accumulation_steps=cfg.grad_accum,
            max_steps=cfg.steps,
            learning_rate=cfg.lr,
            num_generations=cfg.num_generations,
            max_prompt_length=cfg.max_prompt_length,
            max_completion_length=cfg.max_completion_length,
            beta=cfg.beta,
            temperature=cfg.temperature,
            warmup_steps=cfg.warmup_steps,
            weight_decay=cfg.weight_decay,
            logging_steps=10,
            seed=cfg.seed,
            bf16=True,
            report_to=[],
        )

        trainer = TRLGRPOTrainer(
            model=model,
            reward_funcs=[reward_fn],
            args=args,
            train_dataset=ds,
            tokenizer=tokenizer,
        )
        trainer.train()
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        ctx.artifacts["grpo_output_path"] = str(out_dir)
        if cfg.use_lora:
            ctx.artifacts["qlora_adapter_path"] = str(out_dir)
            ctx.artifacts["qlora_base_model"] = ctx.model_path
        ctx.model_path = str(out_dir)
        return ctx
