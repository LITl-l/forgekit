"""EleutherAI ``lm-evaluation-harness`` wrapper.

Upstream: https://github.com/EleutherAI/lm-evaluation-harness (MIT).
Install via ``forgekit[lm-eval]``.

Calls ``lm_eval.simple_evaluate`` against ``ctx.model_path`` for a user-supplied
list of tasks (e.g. ``hellaswag``, ``arc_easy``, ``mmlu``). The full JSON
results are persisted and a summary is surfaced on ``ctx.artifacts``.
"""

from __future__ import annotations

import importlib.util
import json
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from forgekit.stages import StageContext

ModelType = Literal["hf", "hf-auto", "vllm"]


class LMEvalConfig(BaseModel):
    """Validated configuration for the lm-evaluation-harness evaluator."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[str] = Field(min_length=1, description="lm_eval task names.")
    num_fewshot: int | None = None
    limit: int | float | None = None
    batch_size: int | str = "auto"
    model_type: ModelType = "hf"
    device: str | None = None
    trust_remote_code: bool = False
    extra_model_args: dict[str, str] = Field(default_factory=dict)
    output_subdir: str = "lm_eval"
    merge_adapter: bool = True


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_backend() -> None:
    missing = [
        m for m in ("lm_eval", "transformers", "torch") if not _module_available(m)
    ]
    if missing:
        raise RuntimeError(
            f"lm_eval: required modules missing: {', '.join(missing)}. "
            "Install via `forgekit[lm-eval]`."
        )


def _build_model_args(source_model: str, cfg: LMEvalConfig) -> str:
    """Build the ``model_args`` string passed to ``lm_eval.simple_evaluate``."""
    parts = [f"pretrained={source_model}", f"trust_remote_code={str(cfg.trust_remote_code).lower()}"]
    for k, v in cfg.extra_model_args.items():
        parts.append(f"{k}={v}")
    return ",".join(parts)


class LMEvalHarnessEvaluator:
    name: ClassVar[str] = "lm_eval_harness"

    def evaluate(self, ctx: StageContext) -> StageContext:
        cfg = LMEvalConfig.model_validate(ctx.stage_config)
        _require_backend()

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        source_model = _resolve_model_path(ctx, merge_adapter=cfg.merge_adapter)
        model_args = _build_model_args(source_model, cfg)

        from lm_eval import simple_evaluate

        results = simple_evaluate(
            model=cfg.model_type,
            model_args=model_args,
            tasks=cfg.tasks,
            num_fewshot=cfg.num_fewshot,
            batch_size=cfg.batch_size,
            limit=cfg.limit,
            device=cfg.device,
        )

        report_path = out_dir / "lm_eval_results.json"
        report_path.write_text(json.dumps(results, indent=2, default=str) + "\n")

        task_summary: dict[str, Any] = {}
        for task in cfg.tasks:
            entry = results.get("results", {}).get(task) if isinstance(results, dict) else None
            if entry is not None:
                task_summary[task] = entry

        ctx.artifacts["lm_eval_results"] = task_summary
        ctx.artifacts["lm_eval_report_path"] = str(report_path)
        return ctx


def _resolve_model_path(ctx: StageContext, *, merge_adapter: bool) -> str:
    """Pick a loadable checkpoint path, merging a bare qlora adapter if needed."""
    if ctx.artifacts.get("gptq_model_path") or ctx.artifacts.get("awq_model_path"):
        return ctx.model_path

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
