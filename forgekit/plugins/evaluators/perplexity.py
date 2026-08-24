"""Perplexity evaluator — sliding-window NLL with a bootstrap confidence interval.

Default corpus is wikitext-2 (``test`` split). The scoring loop mirrors the
HuggingFace reference (`transformers` docs: "Perplexity of fixed-length models")
— concatenate the text, slide windows of ``seq_len`` with step ``stride``,
mask out overlap tokens from the loss, and report ``exp(mean window NLL)``.

Read this before trusting the number
------------------------------------
Perplexity is a *fluency* measure. It is a poor proxy for the thing most
forgekit users actually care about after compressing a model — whether it still
reasons correctly — and it is systematically most misleading in exactly the
situation forgekit creates.

Empirically, aggressive PTQ degrades reasoning accuracy while *lengthening*
chains of thought, with reported Spearman ρ ≈ -0.73 between accuracy loss and
CoT length growth; quantized models frequently reach the right answer mid-trace
and then talk themselves out of it. None of that moves perplexity much. A
compressed model can hold its perplexity to three decimal places and still lose
several points of task accuracy.

So: use this evaluator as a cheap regression signal — it will catch a genuinely
broken quantization run — and use the ``lm_eval_harness`` evaluator for any
claim about capability.

Two numbers, not one
--------------------
This evaluator reports a bootstrap confidence interval alongside the point
estimate, because a bare perplexity figure invites comparisons it cannot
support. Resampling is over evaluation *windows* (the independent units the
loop actually produces), token-weighted so that a short trailing window does not
count the same as a full one.

The interval is a percentile bootstrap on mean NLL, exponentiated at the
endpoints. ``exp`` is monotonic, so the transformed endpoints remain a valid
interval for perplexity — which is why this is done here rather than
bootstrapping perplexity directly. A percentile bootstrap is used in preference
to a CLT/normal interval because window counts are routinely in the tens, where
normal-approximation intervals are known to be unreliable.

Backends: ``transformers`` + ``datasets`` + ``torch``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forgekit.stages import StageContext

# Alias → (path, name, split, text_column).
_DATASET_ALIASES: dict[str, tuple[str, str | None, str, str]] = {
    "wikitext2": ("wikitext", "wikitext-2-raw-v1", "test", "text"),
    "wikitext103": ("wikitext", "wikitext-103-raw-v1", "test", "text"),
    "ptb": ("ptb_text_only", None, "test", "sentence"),
    "c4": ("allenai/c4", "en", "validation", "text"),
}

# Below this many windows a resampled interval is too coarse to mean anything;
# report the point estimate and say why rather than printing a fake interval.
_MIN_WINDOWS_FOR_CI = 5


class PerplexityConfig(BaseModel):
    """Validated configuration for the perplexity evaluator."""

    model_config = ConfigDict(extra="forbid")

    dataset: str = "wikitext2"
    dataset_path: str | None = None
    dataset_name: str | None = None
    dataset_split: str | None = None
    text_column: str | None = None
    seq_len: int = 2048
    stride: int | None = None
    max_samples: int | None = None
    device: str | None = None
    merge_adapter: bool = True
    bootstrap_resamples: int = Field(
        default=1000,
        ge=0,
        description="Bootstrap iterations for the CI. 0 disables the interval.",
    )
    ci_level: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Confidence level for the reported interval.",
    )
    seed: int = Field(default=0, description="Seed for bootstrap resampling.")
    output_subdir: str = "perplexity"

    @model_validator(mode="after")
    def _validate_dataset(self) -> Self:
        if self.dataset_path is None and self.dataset not in _DATASET_ALIASES:
            raise ValueError(
                f"perplexity: unknown dataset alias {self.dataset!r}. "
                f"Known aliases: {sorted(_DATASET_ALIASES)}. "
                "Pass `dataset_path` to point at a custom HF dataset."
            )
        return self

    def resolved_dataset(self) -> tuple[str, str | None, str, str]:
        """Return the concrete (path, name, split, text_column) to load."""
        if self.dataset_path is not None:
            return (
                self.dataset_path,
                self.dataset_name,
                self.dataset_split or "test",
                self.text_column or "text",
            )
        path, name, split, col = _DATASET_ALIASES[self.dataset]
        return (
            path,
            self.dataset_name if self.dataset_name is not None else name,
            self.dataset_split or split,
            self.text_column or col,
        )


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_backend() -> None:
    missing = [m for m in ("torch", "transformers", "datasets") if not _module_available(m)]
    if missing:
        raise RuntimeError(
            f"perplexity: required modules missing: {', '.join(missing)}. "
            "Install via `forgekit[trl]` or `forgekit[gptq]` — either extra "
            "pulls in transformers, datasets, and torch."
        )


def weighted_mean(values: list[float], weights: list[int]) -> float:
    """Token-weighted mean. Falls back to the unweighted mean if all weights are 0."""
    total_weight = sum(weights)
    if total_weight <= 0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights, strict=True)) / total_weight


def bootstrap_perplexity_ci(
    nlls: list[float],
    weights: list[int],
    *,
    resamples: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float] | None:
    """Percentile-bootstrap CI for perplexity, resampling evaluation windows.

    Returns ``(low, high)`` or ``None`` when there is too little data — see
    ``_MIN_WINDOWS_FOR_CI``. Resamples the (nll, weight) pairs together so the
    token weighting stays coherent within each replicate.
    """
    n = len(nlls)
    if resamples <= 0 or n < _MIN_WINDOWS_FOR_CI:
        return None

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(
            weighted_mean([nlls[i] for i in idx], [weights[i] for i in idx])
        )
    means.sort()

    tail = (1.0 - ci_level) / 2.0
    lo_idx = max(0, min(len(means) - 1, round(tail * (len(means) - 1))))
    hi_idx = max(0, min(len(means) - 1, round((1.0 - tail) * (len(means) - 1))))
    # exp() is monotonic, so exponentiating the NLL endpoints yields a valid
    # percentile interval for perplexity itself.
    return math.exp(means[lo_idx]), math.exp(means[hi_idx])


class PerplexityEvaluator:
    name: ClassVar[str] = "perplexity"

    def evaluate(self, ctx: StageContext) -> StageContext:
        cfg = PerplexityConfig.model_validate(ctx.stage_config)
        _require_backend()

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = _resolve_model_path(ctx, merge_adapter=cfg.merge_adapter)
        nlls, weights = _window_nlls(model_path, cfg)

        mean_nll = weighted_mean(nlls, weights)
        ppl = math.exp(mean_nll)
        ci = bootstrap_perplexity_ci(
            nlls,
            weights,
            resamples=cfg.bootstrap_resamples,
            ci_level=cfg.ci_level,
            seed=cfg.seed,
        )

        path, name, split, col = cfg.resolved_dataset()
        report: dict[str, Any] = {
            "perplexity": ppl,
            "ci_level": cfg.ci_level if ci else None,
            "ci_low": ci[0] if ci else None,
            "ci_high": ci[1] if ci else None,
            "n_windows": len(nlls),
            "n_scored_tokens": sum(weights),
            "bootstrap_resamples": cfg.bootstrap_resamples if ci else 0,
            "model_path": model_path,
            "dataset": {"path": path, "name": name, "split": split, "text_column": col},
            "seq_len": cfg.seq_len,
        }
        if ci is None:
            report["ci_note"] = (
                f"no interval reported: needs >= {_MIN_WINDOWS_FOR_CI} windows and "
                "bootstrap_resamples > 0"
            )
        report["interpretation_note"] = (
            "Perplexity measures fluency, not capability. Compression can hold "
            "perplexity steady while degrading reasoning accuracy — use the "
            "lm_eval_harness evaluator for capability claims."
        )
        (out_dir / "perplexity.json").write_text(json.dumps(report, indent=2) + "\n")

        ctx.artifacts["perplexity"] = float(ppl)
        if ci is not None:
            ctx.artifacts["perplexity_ci"] = (float(ci[0]), float(ci[1]))
            ctx.artifacts["perplexity_ci_level"] = cfg.ci_level
        ctx.artifacts["perplexity_report_path"] = str(out_dir / "perplexity.json")
        return ctx


def _resolve_model_path(ctx: StageContext, *, merge_adapter: bool) -> str:
    """Pick a loadable checkpoint path.

    If any compressor stage ran, ``ctx.model_path`` already points at a full
    quantized checkpoint — use it. Otherwise, if qlora left a bare adapter,
    merge it into the base (once, cached) so ``AutoModelForCausalLM`` can
    load it directly.
    """
    # Every compressor sets `quantized_model_path`; the gptq-specific key is
    # kept for recipes and artifacts written before that key existed.
    if ctx.artifacts.get("quantized_model_path") or ctx.artifacts.get("gptq_model_path"):
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


def _window_nlls(model_path: str, cfg: PerplexityConfig) -> tuple[list[float], list[int]]:
    """Score the corpus and return per-window (nll, scored_token_count).

    Returning the per-window series rather than a single aggregate is what makes
    the bootstrap possible — the windows are the resampling units.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path, name, split, text_column = cfg.resolved_dataset()
    ds_kwargs: dict[str, Any] = {"split": split}
    ds = (
        load_dataset(path, name, **ds_kwargs)
        if name is not None
        else load_dataset(path, **ds_kwargs)
    )
    if text_column not in ds.column_names:
        raise ValueError(
            f"perplexity: text_column {text_column!r} not in dataset columns {ds.column_names}"
        )

    rows = ds[text_column]
    if cfg.max_samples is not None:
        rows = rows[: cfg.max_samples]
    joined = "\n\n".join(r for r in rows if isinstance(r, str) and r)
    if not joined:
        raise RuntimeError("perplexity: dataset produced no non-empty text rows.")

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto").to(device)
    model.eval()

    encodings = tokenizer(joined, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    n_tokens = int(input_ids.size(1))
    if n_tokens < 2:
        raise RuntimeError("perplexity: dataset tokenizes to fewer than 2 tokens.")

    window_len = min(cfg.seq_len, n_tokens)
    stride = cfg.stride or max(window_len // 2, 1)

    nlls: list[float] = []
    weights: list[int] = []
    prev_end = 0
    for begin in range(0, n_tokens, stride):
        end = min(begin + window_len, n_tokens)
        trg_len = end - prev_end
        window = input_ids[:, begin:end]
        target = window.clone()
        target[:, :-trg_len] = -100

        with torch.no_grad():
            loss = model(window, labels=target).loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"perplexity: non-finite loss at window [{begin}:{end}] — "
                "check model dtype / device."
            )
        nlls.append(float(loss.item()))
        weights.append(int(trg_len))
        prev_end = end
        if end == n_tokens:
            break

    if not nlls:
        raise RuntimeError("perplexity: no evaluation windows produced.")
    return nlls, weights
