"""Perplexity evaluator — sliding-window NLL over a held-out corpus.

Default corpus is wikitext-2 (``test`` split). The scoring loop mirrors the
HuggingFace reference (`transformers` docs: "Perplexity of fixed-length models")
— concatenate the text, slide windows of ``seq_len`` with step ``stride``,
mask out overlap tokens from the loss, and report ``exp(mean window NLL)``.

Backends: ``transformers`` + ``datasets`` + ``torch``. All of these are pulled
in by the ``[trl]`` and ``[gptq]`` extras, so most users already have them.
"""

from __future__ import annotations

import importlib.util
import json
import math
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, model_validator

from forgekit.stages import StageContext

# Alias → (path, name, split, text_column).
_DATASET_ALIASES: dict[str, tuple[str, str | None, str, str]] = {
    "wikitext2": ("wikitext", "wikitext-2-raw-v1", "test", "text"),
    "wikitext103": ("wikitext", "wikitext-103-raw-v1", "test", "text"),
    "ptb": ("ptb_text_only", None, "test", "sentence"),
    "c4": ("allenai/c4", "en", "validation", "text"),
}


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
    output_subdir: str = "perplexity"

    @model_validator(mode="after")
    def _validate_dataset(self) -> PerplexityConfig:
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


class PerplexityEvaluator:
    name: ClassVar[str] = "perplexity"

    def evaluate(self, ctx: StageContext) -> StageContext:
        cfg = PerplexityConfig.model_validate(ctx.stage_config)
        _require_backend()

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = _resolve_model_path(ctx, merge_adapter=cfg.merge_adapter)
        ppl = _compute_perplexity(model_path, cfg)

        path, name, split, col = cfg.resolved_dataset()
        report = {
            "perplexity": ppl,
            "model_path": model_path,
            "dataset": {"path": path, "name": name, "split": split, "text_column": col},
            "seq_len": cfg.seq_len,
        }
        (out_dir / "perplexity.json").write_text(json.dumps(report, indent=2) + "\n")

        ctx.artifacts["perplexity"] = float(ppl)
        ctx.artifacts["perplexity_report_path"] = str(out_dir / "perplexity.json")
        return ctx


def _resolve_model_path(ctx: StageContext, *, merge_adapter: bool) -> str:
    """Pick a loadable checkpoint path.

    If a prior ``gptq`` stage ran, ``ctx.model_path`` already points at a full
    quantized checkpoint — use it. Otherwise, if qlora left a bare adapter,
    merge it into the base (once, cached) so ``AutoModelForCausalLM`` can
    load it directly.
    """
    if ctx.artifacts.get("gptq_model_path"):
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


def _compute_perplexity(model_path: str, cfg: PerplexityConfig) -> float:
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

    nlls: list[torch.Tensor] = []
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
        nlls.append(loss)
        prev_end = end
        if end == n_tokens:
            break

    if not nlls:
        raise RuntimeError("perplexity: no evaluation windows produced.")

    mean_nll = torch.stack(nlls).mean().item()
    return math.exp(mean_nll)
