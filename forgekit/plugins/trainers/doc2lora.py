"""Doc-to-LoRA — Sakana AI hypernetwork that emits LoRA adapters from documents.

Upstream: https://github.com/SakanaAI/doc-to-lora (CC BY-SA 4.0).
Install via ``forgekit[doc2lora]`` (currently a git+ install — pin the commit
in your project-level ``pyproject.toml``).

The Sakana upstream is pre-release; its public API may still move. If the
expected ``Doc2LoRAPipeline`` import path changes, adjust the lazy import in
``train`` — the rest of the plugin (config validation, document loading,
artifact handoff) is independent of upstream shape.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forgekit.stages import StageContext


class Doc2LoRAConfig(BaseModel):
    """Validated configuration for the doc2lora trainer."""

    model_config = ConfigDict(extra="forbid")

    documents: list[str] = Field(
        description="Source material — file paths or raw text. At least one required."
    )
    hypernetwork_checkpoint: str | None = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] | None = None
    device: str | None = None
    max_doc_tokens: int = 4096
    output_subdir: str = "doc2lora"

    @model_validator(mode="after")
    def _validate_nonempty(self) -> Doc2LoRAConfig:
        if not self.documents:
            raise ValueError("doc2lora: `documents` must contain at least one entry.")
        return self


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_backend() -> None:
    missing = [
        m for m in ("doc2lora", "transformers", "torch") if not _module_available(m)
    ]
    if missing:
        raise RuntimeError(
            f"doc2lora: required modules missing: {', '.join(missing)}. "
            "Install via `forgekit[doc2lora]` "
            "(https://github.com/SakanaAI/doc-to-lora)."
        )


def _load_documents(entries: list[str]) -> list[str]:
    """Each entry: if it's an existing file, read as text; otherwise keep as literal."""
    out: list[str] = []
    for entry in entries:
        try:
            path = Path(entry)
            if path.is_file():
                out.append(path.read_text())
                continue
        except OSError:
            pass
        out.append(entry)
    return out


class Doc2LoRATrainer:
    name: ClassVar[str] = "doc2lora"

    def train(self, ctx: StageContext) -> StageContext:
        cfg = Doc2LoRAConfig.model_validate(ctx.stage_config)
        _require_backend()

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        docs = _load_documents(cfg.documents)

        _doc2lora = importlib.import_module("doc2lora")
        Doc2LoRAPipeline: Any = _doc2lora.Doc2LoRAPipeline

        pipeline: Any = Doc2LoRAPipeline.from_pretrained(
            cfg.hypernetwork_checkpoint or "SakanaAI/doc-to-lora-default",
            base_model=ctx.model_path,
            device=cfg.device,
        )
        adapter_path = pipeline.generate_adapter(
            documents=docs,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            save_dir=str(out_dir),
            max_tokens=cfg.max_doc_tokens,
        )

        base_model = ctx.model_path
        ctx.artifacts["doc2lora_adapter_path"] = str(adapter_path)
        ctx.artifacts["doc2lora_base_model"] = base_model
        # Piggyback on qlora's artifact keys so compressors merge adapters uniformly.
        ctx.artifacts["qlora_adapter_path"] = str(adapter_path)
        ctx.artifacts["qlora_base_model"] = base_model
        ctx.model_path = str(adapter_path)
        return ctx
