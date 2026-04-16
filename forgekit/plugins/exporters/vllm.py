"""vLLM exporter — materialize an HF checkpoint + serve config for vLLM.

Backend: ``transformers`` + ``torch`` (and ``peft`` when merging a qlora
adapter into its base). Install via ``forgekit[vllm]``.

vLLM serves Hugging Face-format checkpoints directly from a directory of
safetensors + ``config.json``. The exporter's job is therefore:

1. Materialize a canonical HF checkpoint on disk under ``work_dir/<subdir>``
   (merging a bare qlora adapter into its base weights when necessary, and
   pass-through for already-quantized GPTQ / AWQ / HQQ / BNB checkpoints
   produced by a prior stage).
2. Emit a small ``vllm_config.json`` capturing the recommended serve args so
   it can be consumed directly by ``vllm serve`` / ``vllm.LLM(**config)``.
3. Optionally (``smoke_test=True``) instantiate ``vllm.LLM`` against the
   exported directory and run a one-token completion to prove the checkpoint
   loads — expensive, so off by default.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from forgekit.stages import StageContext

Dtype = Literal["auto", "bfloat16", "float16", "float32"]
Quantization = Literal["auto", "awq", "gptq", "bitsandbytes", "fp8", "none"]


class VLLMConfig(BaseModel):
    """Validated configuration for the vLLM exporter."""

    model_config = ConfigDict(extra="forbid")

    output_subdir: str = "vllm"
    merge_adapter: bool = True
    dtype: Dtype = "auto"
    max_model_len: int | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = False
    smoke_test: bool = False
    quantization: Quantization = "auto"


_QUANT_ARTIFACT_KEYS: tuple[str, ...] = (
    "gptq_model_path",
    "awq_model_path",
    "hqq_model_path",
    "bnb_model_path",
)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_backend(*, smoke_test: bool) -> None:
    required = ["transformers", "torch"]
    if smoke_test:
        required.append("vllm")
    missing = [m for m in required if not _module_available(m)]
    if missing:
        raise RuntimeError(
            f"vllm: required modules missing: {', '.join(missing)}. "
            "Install via `forgekit[vllm]` (plus transformers / torch for the HF checkpoint)."
        )


class VLLMExporter:
    name: ClassVar[str] = "vllm"

    def export(self, ctx: StageContext) -> StageContext:
        cfg = VLLMConfig.model_validate(ctx.stage_config)
        _require_backend(smoke_test=cfg.smoke_test)

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        source = _resolve_source_model(ctx, merge_adapter=cfg.merge_adapter)
        source_path = Path(source)

        if source_path.resolve() != out_dir.resolve():
            if source_path.is_dir():
                shutil.copytree(source_path, out_dir, dirs_exist_ok=True)
            else:
                # Not a local directory (e.g. a bare HF Hub id); materialize
                # via transformers so vLLM can load from a local path.
                from transformers import AutoModelForCausalLM, AutoTokenizer

                model = AutoModelForCausalLM.from_pretrained(source)
                model.save_pretrained(str(out_dir))
                AutoTokenizer.from_pretrained(source).save_pretrained(str(out_dir))

        serve_config: dict[str, object] = {
            "model": str(out_dir),
            "dtype": cfg.dtype,
            "max_model_len": cfg.max_model_len,
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "trust_remote_code": cfg.trust_remote_code,
            "quantization": cfg.quantization if cfg.quantization != "none" else None,
        }
        config_path = out_dir / "vllm_config.json"
        config_path.write_text(json.dumps(serve_config, indent=2) + "\n")

        if cfg.smoke_test:
            from vllm import LLM, SamplingParams

            llm = LLM(
                model=str(out_dir),
                dtype=cfg.dtype,
                tensor_parallel_size=cfg.tensor_parallel_size,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
                trust_remote_code=cfg.trust_remote_code,
                max_model_len=cfg.max_model_len,
                quantization=serve_config["quantization"],
            )
            llm.generate(["hello"], SamplingParams(max_tokens=1))

        ctx.artifacts["vllm_export_path"] = str(out_dir)
        ctx.artifacts["vllm_config_path"] = str(config_path)
        ctx.model_path = str(out_dir)
        return ctx


def _resolve_source_model(ctx: StageContext, *, merge_adapter: bool) -> str:
    """Return the HF-compatible checkpoint to stage for vLLM.

    Priority:

    1. If an upstream compressor announced a quantized checkpoint via
       ``{gptq,awq,hqq,bnb}_model_path``, pass ``ctx.model_path`` through —
       those checkpoints are already self-contained and vLLM knows how to
       load them given the right ``quantization`` hint.
    2. If the model is a bare qlora adapter (``qlora_adapter_path`` +
       ``qlora_base_model``) and ``merge_adapter`` is True, merge it into
       its base so vLLM sees a dense HF checkpoint.
    3. Otherwise, use ``ctx.model_path`` as-is.
    """
    for key in _QUANT_ARTIFACT_KEYS:
        if ctx.artifacts.get(key):
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
