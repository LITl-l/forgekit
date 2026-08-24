"""AutoRound — Cheng et al., "Optimize Weight Rounding via Signed Gradient Descent
for the Quantization of LLMs" (https://arxiv.org/abs/2309.05516).

Backend: ``auto-round`` (Intel, Apache-2.0). Install via ``forgekit[autoround]``.

Why this plugin exists
----------------------
GPTQ and AWQ both round weights with a fixed rule and correct for the error
afterwards. AutoRound instead *learns* the rounding decision and the clipping
range together with signed gradient descent over a few hundred steps. The
practical consequence, per Intel's published comparisons and the low-bit
leaderboard:

* **W4**: roughly on par with GPTQ / AWQ.
* **W3 and W2**: consistently ahead — this is where the fixed-rule methods fall
  apart and AutoRound's search still finds usable roundings.
* **No inference overhead.** The output is an ordinary quantized checkpoint;
  the extra work is all at quantization time.

So: reach for ``gptq``/``awq`` at 4-bit if you already have a working recipe,
and reach for ``autoround`` when you want to go below 4-bit, or when you want
one plugin that can emit every downstream format.

Formats and schemes
-------------------
AutoRound decouples *what* it quantizes to (``scheme``) from *how the
checkpoint is written* (``format``). That is unusually useful here: a single
stage can produce a compressed-tensors checkpoint for vLLM, a GGUF for
llama.cpp, or an auto-gptq-compatible directory, without changing algorithms.

``scheme`` also covers ``NVFP4`` and ``MXFP4``. Those need Blackwell-class
hardware to run *fast*, and forgekit checks that for you — see
``_check_scheme_supported``. Note that "the hardware has FP4 tensor cores" and
"your serving stack has FP4 kernels for this chip" are different questions, and
the second one has been changing month to month on GB10 specifically. Measure
before you commit to a format; do not assume fewer bits means faster.
"""

from __future__ import annotations

import importlib.util
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forgekit.stages import StageContext

# Weight-only integer schemes, plus the two 4-bit float formats.
_INT_SCHEMES = frozenset({"W2A16", "W3A16", "W4A16", "W8A16"})
_FP4_SCHEMES = frozenset({"NVFP4", "MXFP4"})
_KNOWN_SCHEMES = _INT_SCHEMES | _FP4_SCHEMES

# Checkpoint writers AutoRound can target.
_KNOWN_FORMATS = frozenset(
    {"auto_round", "auto_gptq", "auto_awq", "llm_compressor"}
)


class AutoRoundConfig(BaseModel):
    """Validated configuration for the AutoRound compressor."""

    model_config = ConfigDict(extra="forbid")

    scheme: str = Field(
        default="W4A16",
        description=(
            "Quantization scheme: W2A16 / W3A16 / W4A16 / W8A16, NVFP4, MXFP4, "
            "or a GGUF target like 'GGUF:Q4_K_M'."
        ),
    )
    format: str = Field(
        default="auto_round",
        description=(
            "Checkpoint format: auto_round (default), llm_compressor (vLLM-native), "
            "auto_gptq, auto_awq, or a GGUF target like 'gguf:q4_k_m'."
        ),
    )
    nsamples: int = Field(default=128, description="Calibration samples.")
    iters: int = Field(default=200, description="Signed-gradient tuning steps.")
    lr: float | None = Field(
        default=None,
        description="Rounding learning rate. None lets AutoRound derive it from iters.",
    )
    seq_len: int = 2048
    low_gpu_mem_usage: bool = Field(
        default=False,
        description=(
            "Offload intermediate features to CPU. Saves substantial VRAM at "
            "roughly 30% slower quantization — worth it on 8-12 GB cards."
        ),
    )
    allow_unsupported_scheme: bool = Field(
        default=False,
        description=(
            "Skip the hardware capability check. Quantizing to FP4 on non-FP4 "
            "hardware works but the result cannot be served efficiently there."
        ),
    )
    output_subdir: str = "autoround"
    merge_adapter: bool = True

    @model_validator(mode="after")
    def _validate_scheme_and_format(self) -> Self:
        scheme_ok = self.scheme in _KNOWN_SCHEMES or self.scheme.upper().startswith(
            "GGUF:"
        )
        if not scheme_ok:
            raise ValueError(
                f"autoround: unknown scheme {self.scheme!r}. Expected one of "
                f"{sorted(_KNOWN_SCHEMES)} or a 'GGUF:<quant>' target."
            )
        format_ok = self.format in _KNOWN_FORMATS or self.format.lower().startswith(
            "gguf:"
        )
        if not format_ok:
            raise ValueError(
                f"autoround: unknown format {self.format!r}. Expected one of "
                f"{sorted(_KNOWN_FORMATS)} or a 'gguf:<quant>' target."
            )
        if self.iters < 0:
            raise ValueError("autoround: iters must be >= 0.")
        if self.nsamples < 1:
            raise ValueError("autoround: nsamples must be >= 1.")
        return self

    @property
    def is_fp4(self) -> bool:
        return self.scheme in _FP4_SCHEMES


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_backend() -> None:
    if not _module_available("auto_round"):
        raise RuntimeError(
            "autoround: `auto-round` is not installed. Install via "
            "`forgekit[autoround]`."
        )


def _check_scheme_supported(cfg: AutoRoundConfig, ctx: StageContext) -> None:
    """Fail early when an FP4 scheme is requested on hardware that lacks FP4.

    This is the guard that keeps forgekit honest away from its primary target.
    Quantizing to NVFP4 on an RTX 3090 produces a checkpoint that will be
    dequantized to bf16 on every forward pass — smaller on disk, slower to run,
    and less accurate than the W4A16 the user probably wanted.
    """
    if not cfg.is_fp4 or cfg.allow_unsupported_scheme:
        return
    if ctx.hw.supports_fp4:
        return
    raise RuntimeError(
        f"autoround: scheme {cfg.scheme!r} needs FP4 tensor cores, but the "
        f"detected device is {ctx.hw.arch!r} (backend={ctx.hw.backend}, "
        f"fp4={ctx.hw.supports_fp4}). Use scheme='W4A16' for this hardware, or "
        "set allow_unsupported_scheme=true if you are quantizing here to deploy "
        "elsewhere."
    )


class AutoRoundCompressor:
    name: ClassVar[str] = "autoround"

    def compress(self, ctx: StageContext) -> StageContext:
        cfg = AutoRoundConfig.model_validate(ctx.stage_config)
        _require_backend()
        _check_scheme_supported(cfg, ctx)

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        source_model = _resolve_source_model(ctx, merge_adapter=cfg.merge_adapter)

        from auto_round import AutoRound

        kwargs: dict[str, Any] = {
            "scheme": cfg.scheme,
            "nsamples": cfg.nsamples,
            "iters": cfg.iters,
            "seqlen": cfg.seq_len,
            "low_gpu_mem_usage": cfg.low_gpu_mem_usage,
        }
        if cfg.lr is not None:
            kwargs["lr"] = cfg.lr

        ar = AutoRound(source_model, **kwargs)
        ar.quantize_and_save(output_dir=str(out_dir), format=cfg.format)

        ctx.artifacts["autoround_model_path"] = str(out_dir)
        ctx.artifacts["autoround_scheme"] = cfg.scheme
        ctx.artifacts["autoround_format"] = cfg.format
        ctx.artifacts["quantized_model_path"] = str(out_dir)
        ctx.artifacts["quantized_format"] = cfg.format
        ctx.model_path = str(out_dir)
        return ctx


def _resolve_source_model(ctx: StageContext, *, merge_adapter: bool) -> str:
    """Merge a preceding LoRA adapter into its base, once, if one was produced."""
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
