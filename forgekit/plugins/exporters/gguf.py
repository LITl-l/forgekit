"""GGUF exporter — llama.cpp's portable weights format.

Backend: llama.cpp's ``convert_hf_to_gguf.py`` script plus the ``llama-quantize``
binary. Either ship via the ``llama-cpp-python`` package (which vendors
``convert_hf_to_gguf.py`` alongside the module) or the standalone ``llama.cpp``
repo on ``$PATH``. The pure-python ``gguf`` wheel also satisfies the backend
check — it's the writer library the conversion script itself depends on.

Install via ``forgekit[gguf]``.

The exporter runs two subprocess steps:

1. ``python convert_hf_to_gguf.py <source_model> --outfile <intermediate.gguf>``
   produces an f16 GGUF dump of the Hugging Face checkpoint.
2. ``llama-quantize <intermediate.gguf> <output.gguf> <QUANT>`` re-encodes that
   intermediate into the requested llama.cpp quant type. When the user asks for
   f16 (no quantization) we skip step 2 and rename the intermediate instead.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from forgekit.stages import StageContext

QuantType = Literal[
    "f32",
    "f16",
    "bf16",
    "q8_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q4_k_m",
    "q4_k_s",
    "q5_k_m",
    "q5_k_s",
    "q6_k",
    "q3_k_m",
    "q2_k",
]

_FLOAT_QUANTS = frozenset({"f32", "f16", "bf16"})


class GGUFConfig(BaseModel):
    """Validated configuration for the GGUF exporter."""

    model_config = ConfigDict(extra="forbid")

    quant: QuantType = "q4_k_m"
    convert_script: str | None = None
    quantize_binary: str | None = None
    output_subdir: str = "gguf"
    filename: str | None = None
    merge_adapter: bool = True


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_backend() -> None:
    """Backend check: either the ``gguf`` writer module or the conversion script."""
    if _module_available("gguf"):
        return
    try:
        _find_convert_script(None)
    except RuntimeError:
        pass
    else:
        return
    raise RuntimeError(
        "gguf: install `forgekit[gguf]` or ensure llama.cpp's "
        "`convert_hf_to_gguf.py` is on PATH via `convert_script`."
    )


def _find_convert_script(explicit: str | None) -> Path:
    """Locate ``convert_hf_to_gguf.py``. Precedence: explicit > PATH > llama_cpp pkg."""
    considered: list[str] = []

    if explicit is not None:
        candidate = Path(explicit).expanduser()
        considered.append(str(candidate))
        if candidate.is_file():
            return candidate

    on_path = shutil.which("convert_hf_to_gguf.py")
    if on_path is not None:
        return Path(on_path)
    considered.append("PATH:convert_hf_to_gguf.py")

    if _module_available("llama_cpp"):
        import llama_cpp

        module_file = getattr(llama_cpp, "__file__", None)
        if module_file is not None:
            candidate = Path(module_file).parent / "convert_hf_to_gguf.py"
            considered.append(str(candidate))
            if candidate.is_file():
                return candidate
    else:
        considered.append("llama_cpp:(not installed)")

    raise RuntimeError(
        "gguf: could not locate `convert_hf_to_gguf.py`. "
        "Set `convert_script` in the stage config or install llama.cpp. "
        f"Considered: {considered}"
    )


def _find_quantize_binary(explicit: str | None) -> Path:
    """Locate ``llama-quantize``. Precedence: explicit > PATH(llama-quantize) > PATH(quantize)."""
    considered: list[str] = []

    if explicit is not None:
        candidate = Path(explicit).expanduser()
        considered.append(str(candidate))
        if candidate.is_file():
            return candidate

    for binary_name in ("llama-quantize", "quantize"):
        on_path = shutil.which(binary_name)
        if on_path is not None:
            return Path(on_path)
        considered.append(f"PATH:{binary_name}")

    raise RuntimeError(
        "gguf: could not locate `llama-quantize` binary. "
        "Set `quantize_binary` in the stage config or install llama.cpp. "
        f"Considered: {considered}"
    )


class GGUFExporter:
    name: ClassVar[str] = "gguf"

    def export(self, ctx: StageContext) -> StageContext:
        cfg = GGUFConfig.model_validate(ctx.stage_config)
        _require_backend()

        out_dir = ctx.work_dir / cfg.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        source_model = _resolve_source_model(ctx, merge_adapter=cfg.merge_adapter)

        convert_script = _find_convert_script(cfg.convert_script)

        filename = cfg.filename or f"{ctx.recipe_name}.gguf"
        output_path = out_dir / filename

        needs_quantize = cfg.quant not in _FLOAT_QUANTS
        intermediate_path = (
            out_dir / f"{ctx.recipe_name}.f16.gguf" if needs_quantize else output_path
        )

        subprocess.run(
            [
                sys.executable,
                str(convert_script),
                str(source_model),
                "--outfile",
                str(intermediate_path),
            ],
            check=True,
        )

        if needs_quantize:
            quantize_binary = _find_quantize_binary(cfg.quantize_binary)
            subprocess.run(
                [
                    str(quantize_binary),
                    str(intermediate_path),
                    str(output_path),
                    cfg.quant.upper(),
                ],
                check=True,
            )
            intermediate_path.unlink(missing_ok=True)

        ctx.artifacts["gguf_path"] = str(output_path)
        ctx.artifacts["gguf_quant"] = cfg.quant
        ctx.model_path = str(output_path)
        return ctx


def _resolve_source_model(ctx: StageContext, *, merge_adapter: bool) -> str:
    """Merge a bare qlora adapter into its base so the converter sees dense weights."""
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
