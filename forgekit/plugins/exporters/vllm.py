"""vLLM exporter — write safetensors + config suitable for vLLM serving.

Intended to wire in TriAttention KV-cache compression (Apache-2.0) at serve
time where supported. Install via ``forgekit[vllm]``.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class VLLMExporter:
    name: ClassVar[str] = "vllm"

    def export(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("vllm exporter is a scaffold stub.")
