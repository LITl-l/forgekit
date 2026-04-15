"""GGUF exporter — llama.cpp's portable weights format.

Upstream: https://github.com/ggerganov/llama.cpp + ``gguf`` python package.
Install via ``forgekit[gguf]``.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class GGUFExporter:
    name: ClassVar[str] = "gguf"

    def export(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("gguf exporter is a scaffold stub.")
