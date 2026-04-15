"""Exporter stage Protocol — convert the final model to a serving format (GGUF, vLLM, MLX)."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from forgekit.stages import StageContext


@runtime_checkable
class ExporterPlugin(Protocol):
    name: ClassVar[str]

    def export(self, ctx: StageContext) -> StageContext: ...
