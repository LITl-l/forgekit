"""Compressor stage Protocol — post-training quantization / compression."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from forgekit.stages import StageContext


@runtime_checkable
class CompressorPlugin(Protocol):
    name: ClassVar[str]

    def compress(self, ctx: StageContext) -> StageContext: ...
