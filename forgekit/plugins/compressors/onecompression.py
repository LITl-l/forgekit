"""OneCompression — Fujitsu's mixed-precision PTQ + AutoBit.

Upstream: https://github.com/FujitsuResearch/OneCompression (MIT).
Install via ``forgekit[onecompression]``. P0 differentiator for forgekit.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class OneCompressionCompressor:
    name: ClassVar[str] = "onecompression"

    def compress(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("onecompression compressor is a scaffold stub.")
