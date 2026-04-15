"""bitsandbytes — 8-bit / 4-bit NF4 runtime quantization.

Backend: ``bitsandbytes``. Install via ``forgekit[bnb]``.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class BnBCompressor:
    name: ClassVar[str] = "bnb"

    def compress(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("bnb compressor is a scaffold stub.")
