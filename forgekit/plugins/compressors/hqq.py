"""HQQ — Half-Quadratic Quantization (Badri & Shaji 2023),
https://mobiusml.github.io/hqq_blog/.

Backend: ``hqq``. Install via ``forgekit[hqq]``.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class HQQCompressor:
    name: ClassVar[str] = "hqq"

    def compress(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("hqq compressor is a scaffold stub.")
