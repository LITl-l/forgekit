"""AWQ — Lin et al. 2023, https://arxiv.org/abs/2306.00978.

Backend: ``autoawq``. Install via ``forgekit[awq]``.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class AWQCompressor:
    name: ClassVar[str] = "awq"

    def compress(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("awq compressor is a scaffold stub.")
