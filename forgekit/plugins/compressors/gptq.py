"""GPTQ — Frantar et al. 2022, https://arxiv.org/abs/2210.17323.

Backend: ``auto-gptq``. Install via ``forgekit[gptq]``.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class GPTQCompressor:
    name: ClassVar[str] = "gptq"

    def compress(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("gptq compressor is a scaffold stub.")
