"""AQLM — Additive Quantization of Language Models (Egiazarian et al. 2024),
https://arxiv.org/abs/2401.06118.

Backend: ``aqlm``. Install via ``forgekit[aqlm]``.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class AQLMCompressor:
    name: ClassVar[str] = "aqlm"

    def compress(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("aqlm compressor is a scaffold stub.")
