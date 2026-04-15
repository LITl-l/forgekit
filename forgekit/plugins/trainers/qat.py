"""QAT trainer — Quantization-Aware Training.

Backend: ``torchtune`` (``QATRecipe``) or ``transformers`` with custom hooks.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class QATTrainer:
    name: ClassVar[str] = "qat"

    def train(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("qat trainer is a scaffold stub.")
