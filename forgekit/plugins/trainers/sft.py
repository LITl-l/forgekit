"""SFT trainer — supervised fine-tuning (full-precision LoRA or full weights).

Backends: ``trl`` (``SFTTrainer``), ``unsloth``, ``torchtune``.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class SFTTrainer:
    name: ClassVar[str] = "sft"

    def train(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("sft trainer is a scaffold stub.")
