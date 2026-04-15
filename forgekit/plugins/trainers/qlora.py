"""QLoRA trainer — LoRA over 4-bit NF4 base weights.

Backends: ``unsloth`` (default ≤24 GB), ``trl`` (``SFTTrainer`` + PEFT),
``torchtune``. Reference: Dettmers et al. 2023, https://arxiv.org/abs/2305.14314.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class QLoRATrainer:
    name: ClassVar[str] = "qlora"

    def train(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError(
            "qlora trainer is a scaffold stub. Install `forgekit[unsloth]` or "
            "`forgekit[trl]` and wait for the follow-up implementation PR."
        )
