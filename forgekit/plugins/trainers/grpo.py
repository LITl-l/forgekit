"""GRPO trainer — Group Relative Policy Optimization (DeepSeek-R1 style).

Backend: ``trl`` (``GRPOTrainer``). Reference: Shao et al. 2024,
https://arxiv.org/abs/2402.03300.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class GRPOTrainer:
    name: ClassVar[str] = "grpo"

    def train(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("grpo trainer is a scaffold stub.")
