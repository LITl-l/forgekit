"""DPO trainer — Direct Preference Optimization.

Backend: ``trl`` (``DPOTrainer``). Reference: Rafailov et al. 2023,
https://arxiv.org/abs/2305.18290.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class DPOTrainer:
    name: ClassVar[str] = "dpo"

    def train(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("dpo trainer is a scaffold stub.")
