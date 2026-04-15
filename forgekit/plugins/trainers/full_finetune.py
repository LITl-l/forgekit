"""Full-weight fine-tuning (no adapter).

Backend: ``torchtune`` (default on GB10/128 GB unified memory) or
``transformers``+``accelerate``.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class FullFinetuneTrainer:
    name: ClassVar[str] = "full_finetune"

    def train(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("full_finetune trainer is a scaffold stub.")
