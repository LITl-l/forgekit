"""Trainer stage Protocol — fine-tune (LoRA/QLoRA/SFT/DPO/GRPO/QAT/full) a model."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from forgekit.stages import StageContext


@runtime_checkable
class TrainerPlugin(Protocol):
    name: ClassVar[str]

    def train(self, ctx: StageContext) -> StageContext: ...
