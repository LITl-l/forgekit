"""Doc-to-LoRA — Sakana AI hypernetwork that emits LoRA adapters from documents.

Upstream: https://github.com/SakanaAI/doc-to-lora (CC BY-SA 4.0).
Install via ``forgekit[doc2lora]``.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class Doc2LoRATrainer:
    name: ClassVar[str] = "doc2lora"

    def train(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError(
            "doc2lora trainer is a scaffold stub. Requires the SakanaAI/doc-to-lora "
            "upstream — enable via the [doc2lora] extra."
        )
