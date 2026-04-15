"""Perplexity evaluator — default sanity check for any model.

Uses ``transformers`` + a held-out dataset (wikitext2 by default).
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class PerplexityEvaluator:
    name: ClassVar[str] = "perplexity"

    def evaluate(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("perplexity evaluator is a scaffold stub.")
