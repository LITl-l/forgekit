"""EleutherAI lm-evaluation-harness.

Upstream: https://github.com/EleutherAI/lm-evaluation-harness (MIT).
Install via ``forgekit[lm-eval]``.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class LMEvalHarnessEvaluator:
    name: ClassVar[str] = "lm_eval_harness"

    def evaluate(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("lm_eval_harness evaluator is a scaffold stub.")
