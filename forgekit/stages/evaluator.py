"""Evaluator stage Protocol — run benchmarks / perplexity against the model."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from forgekit.stages import StageContext


@runtime_checkable
class EvaluatorPlugin(Protocol):
    name: ClassVar[str]

    def evaluate(self, ctx: StageContext) -> StageContext: ...
