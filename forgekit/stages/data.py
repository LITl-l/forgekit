"""Data stage Protocol — load and preprocess a dataset for training."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from forgekit.stages import StageContext


@runtime_checkable
class DataPlugin(Protocol):
    name: ClassVar[str]

    def prepare(self, ctx: StageContext) -> StageContext: ...
