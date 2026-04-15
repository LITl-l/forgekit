"""MLX exporter — convert to Apple MLX format for on-device inference.

Upstream: ``mlx-lm``. Install via ``forgekit[mlx]``.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class MLXExporter:
    name: ClassVar[str] = "mlx"

    def export(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError("mlx exporter is a scaffold stub.")
