"""Pipeline stage Protocols and the shared StageContext."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forgekit.hw.profiles import HardwareProfile


@dataclass
class StageContext:
    """Runtime information handed to each stage plugin.

    Plugins receive one of these and return an updated one (typically with an
    updated `model_path` after training or compression).
    """

    recipe_name: str
    model_path: str
    work_dir: Path
    hw: HardwareProfile
    stage_config: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


__all__ = ["StageContext"]
