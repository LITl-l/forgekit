"""Recipe schema — the single declarative entry point for a forgekit run.

A Recipe is a YAML file that names a base model, a data source, one trainer,
zero or more compressors, an optional evaluator, and an optional exporter.
Each stage references a plugin by `kind` and passes through a free-form
`config` dict that the plugin validates itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StageSpec(BaseModel):
    """A single pipeline stage: which plugin (`kind`) with what settings (`config`)."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    config: dict[str, Any] = Field(default_factory=dict)


class DataSpec(StageSpec):
    """Data stage — same shape as StageSpec, typed separately for clarity."""


class HardwareHint(BaseModel):
    """Optional hardware profile override (normally autodetected)."""

    model_config = ConfigDict(extra="forbid")

    profile: str | None = None  # e.g. "rtx4090", "gb10_128g"


class RecipeSpec(BaseModel):
    """Top-level recipe. Validated on load; stage configs are opaque to the core."""

    model_config = ConfigDict(extra="forbid")

    name: str
    model: str
    data: DataSpec
    trainer: StageSpec
    compressors: list[StageSpec] = Field(default_factory=list)
    evaluator: StageSpec | None = None
    exporter: StageSpec | None = None
    hardware: HardwareHint | None = None


def load_recipe(path: str | Path) -> RecipeSpec:
    """Parse and validate a recipe YAML file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Recipe {path} must be a YAML mapping at the top level.")
    return RecipeSpec.model_validate(raw)
