"""Every shipped recipe must validate against the plugins it names.

`load_recipe` only checks the top-level `RecipeSpec`; stage `config` blocks are
opaque dicts that each plugin validates at *run* time. That is a good design,
but it meant a recipe could ship with a typo'd or removed field and nothing
would notice until someone ran it — which is how `micro_batch` (the field is
`micro_batch_size`), a missing required `dataset`, and an AutoAWQ-only
`version: gemm` all survived in `recipes/`.

This closes that gap by validating each stage config against the plugin's own
pydantic model. The mapping is explicit rather than discovered: plugins are not
required to expose their config class, and inventing that contract here would
be a bigger change than the problem warrants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from forgekit.plugins.compressors.aqlm import AQLMConfig
from forgekit.plugins.compressors.autoround import AutoRoundConfig
from forgekit.plugins.compressors.awq import AWQConfig
from forgekit.plugins.compressors.bnb import BnBConfig
from forgekit.plugins.compressors.gptq import GPTQConfig
from forgekit.plugins.compressors.hqq import HQQConfig
from forgekit.plugins.compressors.onecompression import OneCompressionConfig
from forgekit.plugins.evaluators.lm_eval_harness import LMEvalConfig
from forgekit.plugins.evaluators.perplexity import PerplexityConfig
from forgekit.plugins.exporters.gguf import GGUFConfig
from forgekit.plugins.exporters.mlx import MLXConfig
from forgekit.plugins.exporters.vllm import VLLMConfig
from forgekit.plugins.trainers.doc2lora import Doc2LoRAConfig
from forgekit.plugins.trainers.full_finetune import FullFinetuneConfig
from forgekit.plugins.trainers.qat import QATConfig
from forgekit.plugins.trainers.qlora import QLoRAConfig
from forgekit.plugins.trainers.sft import SFTConfig
from forgekit.recipe import load_recipe

RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes"

# (stage kind, plugin name) → config model.
CONFIG_MODELS: dict[tuple[str, str], type[BaseModel]] = {
    ("trainer", "qlora"): QLoRAConfig,
    ("trainer", "sft"): SFTConfig,
    ("trainer", "qat"): QATConfig,
    ("trainer", "full_finetune"): FullFinetuneConfig,
    ("trainer", "doc2lora"): Doc2LoRAConfig,
    ("compressor", "gptq"): GPTQConfig,
    ("compressor", "awq"): AWQConfig,
    ("compressor", "autoround"): AutoRoundConfig,
    ("compressor", "hqq"): HQQConfig,
    ("compressor", "bnb"): BnBConfig,
    ("compressor", "aqlm"): AQLMConfig,
    ("compressor", "onecompression"): OneCompressionConfig,
    ("evaluator", "perplexity"): PerplexityConfig,
    ("evaluator", "lm_eval_harness"): LMEvalConfig,
    ("exporter", "gguf"): GGUFConfig,
    ("exporter", "vllm"): VLLMConfig,
    ("exporter", "mlx"): MLXConfig,
}

RECIPE_PATHS = sorted(RECIPES_DIR.glob("*.y*ml"))


def _stages(recipe: Any) -> list[tuple[str, str, dict[str, Any]]]:
    stages: list[tuple[str, str, dict[str, Any]]] = [
        ("trainer", recipe.trainer.kind, recipe.trainer.config)
    ]
    stages += [("compressor", c.kind, c.config) for c in recipe.compressors]
    if recipe.evaluator is not None:
        stages.append(("evaluator", recipe.evaluator.kind, recipe.evaluator.config))
    if recipe.exporter is not None:
        stages.append(("exporter", recipe.exporter.kind, recipe.exporter.config))
    return stages


def test_recipes_directory_is_not_empty() -> None:
    assert RECIPE_PATHS, "no recipes found — check RECIPES_DIR"


@pytest.mark.parametrize("path", RECIPE_PATHS, ids=lambda p: p.name)
def test_recipe_parses(path: Path) -> None:
    spec = load_recipe(path)
    assert spec.name
    assert spec.model


@pytest.mark.parametrize("path", RECIPE_PATHS, ids=lambda p: p.name)
def test_recipe_stage_configs_validate(path: Path) -> None:
    """Each stage config must satisfy its plugin's model, with no unknown keys."""
    recipe = load_recipe(path)
    for stage_kind, plugin_name, config in _stages(recipe):
        model = CONFIG_MODELS.get((stage_kind, plugin_name))
        if model is None:
            pytest.skip(f"no config model mapped for {stage_kind}:{plugin_name}")
        # Raises ValidationError on unknown or missing fields — every plugin
        # config uses extra="forbid".
        model.model_validate(config)


@pytest.mark.parametrize("path", RECIPE_PATHS, ids=lambda p: p.name)
def test_recipe_names_only_registered_plugins(path: Path) -> None:
    from forgekit import registry

    recipe = load_recipe(path)
    for stage_kind, plugin_name, _ in _stages(recipe):
        registry.get(stage_kind, plugin_name)  # raises KeyError on a typo


def test_every_mapped_plugin_is_registered() -> None:
    """Guards the mapping above against drifting from the entry points."""
    from forgekit import registry

    for stage_kind, plugin_name in CONFIG_MODELS:
        registry.get(stage_kind, plugin_name)


def test_awq_recipe_rejects_removed_version_field() -> None:
    """Regression: `version: gemm` was valid under AutoAWQ and is now not."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AWQConfig.model_validate({"bits": 4, "group_size": 128, "version": "gemm"})


def test_qlora_recipe_rejects_micro_batch_typo() -> None:
    """Regression: the field is `micro_batch_size`; `micro_batch` shipped for months."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QLoRAConfig.model_validate(
            {"dataset": {"path": "x"}, "micro_batch": 2}
        )
