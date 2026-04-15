"""RecipeSpec schema round-trip and validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from forgekit.recipe import RecipeSpec, load_recipe

RECIPES_DIR = Path(__file__).parent.parent / "recipes"


@pytest.mark.parametrize(
    "recipe_file",
    sorted(RECIPES_DIR.glob("*.yaml")),
    ids=lambda p: p.stem,
)
def test_example_recipes_load(recipe_file: Path) -> None:
    spec = load_recipe(recipe_file)
    assert spec.name
    assert spec.model
    assert spec.trainer.kind


def test_round_trip() -> None:
    spec = RecipeSpec.model_validate(
        {
            "name": "toy",
            "model": "gpt2",
            "data": {"kind": "hf_dataset", "config": {"path": "wikitext"}},
            "trainer": {"kind": "qlora", "config": {"lr": 1e-4}},
        }
    )
    round_tripped = RecipeSpec.model_validate(spec.model_dump())
    assert round_tripped == spec


def test_missing_required_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        RecipeSpec.model_validate({"name": "x"})  # no model, data, trainer


def test_extra_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        RecipeSpec.model_validate(
            {
                "name": "toy",
                "model": "gpt2",
                "data": {"kind": "hf_dataset"},
                "trainer": {"kind": "qlora"},
                "unexpected": True,
            }
        )
