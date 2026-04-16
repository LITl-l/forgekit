"""Perplexity evaluator — CPU-safe unit tests.

These tests never import ``torch``, ``transformers``, or ``datasets``. They
exercise only registry resolution, config validation, dataset-alias resolution,
and the backend-availability guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import unknown_profile
from forgekit.plugins.evaluators.perplexity import (
    PerplexityConfig,
    PerplexityEvaluator,
    _require_backend,
)
from forgekit.stages import StageContext


def test_registry_resolves_perplexity() -> None:
    cls = registry.get("evaluator", "perplexity")
    assert cls is PerplexityEvaluator
    assert cls.name == "perplexity"


def test_config_defaults() -> None:
    cfg = PerplexityConfig.model_validate({})
    assert cfg.dataset == "wikitext2"
    assert cfg.seq_len == 2048
    assert cfg.stride is None
    assert cfg.merge_adapter is True
    assert cfg.output_subdir == "perplexity"


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PerplexityConfig.model_validate({"bogus": 1})


def test_config_rejects_unknown_dataset_alias() -> None:
    with pytest.raises(ValidationError, match=r"unknown dataset alias"):
        PerplexityConfig.model_validate({"dataset": "not-a-real-dataset"})


def test_resolved_dataset_uses_alias_defaults() -> None:
    cfg = PerplexityConfig.model_validate({"dataset": "wikitext2"})
    assert cfg.resolved_dataset() == ("wikitext", "wikitext-2-raw-v1", "test", "text")


def test_resolved_dataset_custom_path_overrides_alias() -> None:
    cfg = PerplexityConfig.model_validate(
        {
            "dataset": "wikitext2",  # ignored because dataset_path is set
            "dataset_path": "my/dataset",
            "dataset_split": "validation",
            "text_column": "content",
        }
    )
    assert cfg.resolved_dataset() == ("my/dataset", None, "validation", "content")


def test_resolved_dataset_respects_explicit_overrides_on_alias() -> None:
    cfg = PerplexityConfig.model_validate(
        {"dataset": "wikitext2", "dataset_split": "validation"}
    )
    path, name, split, col = cfg.resolved_dataset()
    assert (path, name, split, col) == ("wikitext", "wikitext-2-raw-v1", "validation", "text")


def test_require_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.evaluators.perplexity.importlib.util.find_spec",
        lambda _name: None,
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        _require_backend()


def test_require_backend_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.evaluators.perplexity.importlib.util.find_spec",
        lambda _name: object(),
    )
    _require_backend()


def test_evaluate_missing_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.evaluators.perplexity.importlib.util.find_spec",
        lambda _name: None,
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={},
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        PerplexityEvaluator().evaluate(ctx)


def test_evaluate_invalid_config_raises_before_backend_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, Any] = {}

    def fake_find_spec(_name: str) -> Any:
        called["did"] = True
        return object()

    monkeypatch.setattr(
        "forgekit.plugins.evaluators.perplexity.importlib.util.find_spec", fake_find_spec
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"bogus": True},
    )
    with pytest.raises(ValidationError):
        PerplexityEvaluator().evaluate(ctx)
    assert "did" not in called, "backend check must not run when config is invalid"
