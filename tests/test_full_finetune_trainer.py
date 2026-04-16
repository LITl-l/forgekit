"""Full-finetune trainer — CPU-safe unit tests."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.plugins.trainers.full_finetune import (
    FullFinetuneConfig,
    FullFinetuneTrainer,
    _resolve_backend,
)


def test_registry_resolves_full_finetune() -> None:
    cls = registry.get("trainer", "full_finetune")
    assert cls is FullFinetuneTrainer
    assert cls.name == "full_finetune"


def test_config_requires_dataset() -> None:
    with pytest.raises(ValidationError):
        FullFinetuneConfig.model_validate({})


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        FullFinetuneConfig.model_validate(
            {"dataset": {"path": "x/y"}, "bogus": 1}
        )


def test_config_defaults() -> None:
    cfg = FullFinetuneConfig.model_validate({"dataset": {"path": "x/y"}})
    assert cfg.backend == "auto"
    assert cfg.lr == pytest.approx(2e-5)
    assert cfg.steps == 1000
    assert cfg.bf16 is True
    assert cfg.gradient_checkpointing is True
    assert cfg.optim == "adamw_torch"


def test_config_rejects_invalid_optim() -> None:
    with pytest.raises(ValidationError):
        FullFinetuneConfig.model_validate(
            {"dataset": {"path": "x/y"}, "optim": "sgd"}
        )


def test_config_rejects_invalid_backend() -> None:
    with pytest.raises(ValidationError):
        FullFinetuneConfig.model_validate(
            {"dataset": {"path": "x/y"}, "backend": "bogus"}
        )


def test_resolve_backend_prefers_torchtune(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(name: str) -> Any:
        return object() if name == "torchtune" else None

    monkeypatch.setattr(
        "forgekit.plugins.trainers.full_finetune.importlib.util.find_spec", fake
    )
    assert _resolve_backend("auto") == "torchtune"


def test_resolve_backend_falls_back_to_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(name: str) -> Any:
        return object() if name == "transformers" else None

    monkeypatch.setattr(
        "forgekit.plugins.trainers.full_finetune.importlib.util.find_spec", fake
    )
    assert _resolve_backend("auto") == "transformers"


def test_resolve_backend_raises_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.full_finetune.importlib.util.find_spec",
        lambda _n: None,
    )
    with pytest.raises(RuntimeError, match="no backend available"):
        _resolve_backend("auto")


def test_resolve_backend_explicit_torchtune_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.full_finetune.importlib.util.find_spec",
        lambda _n: None,
    )
    with pytest.raises(RuntimeError, match=r"torchtune.*not installed"):
        _resolve_backend("torchtune")


def test_resolve_backend_explicit_transformers_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.full_finetune.importlib.util.find_spec",
        lambda _n: None,
    )
    with pytest.raises(RuntimeError, match=r"transformers.*not installed"):
        _resolve_backend("transformers")
