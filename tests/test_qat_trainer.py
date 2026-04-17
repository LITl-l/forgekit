"""QAT trainer — CPU-safe unit tests."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.plugins.trainers.qat import (
    QATConfig,
    QATTrainer,
    _resolve_backend,
)


def test_registry_resolves_qat() -> None:
    cls = registry.get("trainer", "qat")
    assert cls is QATTrainer
    assert cls.name == "qat"


def test_config_requires_dataset() -> None:
    with pytest.raises(ValidationError):
        QATConfig.model_validate({})


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        QATConfig.model_validate({"dataset": {"path": "x/y"}, "bogus": 1})


def test_config_defaults() -> None:
    cfg = QATConfig.model_validate({"dataset": {"path": "x/y"}})
    assert cfg.target_bits == 4
    assert cfg.group_size == 32
    assert cfg.backend == "auto"
    assert cfg.bf16 is True


def test_config_rejects_invalid_target_bits() -> None:
    with pytest.raises(ValidationError):
        QATConfig.model_validate({"dataset": {"path": "x/y"}, "target_bits": 3})


def test_config_accepts_valid_target_bits() -> None:
    for bits in (4, 8):
        cfg = QATConfig.model_validate({"dataset": {"path": "x/y"}, "target_bits": bits})
        assert cfg.target_bits == bits


def test_resolve_backend_prefers_torchao(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(name: str) -> Any:
        return object() if name == "torchao" else None

    monkeypatch.setattr(
        "forgekit.plugins.trainers.qat.importlib.util.find_spec", fake
    )
    assert _resolve_backend("auto") == "torchao"


def test_resolve_backend_falls_back_to_torchtune(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(name: str) -> Any:
        return object() if name == "torchtune" else None

    monkeypatch.setattr(
        "forgekit.plugins.trainers.qat.importlib.util.find_spec", fake
    )
    assert _resolve_backend("auto") == "torchtune"


def test_resolve_backend_raises_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.qat.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match="no backend available"):
        _resolve_backend("auto")


def test_resolve_backend_explicit_torchao_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.qat.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"torchao.*not installed"):
        _resolve_backend("torchao")


def test_resolve_backend_explicit_torchtune_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.qat.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"torchtune.*not installed"):
        _resolve_backend("torchtune")
