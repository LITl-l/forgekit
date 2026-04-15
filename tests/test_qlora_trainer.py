"""QLoRA trainer — CPU-safe unit tests.

These tests never import `unsloth`, `trl`, `transformers`, `peft`, or `torch`.
They only exercise the plugin's config validation and backend-resolution logic.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.plugins.trainers.qlora import (
    QLoRAConfig,
    QLoRATrainer,
    _resolve_backend,
)


def test_registry_resolves_qlora() -> None:
    cls = registry.get("trainer", "qlora")
    assert cls is QLoRATrainer
    assert cls.name == "qlora"


def test_config_requires_dataset() -> None:
    with pytest.raises(ValidationError):
        QLoRAConfig.model_validate({})


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        QLoRAConfig.model_validate(
            {"dataset": {"path": "tatsu-lab/alpaca"}, "bogus_field": 1}
        )


def test_config_defaults_populated() -> None:
    cfg = QLoRAConfig.model_validate({"dataset": {"path": "tatsu-lab/alpaca"}})
    assert cfg.backend == "auto"
    assert cfg.lr == pytest.approx(2e-4)
    assert cfg.steps == 200
    assert cfg.lora_r == 16
    assert cfg.dataset.split == "train"
    assert cfg.dataset.text_column == "text"


def test_resolve_backend_prefers_unsloth_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_find_spec(name: str) -> Any:
        return object() if name == "unsloth" else None

    monkeypatch.setattr(
        "forgekit.plugins.trainers.qlora.importlib.util.find_spec", fake_find_spec
    )
    assert _resolve_backend("auto") == "unsloth"


def test_resolve_backend_falls_back_to_trl(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_find_spec(name: str) -> Any:
        return object() if name == "trl" else None

    monkeypatch.setattr(
        "forgekit.plugins.trainers.qlora.importlib.util.find_spec", fake_find_spec
    )
    assert _resolve_backend("auto") == "trl"


def test_resolve_backend_raises_when_no_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.qlora.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match="no backend available"):
        _resolve_backend("auto")


def test_resolve_backend_explicit_unsloth_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.qlora.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"unsloth.*not installed"):
        _resolve_backend("unsloth")


def test_resolve_backend_explicit_trl_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.qlora.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"trl.*not installed"):
        _resolve_backend("trl")
