"""SFT trainer — CPU-safe unit tests."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.plugins.trainers.sft import SFTConfig, SFTTrainer, _resolve_backend


def test_registry_resolves_sft() -> None:
    cls = registry.get("trainer", "sft")
    assert cls is SFTTrainer
    assert cls.name == "sft"


def test_config_requires_dataset() -> None:
    with pytest.raises(ValidationError):
        SFTConfig.model_validate({})


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SFTConfig.model_validate(
            {"dataset": {"path": "tatsu-lab/alpaca"}, "bogus": 1}
        )


def test_config_defaults_populated() -> None:
    cfg = SFTConfig.model_validate({"dataset": {"path": "tatsu-lab/alpaca"}})
    assert cfg.backend == "auto"
    assert cfg.lr == pytest.approx(2e-5)
    assert cfg.steps == 500
    assert cfg.use_lora is False
    assert cfg.load_in_8bit is False


def test_resolve_backend_prefers_unsloth(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_find_spec(name: str) -> Any:
        return object() if name == "unsloth" else None

    monkeypatch.setattr(
        "forgekit.plugins.trainers.sft.importlib.util.find_spec", fake_find_spec
    )
    assert _resolve_backend("auto") == "unsloth"


def test_resolve_backend_falls_back_to_trl(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_find_spec(name: str) -> Any:
        return object() if name == "trl" else None

    monkeypatch.setattr(
        "forgekit.plugins.trainers.sft.importlib.util.find_spec", fake_find_spec
    )
    assert _resolve_backend("auto") == "trl"


def test_resolve_backend_raises_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.sft.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match="no backend available"):
        _resolve_backend("auto")


def test_resolve_backend_explicit_unsloth_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.sft.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"unsloth.*not installed"):
        _resolve_backend("unsloth")


def test_resolve_backend_explicit_trl_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.sft.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"trl.*not installed"):
        _resolve_backend("trl")
