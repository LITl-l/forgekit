"""DPO trainer — CPU-safe unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import unknown_profile
from forgekit.plugins.trainers.dpo import DPOConfigModel, DPOTrainer, _require_backend
from forgekit.stages import StageContext


def test_registry_resolves_dpo() -> None:
    cls = registry.get("trainer", "dpo")
    assert cls is DPOTrainer
    assert cls.name == "dpo"


def test_config_requires_dataset() -> None:
    with pytest.raises(ValidationError):
        DPOConfigModel.model_validate({})


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DPOConfigModel.model_validate(
            {"dataset": {"path": "argilla/ultrafeedback-binarized"}, "bogus": 1}
        )


def test_config_defaults() -> None:
    cfg = DPOConfigModel.model_validate({"dataset": {"path": "argilla/ultrafeedback-binarized"}})
    assert cfg.lr == pytest.approx(5e-7)
    assert cfg.beta == pytest.approx(0.1)
    assert cfg.loss_type == "sigmoid"
    assert cfg.use_lora is True
    assert cfg.load_in_4bit is False


def test_config_rejects_invalid_loss_type() -> None:
    with pytest.raises(ValidationError):
        DPOConfigModel.model_validate(
            {"dataset": {"path": "x/y"}, "loss_type": "bogus"}
        )


def test_config_accepts_valid_loss_types() -> None:
    for lt in ("sigmoid", "hinge", "ipo", "kto_pair"):
        cfg = DPOConfigModel.model_validate(
            {"dataset": {"path": "x/y"}, "loss_type": lt}
        )
        assert cfg.loss_type == lt


def test_require_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.dpo.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        _require_backend()


def test_require_backend_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.dpo.importlib.util.find_spec", lambda _n: object()
    )
    _require_backend()


def test_train_missing_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.dpo.importlib.util.find_spec", lambda _n: None
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"dataset": {"path": "x/y"}},
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        DPOTrainer().train(ctx)


def test_train_invalid_config_raises_before_backend_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, Any] = {}

    def fake_find_spec(_name: str) -> Any:
        called["did"] = True
        return object()

    monkeypatch.setattr(
        "forgekit.plugins.trainers.dpo.importlib.util.find_spec", fake_find_spec
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"dataset": {"path": "x/y"}, "loss_type": "bogus"},
    )
    with pytest.raises(ValidationError):
        DPOTrainer().train(ctx)
    assert "did" not in called
