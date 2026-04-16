"""GRPO trainer — CPU-safe unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import unknown_profile
from forgekit.plugins.trainers.grpo import (
    GRPOConfigModel,
    GRPOTrainer,
    _load_reward_function,
    _require_backend,
)
from forgekit.stages import StageContext


def test_registry_resolves_grpo() -> None:
    cls = registry.get("trainer", "grpo")
    assert cls is GRPOTrainer
    assert cls.name == "grpo"


def test_config_requires_dataset_and_reward() -> None:
    with pytest.raises(ValidationError):
        GRPOConfigModel.model_validate({})
    with pytest.raises(ValidationError):
        GRPOConfigModel.model_validate({"dataset": {"path": "x/y"}})


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        GRPOConfigModel.model_validate(
            {"dataset": {"path": "x/y"}, "reward_function": "m:f", "bogus": 1}
        )


def test_config_defaults() -> None:
    cfg = GRPOConfigModel.model_validate(
        {"dataset": {"path": "x/y"}, "reward_function": "m:f"}
    )
    assert cfg.num_generations == 8
    assert cfg.beta == pytest.approx(0.04)
    assert cfg.temperature == pytest.approx(0.9)
    assert cfg.use_lora is True


def test_load_reward_function_invalid_format() -> None:
    with pytest.raises(ValueError, match=r"module:name"):
        _load_reward_function("not-a-dotted-path")
    with pytest.raises(ValueError, match=r"module:name"):
        _load_reward_function(":bare")
    with pytest.raises(ValueError, match=r"module:name"):
        _load_reward_function("module:")


def test_load_reward_function_resolves_real_callable() -> None:
    fn = _load_reward_function("math:sqrt")
    assert callable(fn)
    assert fn(4) == 2.0


def test_load_reward_function_missing_attribute() -> None:
    with pytest.raises(ValueError, match=r"not found in module"):
        _load_reward_function("math:does_not_exist_xyz")


def test_require_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.grpo.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        _require_backend()


def test_require_backend_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.grpo.importlib.util.find_spec", lambda _n: object()
    )
    _require_backend()


def test_train_missing_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.grpo.importlib.util.find_spec", lambda _n: None
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"dataset": {"path": "x/y"}, "reward_function": "math:sqrt"},
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        GRPOTrainer().train(ctx)


def test_train_invalid_config_raises_before_backend_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, Any] = {}

    def fake_find_spec(_name: str) -> Any:
        called["did"] = True
        return object()

    monkeypatch.setattr(
        "forgekit.plugins.trainers.grpo.importlib.util.find_spec", fake_find_spec
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"dataset": {"path": "x/y"}},  # missing reward_function
    )
    with pytest.raises(ValidationError):
        GRPOTrainer().train(ctx)
    assert "did" not in called
