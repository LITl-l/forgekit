"""lm_eval_harness evaluator — CPU-safe unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import unknown_profile
from forgekit.plugins.evaluators.lm_eval_harness import (
    LMEvalConfig,
    LMEvalHarnessEvaluator,
    _build_model_args,
    _require_backend,
)
from forgekit.stages import StageContext


def test_registry_resolves_lm_eval() -> None:
    cls = registry.get("evaluator", "lm_eval_harness")
    assert cls is LMEvalHarnessEvaluator
    assert cls.name == "lm_eval_harness"


def test_config_requires_tasks() -> None:
    with pytest.raises(ValidationError):
        LMEvalConfig.model_validate({})
    with pytest.raises(ValidationError):
        LMEvalConfig.model_validate({"tasks": []})


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        LMEvalConfig.model_validate({"tasks": ["hellaswag"], "bogus": 1})


def test_config_defaults() -> None:
    cfg = LMEvalConfig.model_validate({"tasks": ["hellaswag"]})
    assert cfg.num_fewshot is None
    assert cfg.batch_size == "auto"
    assert cfg.model_type == "hf"
    assert cfg.trust_remote_code is False


def test_config_rejects_invalid_model_type() -> None:
    with pytest.raises(ValidationError):
        LMEvalConfig.model_validate(
            {"tasks": ["hellaswag"], "model_type": "bogus"}
        )


def test_build_model_args_basic() -> None:
    cfg = LMEvalConfig.model_validate({"tasks": ["hellaswag"]})
    out = _build_model_args("path/to/model", cfg)
    assert out == "pretrained=path/to/model,trust_remote_code=false"


def test_build_model_args_with_trust_and_extras() -> None:
    cfg = LMEvalConfig.model_validate(
        {
            "tasks": ["hellaswag"],
            "trust_remote_code": True,
            "extra_model_args": {"dtype": "bfloat16", "parallelize": "True"},
        }
    )
    out = _build_model_args("p", cfg)
    assert out.startswith("pretrained=p,trust_remote_code=true")
    assert "dtype=bfloat16" in out
    assert "parallelize=True" in out


def test_require_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.evaluators.lm_eval_harness.importlib.util.find_spec",
        lambda _n: None,
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        _require_backend()


def test_require_backend_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.evaluators.lm_eval_harness.importlib.util.find_spec",
        lambda _n: object(),
    )
    _require_backend()


def test_evaluate_missing_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.evaluators.lm_eval_harness.importlib.util.find_spec",
        lambda _n: None,
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"tasks": ["hellaswag"]},
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        LMEvalHarnessEvaluator().evaluate(ctx)


def test_evaluate_invalid_config_raises_before_backend_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, Any] = {}

    def fake_find_spec(_name: str) -> Any:
        called["did"] = True
        return object()

    monkeypatch.setattr(
        "forgekit.plugins.evaluators.lm_eval_harness.importlib.util.find_spec",
        fake_find_spec,
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"tasks": []},
    )
    with pytest.raises(ValidationError):
        LMEvalHarnessEvaluator().evaluate(ctx)
    assert "did" not in called
