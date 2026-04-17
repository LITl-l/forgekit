"""Doc2LoRA trainer — CPU-safe unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import unknown_profile
from forgekit.plugins.trainers.doc2lora import (
    Doc2LoRAConfig,
    Doc2LoRATrainer,
    _load_documents,
    _require_backend,
)
from forgekit.stages import StageContext


def test_registry_resolves_doc2lora() -> None:
    cls = registry.get("trainer", "doc2lora")
    assert cls is Doc2LoRATrainer
    assert cls.name == "doc2lora"


def test_config_requires_documents() -> None:
    with pytest.raises(ValidationError):
        Doc2LoRAConfig.model_validate({})


def test_config_rejects_empty_documents() -> None:
    with pytest.raises(ValidationError):
        Doc2LoRAConfig.model_validate({"documents": []})


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Doc2LoRAConfig.model_validate({"documents": ["hi"], "bogus": 1})


def test_config_defaults() -> None:
    cfg = Doc2LoRAConfig.model_validate({"documents": ["hi"]})
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.max_doc_tokens == 4096
    assert cfg.hypernetwork_checkpoint is None


def test_load_documents_mixed_paths_and_literals(tmp_path: Path) -> None:
    file1 = tmp_path / "a.txt"
    file1.write_text("alpha content")
    file2 = tmp_path / "b.txt"
    file2.write_text("beta content")
    docs = _load_documents([str(file1), "raw literal text", str(file2)])
    assert docs == ["alpha content", "raw literal text", "beta content"]


def test_load_documents_nonexistent_path_treated_as_literal() -> None:
    docs = _load_documents(["/this/path/does/not/exist.txt"])
    assert docs == ["/this/path/does/not/exist.txt"]


def test_require_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.doc2lora.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        _require_backend()


def test_require_backend_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.doc2lora.importlib.util.find_spec",
        lambda _n: object(),
    )
    _require_backend()


def test_train_missing_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.doc2lora.importlib.util.find_spec", lambda _n: None
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"documents": ["hi"]},
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        Doc2LoRATrainer().train(ctx)


def test_train_invalid_config_raises_before_backend_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, Any] = {}

    def fake_find_spec(_name: str) -> Any:
        called["did"] = True
        return object()

    monkeypatch.setattr(
        "forgekit.plugins.trainers.doc2lora.importlib.util.find_spec", fake_find_spec
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"documents": []},
    )
    with pytest.raises(ValidationError):
        Doc2LoRATrainer().train(ctx)
    assert "did" not in called
