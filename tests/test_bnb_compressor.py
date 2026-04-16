"""bnb compressor — CPU-safe unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import unknown_profile
from forgekit.plugins.compressors.bnb import BnBCompressor, BnBConfig, _require_backend
from forgekit.stages import StageContext


def test_registry_resolves_bnb() -> None:
    cls = registry.get("compressor", "bnb")
    assert cls is BnBCompressor
    assert cls.name == "bnb"


def test_config_defaults() -> None:
    cfg = BnBConfig.model_validate({})
    assert cfg.bits == 4
    assert cfg.quant_type == "nf4"
    assert cfg.double_quant is True
    assert cfg.compute_dtype == "bfloat16"
    assert cfg.output_subdir == "bnb"


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        BnBConfig.model_validate({"bogus": 1})


def test_config_rejects_invalid_bits() -> None:
    with pytest.raises(ValidationError):
        BnBConfig.model_validate({"bits": 3})


def test_config_accepts_valid_bits() -> None:
    for bits in (4, 8):
        cfg = BnBConfig.model_validate({"bits": bits})
        assert cfg.bits == bits


def test_config_rejects_invalid_quant_type() -> None:
    with pytest.raises(ValidationError):
        BnBConfig.model_validate({"quant_type": "int4"})


def test_require_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.bnb.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        _require_backend()


def test_require_backend_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.bnb.importlib.util.find_spec", lambda _n: object()
    )
    _require_backend()


def test_compress_missing_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.bnb.importlib.util.find_spec", lambda _n: None
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={},
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        BnBCompressor().compress(ctx)


def test_compress_invalid_config_raises_before_backend_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, Any] = {}

    def fake_find_spec(_name: str) -> Any:
        called["did"] = True
        return object()

    monkeypatch.setattr(
        "forgekit.plugins.compressors.bnb.importlib.util.find_spec", fake_find_spec
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"bits": 3},
    )
    with pytest.raises(ValidationError):
        BnBCompressor().compress(ctx)
    assert "did" not in called, "backend check must not run when config is invalid"
