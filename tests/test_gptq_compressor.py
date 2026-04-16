"""GPTQ compressor — CPU-safe unit tests.

These tests never import `auto_gptq`, `transformers`, `peft`, `datasets`, or
`torch`. They exercise only config validation, registry resolution, and the
`auto-gptq` availability guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import unknown_profile
from forgekit.plugins.compressors.gptq import (
    GPTQCompressor,
    GPTQConfig,
    _require_auto_gptq,
)
from forgekit.stages import StageContext


def test_registry_resolves_gptq() -> None:
    cls = registry.get("compressor", "gptq")
    assert cls is GPTQCompressor
    assert cls.name == "gptq"


def test_config_defaults() -> None:
    cfg = GPTQConfig.model_validate({})
    assert cfg.bits == 4
    assert cfg.group_size == 128
    assert cfg.sym is True
    assert cfg.calibration.path == "wikitext"
    assert cfg.calibration.name == "wikitext-2-raw-v1"
    assert cfg.calibration.num_samples == 128


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        GPTQConfig.model_validate({"bogus": 1})


def test_config_rejects_invalid_bits() -> None:
    with pytest.raises(ValidationError):
        GPTQConfig.model_validate({"bits": 5})


def test_config_accepts_valid_bits() -> None:
    for bits in (2, 3, 4, 8):
        cfg = GPTQConfig.model_validate({"bits": bits})
        assert cfg.bits == bits


def test_require_auto_gptq_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.gptq.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"auto-gptq.*not installed"):
        _require_auto_gptq()


def test_require_auto_gptq_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.gptq.importlib.util.find_spec",
        lambda name: object() if name == "auto_gptq" else None,
    )
    _require_auto_gptq()


def test_compress_missing_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.gptq.importlib.util.find_spec", lambda _n: None
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={},
    )
    with pytest.raises(RuntimeError, match=r"auto-gptq.*not installed"):
        GPTQCompressor().compress(ctx)


def test_compress_invalid_config_raises_before_backend_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, Any] = {}

    def fake_find_spec(_n: str) -> Any:
        called["did"] = True
        return object()

    monkeypatch.setattr(
        "forgekit.plugins.compressors.gptq.importlib.util.find_spec", fake_find_spec
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"bits": 7},
    )
    with pytest.raises(ValidationError):
        GPTQCompressor().compress(ctx)
    assert "did" not in called, "backend check must not run when config is invalid"
