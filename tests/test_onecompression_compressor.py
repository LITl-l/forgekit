"""OneCompression compressor — CPU-safe unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import unknown_profile
from forgekit.plugins.compressors.onecompression import (
    OneCompressionCompressor,
    OneCompressionConfig,
    _find_onecompression_script,
    _require_backend,
)
from forgekit.stages import StageContext


def test_registry_resolves_onecompression() -> None:
    cls = registry.get("compressor", "onecompression")
    assert cls is OneCompressionCompressor
    assert cls.name == "onecompression"


def test_config_defaults() -> None:
    cfg = OneCompressionConfig.model_validate({})
    assert cfg.mode == "autobit"
    assert cfg.target_avg_bits == pytest.approx(4.0)
    assert cfg.min_bits == 2
    assert cfg.max_bits == 8
    assert cfg.calibration.path == "wikitext"
    assert cfg.output_subdir == "onecompression"


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        OneCompressionConfig.model_validate({"bogus": 1})


def test_config_rejects_invalid_mode() -> None:
    with pytest.raises(ValidationError):
        OneCompressionConfig.model_validate({"mode": "bogus"})


def test_config_rejects_invalid_min_bits() -> None:
    with pytest.raises(ValidationError):
        OneCompressionConfig.model_validate({"min_bits": 1})


def test_config_rejects_invalid_max_bits() -> None:
    with pytest.raises(ValidationError):
        OneCompressionConfig.model_validate({"max_bits": 16})


def test_require_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.onecompression.importlib.util.find_spec",
        lambda _n: None,
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        _require_backend()


def test_require_backend_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.onecompression.importlib.util.find_spec",
        lambda _n: object(),
    )
    _require_backend()


def test_find_script_explicit_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"does not exist"):
        _find_onecompression_script(str(tmp_path / "nonexistent.py"))


def test_find_script_explicit_present(tmp_path: Path) -> None:
    script = tmp_path / "main.py"
    script.write_text("")
    assert _find_onecompression_script(str(script)) == str(script)


def test_find_script_none_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.onecompression.shutil.which", lambda _n: None
    )
    monkeypatch.setattr(
        "forgekit.plugins.compressors.onecompression._module_available",
        lambda _n: False,
    )
    with pytest.raises(RuntimeError, match=r"could not locate"):
        _find_onecompression_script(None)


def test_compress_missing_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.onecompression.importlib.util.find_spec",
        lambda _n: None,
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={},
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        OneCompressionCompressor().compress(ctx)


def test_compress_invalid_config_raises_before_backend_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, Any] = {}

    def fake_find_spec(_name: str) -> Any:
        called["did"] = True
        return object()

    monkeypatch.setattr(
        "forgekit.plugins.compressors.onecompression.importlib.util.find_spec",
        fake_find_spec,
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"mode": "bogus"},
    )
    with pytest.raises(ValidationError):
        OneCompressionCompressor().compress(ctx)
    assert "did" not in called
