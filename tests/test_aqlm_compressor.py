"""AQLM compressor — CPU-safe unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import unknown_profile
from forgekit.plugins.compressors.aqlm import (
    AQLMCompressor,
    AQLMConfig,
    _find_aqlm_script,
    _require_backend,
)
from forgekit.stages import StageContext


def test_registry_resolves_aqlm() -> None:
    cls = registry.get("compressor", "aqlm")
    assert cls is AQLMCompressor
    assert cls.name == "aqlm"


def test_config_defaults() -> None:
    cfg = AQLMConfig.model_validate({})
    assert cfg.num_codebooks == 1
    assert cfg.nbits_per_codebook == 16
    assert cfg.calibration.path == "allenai/c4"
    assert cfg.calibration.num_samples == 1024
    assert cfg.output_subdir == "aqlm"


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AQLMConfig.model_validate({"bogus": 1})


def test_config_rejects_invalid_num_codebooks() -> None:
    with pytest.raises(ValidationError):
        AQLMConfig.model_validate({"num_codebooks": 3})


def test_config_rejects_invalid_nbits() -> None:
    with pytest.raises(ValidationError):
        AQLMConfig.model_validate({"nbits_per_codebook": 4})


def test_require_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.aqlm.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        _require_backend()


def test_require_backend_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.aqlm.importlib.util.find_spec", lambda _n: object()
    )
    _require_backend()


def test_find_aqlm_script_explicit_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"does not exist"):
        _find_aqlm_script(str(tmp_path / "nonexistent.py"))


def test_find_aqlm_script_explicit_present(tmp_path: Path) -> None:
    script = tmp_path / "main.py"
    script.write_text("")
    assert _find_aqlm_script(str(script)) == str(script)


def test_find_aqlm_script_none_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.aqlm.shutil.which", lambda _n: None
    )
    monkeypatch.setattr(
        "forgekit.plugins.compressors.aqlm._module_available", lambda _n: False
    )
    with pytest.raises(RuntimeError, match=r"could not locate"):
        _find_aqlm_script(None)


def test_compress_missing_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.aqlm.importlib.util.find_spec", lambda _n: None
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={},
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        AQLMCompressor().compress(ctx)


def test_compress_invalid_config_raises_before_backend_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, Any] = {}

    def fake_find_spec(_name: str) -> Any:
        called["did"] = True
        return object()

    monkeypatch.setattr(
        "forgekit.plugins.compressors.aqlm.importlib.util.find_spec", fake_find_spec
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"num_codebooks": 3},
    )
    with pytest.raises(ValidationError):
        AQLMCompressor().compress(ctx)
    assert "did" not in called
