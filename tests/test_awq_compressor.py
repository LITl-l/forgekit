"""AWQ compressor — CPU-safe unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import unknown_profile
from forgekit.plugins.compressors.awq import AWQCompressor, AWQConfig, _require_backend
from forgekit.stages import StageContext


def test_registry_resolves_awq() -> None:
    cls = registry.get("compressor", "awq")
    assert cls is AWQCompressor
    assert cls.name == "awq"


def test_config_defaults() -> None:
    cfg = AWQConfig.model_validate({})
    assert cfg.bits == 4
    assert cfg.group_size == 128
    assert cfg.symmetric is False
    assert cfg.scheme == "W4A16_ASYM"
    assert cfg.ignore == ["lm_head"]
    assert cfg.calibration.path == "mit-han-lab/pile-val-backup"
    assert cfg.calibration.num_samples == 128
    assert cfg.output_subdir == "awq"


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AWQConfig.model_validate({"bogus": 1})


def test_config_rejects_invalid_bits() -> None:
    with pytest.raises(ValidationError):
        AWQConfig.model_validate({"bits": 8})


def test_symmetric_selects_scheme() -> None:
    assert AWQConfig.model_validate({"symmetric": True}).scheme == "W4A16"
    assert AWQConfig.model_validate({"symmetric": False}).scheme == "W4A16_ASYM"


def test_config_rejects_removed_autoawq_fields() -> None:
    """`version` and `zero_point` were AutoAWQ kernel knobs with no successor.

    compressed-tensors defers kernel choice to the serving runtime, so these
    must be rejected rather than silently ignored.
    """
    for dead in ("version", "zero_point"):
        with pytest.raises(ValidationError):
            AWQConfig.model_validate({dead: "gemm"})


def test_require_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.awq.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        _require_backend()


def test_require_backend_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.awq.importlib.util.find_spec", lambda _n: object()
    )
    _require_backend()


def test_compress_missing_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.awq.importlib.util.find_spec", lambda _n: None
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={},
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        AWQCompressor().compress(ctx)


def test_compress_invalid_config_raises_before_backend_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, Any] = {}

    def fake_find_spec(_name: str) -> Any:
        called["did"] = True
        return object()

    monkeypatch.setattr(
        "forgekit.plugins.compressors.awq.importlib.util.find_spec", fake_find_spec
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"bits": 8},
    )
    with pytest.raises(ValidationError):
        AWQCompressor().compress(ctx)
    assert "did" not in called
