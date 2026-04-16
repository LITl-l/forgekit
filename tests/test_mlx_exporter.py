"""MLX exporter — CPU-safe unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import unknown_profile
from forgekit.plugins.exporters.mlx import MLXConfig, MLXExporter, _require_backend
from forgekit.stages import StageContext


def test_registry_resolves_mlx() -> None:
    cls = registry.get("exporter", "mlx")
    assert cls is MLXExporter
    assert cls.name == "mlx"


def test_config_defaults() -> None:
    cfg = MLXConfig.model_validate({})
    assert cfg.quantize is False
    assert cfg.q_bits == 4
    assert cfg.q_group_size == 64
    assert cfg.dtype == "float16"
    assert cfg.output_subdir == "mlx"


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MLXConfig.model_validate({"bogus": 1})


def test_config_rejects_invalid_q_bits() -> None:
    with pytest.raises(ValidationError):
        MLXConfig.model_validate({"q_bits": 3})


def test_config_accepts_valid_q_bits() -> None:
    for b in (2, 4, 8):
        cfg = MLXConfig.model_validate({"q_bits": b})
        assert cfg.q_bits == b


def test_config_rejects_invalid_dtype() -> None:
    with pytest.raises(ValidationError):
        MLXConfig.model_validate({"dtype": "int8"})


def test_require_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.exporters.mlx.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        _require_backend()


def test_require_backend_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.exporters.mlx.importlib.util.find_spec", lambda _n: object()
    )
    _require_backend()


def test_export_missing_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.exporters.mlx.importlib.util.find_spec", lambda _n: None
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={},
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        MLXExporter().export(ctx)


def test_export_invalid_config_raises_before_backend_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, Any] = {}

    def fake_find_spec(_name: str) -> Any:
        called["did"] = True
        return object()

    monkeypatch.setattr(
        "forgekit.plugins.exporters.mlx.importlib.util.find_spec", fake_find_spec
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"q_bits": 3},
    )
    with pytest.raises(ValidationError):
        MLXExporter().export(ctx)
    assert "did" not in called
