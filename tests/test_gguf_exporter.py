"""gguf exporter — CPU-safe unit tests.

These tests never invoke ``subprocess.run`` or touch the real llama.cpp
binaries. They exercise config validation, registry resolution, the backend
guard, and the auto-detect path for ``convert_hf_to_gguf.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import unknown_profile
from forgekit.plugins.exporters.gguf import (
    GGUFConfig,
    GGUFExporter,
    _find_convert_script,
    _require_backend,
)
from forgekit.stages import StageContext


def test_registry_resolves_gguf() -> None:
    cls = registry.get("exporter", "gguf")
    assert cls is GGUFExporter
    assert cls.name == "gguf"


def test_config_defaults() -> None:
    cfg = GGUFConfig.model_validate({})
    assert cfg.quant == "q4_k_m"
    assert cfg.convert_script is None
    assert cfg.quantize_binary is None
    assert cfg.output_subdir == "gguf"
    assert cfg.filename is None
    assert cfg.merge_adapter is True


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        GGUFConfig.model_validate({"bogus": 1})


def test_config_rejects_invalid_quant() -> None:
    with pytest.raises(ValidationError):
        GGUFConfig.model_validate({"quant": "q7_k"})


def test_config_accepts_valid_quants() -> None:
    for quant in (
        "f32",
        "f16",
        "bf16",
        "q8_0",
        "q4_0",
        "q4_1",
        "q5_0",
        "q5_1",
        "q4_k_m",
        "q4_k_s",
        "q5_k_m",
        "q5_k_s",
        "q6_k",
        "q3_k_m",
        "q2_k",
    ):
        cfg = GGUFConfig.model_validate({"quant": quant})
        assert cfg.quant == quant


def test_require_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.exporters.gguf.importlib.util.find_spec", lambda _n: None
    )

    def _raise(_explicit: Any) -> Any:
        raise RuntimeError("no script")

    monkeypatch.setattr("forgekit.plugins.exporters.gguf._find_convert_script", _raise)
    with pytest.raises(RuntimeError, match=r"install `forgekit\[gguf\]`"):
        _require_backend()


def test_require_backend_ok_when_gguf_module_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.exporters.gguf.importlib.util.find_spec",
        lambda name: object() if name == "gguf" else None,
    )

    def _should_not_be_called(_explicit: Any) -> Any:
        raise AssertionError("should short-circuit on gguf module")

    monkeypatch.setattr(
        "forgekit.plugins.exporters.gguf._find_convert_script", _should_not_be_called
    )
    _require_backend()


def test_require_backend_ok_when_convert_script_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.exporters.gguf.importlib.util.find_spec", lambda _n: None
    )
    fake_script = tmp_path / "convert_hf_to_gguf.py"
    fake_script.write_text("# fake\n")
    monkeypatch.setattr(
        "forgekit.plugins.exporters.gguf._find_convert_script", lambda _explicit: fake_script
    )
    _require_backend()


def test_find_convert_script_uses_explicit_when_exists(tmp_path: Path) -> None:
    fake_script = tmp_path / "convert_hf_to_gguf.py"
    fake_script.write_text("# fake\n")
    resolved = _find_convert_script(str(fake_script))
    assert resolved == fake_script


def test_find_convert_script_raises_when_nothing_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("forgekit.plugins.exporters.gguf.shutil.which", lambda _n: None)
    monkeypatch.setattr(
        "forgekit.plugins.exporters.gguf.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"could not locate `convert_hf_to_gguf.py`"):
        _find_convert_script(None)


def test_export_missing_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.exporters.gguf.importlib.util.find_spec", lambda _n: None
    )

    def _raise(_explicit: Any) -> Any:
        raise RuntimeError("no script")

    monkeypatch.setattr("forgekit.plugins.exporters.gguf._find_convert_script", _raise)
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={},
    )
    with pytest.raises(RuntimeError, match=r"install `forgekit\[gguf\]`"):
        GGUFExporter().export(ctx)


def test_export_backend_present_but_convert_script_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Backend check passes (gguf module present), but the conversion script is absent."""
    monkeypatch.setattr(
        "forgekit.plugins.exporters.gguf._require_backend", lambda: None
    )

    def _raise(_explicit: Any) -> Any:
        raise RuntimeError("script gone")

    monkeypatch.setattr("forgekit.plugins.exporters.gguf._find_convert_script", _raise)
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={},
    )
    with pytest.raises(RuntimeError, match=r"script gone"):
        GGUFExporter().export(ctx)


def test_export_invalid_config_raises_before_backend_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, Any] = {}

    def fake_find_spec(_n: str) -> Any:
        called["did"] = True
        return object()

    monkeypatch.setattr(
        "forgekit.plugins.exporters.gguf.importlib.util.find_spec", fake_find_spec
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"quant": "bogus_quant"},
    )
    with pytest.raises(ValidationError):
        GGUFExporter().export(ctx)
    assert "did" not in called, "backend check must not run when config is invalid"
