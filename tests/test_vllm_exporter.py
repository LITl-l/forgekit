"""vLLM exporter — CPU-safe unit tests.

These tests never import ``torch``, ``transformers``, or ``vllm``. They
exercise only registry resolution, config validation, and the backend
availability guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import unknown_profile
from forgekit.plugins.exporters.vllm import (
    VLLMConfig,
    VLLMExporter,
    _require_backend,
)
from forgekit.stages import StageContext


def test_registry_resolves_vllm() -> None:
    cls = registry.get("exporter", "vllm")
    assert cls is VLLMExporter
    assert cls.name == "vllm"


def test_config_defaults() -> None:
    cfg = VLLMConfig.model_validate({})
    assert cfg.output_subdir == "vllm"
    assert cfg.merge_adapter is True
    assert cfg.dtype == "auto"
    assert cfg.max_model_len is None
    assert cfg.tensor_parallel_size == 1
    assert cfg.gpu_memory_utilization == pytest.approx(0.9)
    assert cfg.trust_remote_code is False
    assert cfg.smoke_test is False
    assert cfg.quantization == "auto"


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        VLLMConfig.model_validate({"bogus": 1})


def test_config_rejects_invalid_dtype() -> None:
    with pytest.raises(ValidationError):
        VLLMConfig.model_validate({"dtype": "int8"})


def test_config_rejects_invalid_quantization() -> None:
    with pytest.raises(ValidationError):
        VLLMConfig.model_validate({"quantization": "int4"})


def test_require_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.exporters.vllm.importlib.util.find_spec",
        lambda _name: None,
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        _require_backend(smoke_test=False)


def test_require_backend_ok_when_present_without_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # transformers + torch present, vllm absent — acceptable when smoke_test=False.
    def fake_find_spec(name: str) -> Any:
        return None if name == "vllm" else object()

    monkeypatch.setattr(
        "forgekit.plugins.exporters.vllm.importlib.util.find_spec", fake_find_spec
    )
    _require_backend(smoke_test=False)


def test_require_backend_smoke_test_requires_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # vllm missing must raise when smoke_test=True.
    def fake_find_spec(name: str) -> Any:
        return None if name == "vllm" else object()

    monkeypatch.setattr(
        "forgekit.plugins.exporters.vllm.importlib.util.find_spec", fake_find_spec
    )
    with pytest.raises(RuntimeError, match=r"vllm"):
        _require_backend(smoke_test=True)


def test_export_missing_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.exporters.vllm.importlib.util.find_spec",
        lambda _name: None,
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={},
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        VLLMExporter().export(ctx)


def test_export_invalid_config_raises_before_backend_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, Any] = {}

    def fake_find_spec(_name: str) -> Any:
        called["did"] = True
        return object()

    monkeypatch.setattr(
        "forgekit.plugins.exporters.vllm.importlib.util.find_spec", fake_find_spec
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"bogus": True},
    )
    with pytest.raises(ValidationError):
        VLLMExporter().export(ctx)
    assert "did" not in called, "backend check must not run when config is invalid"
