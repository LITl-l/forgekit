"""GPTQ compressor — CPU-safe unit tests.

These tests never import `gptqmodel`, `transformers`, `peft`, `datasets`, or
`torch`. They exercise only config validation, registry resolution, and the
`gptqmodel` availability guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import unknown_profile
from forgekit.plugins.compressors.gptq import (
    GPTQCompressor,
    GPTQConfig,
    _require_gptqmodel,
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


def test_require_gptqmodel_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.gptq.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"gptqmodel.*not installed"):
        _require_gptqmodel()


def test_require_gptqmodel_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.gptq.importlib.util.find_spec",
        lambda name: object() if name == "gptqmodel" else None,
    )
    _require_gptqmodel()


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
    with pytest.raises(RuntimeError, match=r"gptqmodel.*not installed"):
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


def _install_fake_datasets(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> None:
    """Inject a stand-in `datasets` module so calibration can be tested on CPU CI."""
    import sys
    from types import SimpleNamespace

    class _FakeDS:
        column_names: ClassVar[list[str]] = ["text"]

        def __iter__(self) -> Any:
            return iter(rows)

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *_a, **_k: _FakeDS()),
    )


def test_calibration_returns_raw_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    """gptqmodel tokenizes internally — calibration must be a list[str].

    Regression guard for the auto-gptq migration: the old backend took
    pre-tokenized {"input_ids", "attention_mask"} dicts. Passing that shape to
    gptqmodel fails deep inside quantize(), so the contract is pinned here.
    """
    from forgekit.plugins.compressors.gptq import _build_calibration_texts

    _install_fake_datasets(monkeypatch, [{"text": "word " * 200} for _ in range(10)])
    texts = _build_calibration_texts(GPTQConfig.model_validate({}))

    assert isinstance(texts, list)
    assert texts and all(isinstance(t, str) for t in texts)


def test_calibration_drops_short_and_non_string_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[dict[str, Any]] = [
        {"text": "tiny"},
        {"text": None},
        {"text": 42},
        {"text": "word " * 200},
    ]
    _install_fake_datasets(monkeypatch, rows)
    from forgekit.plugins.compressors.gptq import _build_calibration_texts

    texts = _build_calibration_texts(GPTQConfig.model_validate({}))
    assert len(texts) == 1


def test_calibration_honours_num_samples_and_truncates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_datasets(monkeypatch, [{"text": "word " * 5000} for _ in range(50)])
    from forgekit.plugins.compressors.gptq import _build_calibration_texts

    cfg = GPTQConfig.model_validate(
        {"calibration": {"num_samples": 3, "seq_len": 128}}
    )
    texts = _build_calibration_texts(cfg)
    assert len(texts) == 3
    assert all(len(t) <= 128 * 4 for t in texts)


def test_calibration_empty_dataset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_datasets(monkeypatch, [{"text": "tiny"} for _ in range(5)])
    from forgekit.plugins.compressors.gptq import _build_calibration_texts

    with pytest.raises(RuntimeError, match="could not build any calibration"):
        _build_calibration_texts(GPTQConfig.model_validate({}))
