"""AutoRound compressor — CPU-safe unit tests.

These tests never import `auto_round`, `transformers`, or `torch`. They cover
config validation, registry resolution, the backend guard, and the FP4 hardware
check that keeps forgekit usable away from its primary GB10 target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import GB10_128G, RTX3090, RTX5090, unknown_profile
from forgekit.plugins.compressors.autoround import (
    AutoRoundCompressor,
    AutoRoundConfig,
    _check_scheme_supported,
    _require_backend,
)
from forgekit.stages import StageContext


def _ctx(tmp_path: Path, hw: Any = None, **config: Any) -> StageContext:
    return StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=hw or unknown_profile(),
        stage_config=config,
    )


def test_registry_resolves_autoround() -> None:
    cls = registry.get("compressor", "autoround")
    assert cls is AutoRoundCompressor
    assert cls.name == "autoround"


# --- config ----------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = AutoRoundConfig.model_validate({})
    assert cfg.scheme == "W4A16"
    assert cfg.format == "auto_round"
    assert cfg.nsamples == 128
    assert cfg.iters == 200
    assert cfg.lr is None
    assert cfg.low_gpu_mem_usage is False
    assert cfg.output_subdir == "autoround"


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AutoRoundConfig.model_validate({"bogus": 1})


@pytest.mark.parametrize(
    "scheme", ["W2A16", "W3A16", "W4A16", "W8A16", "NVFP4", "MXFP4", "GGUF:Q4_K_M"]
)
def test_config_accepts_known_schemes(scheme: str) -> None:
    assert AutoRoundConfig.model_validate({"scheme": scheme}).scheme == scheme


def test_config_rejects_unknown_scheme() -> None:
    with pytest.raises(ValidationError, match="unknown scheme"):
        AutoRoundConfig.model_validate({"scheme": "W5A16"})


@pytest.mark.parametrize(
    "fmt", ["auto_round", "auto_gptq", "auto_awq", "llm_compressor", "gguf:q4_k_m"]
)
def test_config_accepts_known_formats(fmt: str) -> None:
    assert AutoRoundConfig.model_validate({"format": fmt}).format == fmt


def test_config_rejects_unknown_format() -> None:
    with pytest.raises(ValidationError, match="unknown format"):
        AutoRoundConfig.model_validate({"format": "safetensors"})


def test_config_rejects_nonsense_numbers() -> None:
    with pytest.raises(ValidationError, match="iters"):
        AutoRoundConfig.model_validate({"iters": -1})
    with pytest.raises(ValidationError, match="nsamples"):
        AutoRoundConfig.model_validate({"nsamples": 0})


def test_is_fp4_only_for_float_schemes() -> None:
    assert AutoRoundConfig.model_validate({"scheme": "NVFP4"}).is_fp4 is True
    assert AutoRoundConfig.model_validate({"scheme": "MXFP4"}).is_fp4 is True
    assert AutoRoundConfig.model_validate({"scheme": "W4A16"}).is_fp4 is False


# --- the hardware guard ----------------------------------------------------


def test_fp4_scheme_rejected_on_non_fp4_hardware(tmp_path: Path) -> None:
    """Requesting NVFP4 on an Ampere card must fail loudly, not silently.

    The resulting checkpoint would be dequantized to bf16 on every forward
    pass: smaller on disk, slower to run, and less accurate than W4A16.
    """
    cfg = AutoRoundConfig.model_validate({"scheme": "NVFP4"})
    with pytest.raises(RuntimeError, match=r"needs FP4 tensor cores"):
        _check_scheme_supported(cfg, _ctx(tmp_path, hw=RTX3090))


@pytest.mark.parametrize("hw", [GB10_128G, RTX5090])
def test_fp4_scheme_allowed_on_fp4_hardware(tmp_path: Path, hw: Any) -> None:
    cfg = AutoRoundConfig.model_validate({"scheme": "NVFP4"})
    _check_scheme_supported(cfg, _ctx(tmp_path, hw=hw))  # must not raise


def test_fp4_guard_can_be_overridden_for_cross_targeting(tmp_path: Path) -> None:
    """Quantizing on one box to deploy on another is legitimate."""
    cfg = AutoRoundConfig.model_validate(
        {"scheme": "NVFP4", "allow_unsupported_scheme": True}
    )
    _check_scheme_supported(cfg, _ctx(tmp_path, hw=RTX3090))  # must not raise


def test_int_schemes_never_hit_the_fp4_guard(tmp_path: Path) -> None:
    cfg = AutoRoundConfig.model_validate({"scheme": "W2A16"})
    _check_scheme_supported(cfg, _ctx(tmp_path, hw=unknown_profile()))


# --- backend guard ---------------------------------------------------------


def test_require_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.autoround.importlib.util.find_spec",
        lambda _n: None,
    )
    with pytest.raises(RuntimeError, match=r"auto-round.*not installed"):
        _require_backend()


def test_require_backend_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.autoround.importlib.util.find_spec",
        lambda name: object() if name == "auto_round" else None,
    )
    _require_backend()


def test_compress_missing_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.compressors.autoround.importlib.util.find_spec",
        lambda _n: None,
    )
    with pytest.raises(RuntimeError, match=r"auto-round.*not installed"):
        AutoRoundCompressor().compress(_ctx(tmp_path))


def test_compress_invalid_config_raises_before_backend_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, Any] = {}

    def fake_find_spec(_name: str) -> Any:
        called["did"] = True
        return object()

    monkeypatch.setattr(
        "forgekit.plugins.compressors.autoround.importlib.util.find_spec",
        fake_find_spec,
    )
    with pytest.raises(ValidationError):
        AutoRoundCompressor().compress(_ctx(tmp_path, scheme="W5A16"))
    assert "did" not in called, "backend check must not run when config is invalid"


def test_compress_checks_hardware_before_importing_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The FP4 guard must fire before any heavy import work."""
    monkeypatch.setattr(
        "forgekit.plugins.compressors.autoround.importlib.util.find_spec",
        lambda _n: object(),
    )
    with pytest.raises(RuntimeError, match=r"needs FP4 tensor cores"):
        AutoRoundCompressor().compress(_ctx(tmp_path, hw=RTX3090, scheme="NVFP4"))
