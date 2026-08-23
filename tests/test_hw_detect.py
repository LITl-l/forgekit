"""Hardware detection tests — mock out torch and verify measured profiles.

The key behavioural change these cover: an unrecognised device is no longer
reported as ``arch="unknown", vram_gb=0.0``. It gets a real profile built from
measured VRAM and queried dtype support, with a cosmetic ``cuda-smXY`` label.
GB10 remains the named primary target.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from forgekit.hw import detect as detect_mod
from forgekit.hw.profiles import build_profile, unknown_profile


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    cc: tuple[int, int] | None,
    vram_gb: float = 24.0,
    cuda_available: bool = True,
    bf16: bool = True,
    hip: str | None = None,
    is_integrated: bool = False,
    mps_available: bool = False,
) -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return cuda_available

        @staticmethod
        def is_bf16_supported() -> bool:
            return bf16

        @staticmethod
        def get_device_capability(_idx: int = 0) -> tuple[int, int]:
            assert cc is not None
            return cc

        @staticmethod
        def get_device_properties(_idx: int = 0) -> object:
            return SimpleNamespace(
                total_memory=int(vram_gb * (1024**3)), is_integrated=is_integrated
            )

    class _FakeMps:
        @staticmethod
        def is_available() -> bool:
            return mps_available

    fake_torch = SimpleNamespace(
        cuda=_FakeCuda,
        version=SimpleNamespace(hip=hip),
        backends=SimpleNamespace(mps=_FakeMps),
        mps=SimpleNamespace(recommended_max_memory=lambda: 32 * (1024**3)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _block_torch_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch blocked for test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("builtins.__import__", _blocked_import)
    importlib.reload(detect_mod)


# --- named archs -----------------------------------------------------------


def test_detects_gb10_as_primary_target(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, (12, 1), vram_gb=128.0)
    profile = detect_mod.detect()
    assert profile.arch == "gb10"
    assert profile.unified_memory is True
    assert profile.supports_fp4 is True
    assert profile.supports_fp8 is True
    assert profile.vram_gb == pytest.approx(128.0, abs=0.5)


@pytest.mark.parametrize(
    ("cc", "expected"),
    [
        ((12, 0), "rtx5090"),
        ((9, 0), "h100"),
        ((8, 9), "rtx4090"),
        ((8, 6), "rtx3090"),
        ((8, 0), "a100"),
    ],
)
def test_detects_named_archs(
    monkeypatch: pytest.MonkeyPatch, cc: tuple[int, int], expected: str
) -> None:
    _install_fake_torch(monkeypatch, cc)
    assert detect_mod.detect().arch == expected


# --- the regression this change is about -----------------------------------


def test_unnamed_cc_still_gets_a_real_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """An arch with no preset must report measured VRAM, not 0.0.

    Previously anything outside a three-entry table fell through to
    `unknown_profile()`, so plugins sized batches against zero VRAM.
    """
    _install_fake_torch(monkeypatch, (7, 5), vram_gb=16.0, bf16=False)
    diag = detect_mod.diagnose()
    assert diag.status is detect_mod.DetectionStatus.UNNAMED_ARCH
    assert diag.profile.arch == "cuda-sm75"
    assert diag.profile.vram_gb == pytest.approx(16.0, abs=0.5)
    assert diag.profile.suggested_micro_batch >= 1
    assert diag.profile.suggested_seq_len >= 1024
    assert "sm_75" in diag.detail


def test_pre_ampere_does_not_claim_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, (7, 5), vram_gb=16.0, bf16=False)
    profile = detect_mod.detect()
    assert profile.supports_bf16 is False
    assert profile.training_dtype == "fp16"


def test_fp8_and_fp4_gated_by_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, (8, 6))  # Ampere: bf16 yes, fp8/fp4 no
    ampere = detect_mod.detect()
    assert (ampere.supports_bf16, ampere.supports_fp8, ampere.supports_fp4) == (
        True,
        False,
        False,
    )

    _install_fake_torch(monkeypatch, (8, 9))  # Ada: fp8 yes, fp4 no
    ada = detect_mod.detect()
    assert (ada.supports_fp8, ada.supports_fp4) == (True, False)


def test_rocm_reports_rocm_backend_and_no_fp4(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, (9, 4), vram_gb=64.0, hip="6.2.0")
    profile = detect_mod.detect()
    assert profile.backend == "rocm"
    assert profile.arch == "rocm-sm94"
    assert profile.supports_fp4 is False
    assert profile.supports_fp8 is False


def test_apple_silicon_detected_when_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, cc=None, cuda_available=False, mps_available=True)
    diag = detect_mod.diagnose()
    assert diag.status is detect_mod.DetectionStatus.OK
    assert diag.profile.backend == "mps"
    assert diag.profile.unified_memory is True
    assert diag.profile.vram_gb == pytest.approx(32.0, abs=0.5)


# --- fallbacks -------------------------------------------------------------


def test_no_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, cc=None, cuda_available=False)
    diag = detect_mod.diagnose()
    assert diag.status is detect_mod.DetectionStatus.NO_ACCELERATOR
    assert diag.profile.arch == "unknown"
    assert diag.profile.backend == "cpu"


def test_no_torch_returns_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_torch_import(monkeypatch)
    assert detect_mod.detect().arch == "unknown"


def test_diagnose_torch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_torch_import(monkeypatch)
    diag = detect_mod.diagnose()
    assert diag.status is detect_mod.DetectionStatus.TORCH_MISSING
    assert diag.profile.arch == "unknown"
    assert "torch" in diag.detail.lower()


# --- profile arithmetic ----------------------------------------------------


def test_usable_vram_discounts_unified_memory() -> None:
    discrete = build_profile(arch="d", vram_gb=100.0, backend="cuda")
    unified = build_profile(arch="u", vram_gb=100.0, backend="cuda", unified_memory=True)
    assert discrete.usable_vram_gb == pytest.approx(90.0)
    assert unified.usable_vram_gb == pytest.approx(70.0)


def test_suggestions_scale_with_vram() -> None:
    small = build_profile(arch="s", vram_gb=8.0, backend="cuda")
    large = build_profile(arch="l", vram_gb=80.0, backend="cuda")
    assert large.suggested_micro_batch > small.suggested_micro_batch
    assert large.suggested_seq_len >= small.suggested_seq_len


def test_unknown_profile_is_cpu_safe() -> None:
    p = unknown_profile()
    assert p.backend == "cpu"
    assert p.training_dtype == "fp32"
    assert p.suggested_micro_batch == 1
