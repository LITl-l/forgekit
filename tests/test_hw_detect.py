"""Hardware detection tests — mock out torch.cuda and verify arch labels."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from forgekit.hw import detect as detect_mod


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    cc: tuple[int, int] | None,
    vram_gb: float = 24.0,
    cuda_available: bool = True,
) -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return cuda_available

        @staticmethod
        def get_device_capability(_idx: int = 0) -> tuple[int, int]:
            assert cc is not None
            return cc

        @staticmethod
        def get_device_properties(_idx: int = 0) -> object:
            return SimpleNamespace(total_memory=int(vram_gb * (1024**3)))

    fake_torch = SimpleNamespace(cuda=_FakeCuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_detects_gb10(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, (12, 1), vram_gb=128.0)
    profile = detect_mod.detect()
    assert profile.arch == "gb10"
    assert profile.unified_memory is True


def test_detects_rtx4090(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, (8, 9))
    profile = detect_mod.detect()
    assert profile.arch == "rtx4090"


def test_detects_rtx3090(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, (8, 6))
    profile = detect_mod.detect()
    assert profile.arch == "rtx3090"


def test_unknown_cc_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, (9, 0))
    profile = detect_mod.detect()
    assert profile.arch == "unknown"


def test_no_torch_returns_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate torch being unimportable by injecting a meta_path finder that blocks it.
    import importlib

    monkeypatch.delitem(sys.modules, "torch", raising=False)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch blocked for test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("builtins.__import__", _blocked_import)
    # Reload detect module so the ImportError is raised from within `detect()`.
    importlib.reload(detect_mod)
    profile = detect_mod.detect()
    assert profile.arch == "unknown"


def test_diagnose_ok_on_known_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, (12, 1), vram_gb=128.0)
    diag = detect_mod.diagnose()
    assert diag.status is detect_mod.DetectionStatus.OK
    assert diag.profile.arch == "gb10"
    assert diag.detail == ""


def test_diagnose_torch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch blocked for test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("builtins.__import__", _blocked_import)
    importlib.reload(detect_mod)
    diag = detect_mod.diagnose()
    assert diag.status is detect_mod.DetectionStatus.TORCH_MISSING
    assert diag.profile.arch == "unknown"
    assert "torch" in diag.detail.lower()


def test_diagnose_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, cc=None, cuda_available=False)
    diag = detect_mod.diagnose()
    assert diag.status is detect_mod.DetectionStatus.CUDA_UNAVAILABLE
    assert diag.profile.arch == "unknown"


def test_diagnose_unrecognized_cc(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch, (9, 0), vram_gb=80.0)
    diag = detect_mod.diagnose()
    assert diag.status is detect_mod.DetectionStatus.UNRECOGNIZED_CC
    assert diag.profile.arch == "unknown"
    assert "sm_90" in diag.detail
