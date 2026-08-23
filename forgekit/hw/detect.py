"""Accelerator detection — capability-first, with named archs as a courtesy.

`torch` is imported inside the functions so CPU-only CI (which doesn't install
torch) still passes the rest of the test suite.

GB10 (Grace Blackwell, sm_121) is forgekit's primary target and keeps a named
profile with tuned defaults. Every other device is supported by *measuring* it:
VRAM comes from the driver and dtype support is queried, so an unrecognised
card gets correct numbers rather than a zeroed-out placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from forgekit.hw.profiles import HardwareProfile, build_profile, unknown_profile

# Compute capability → friendly arch label. Purely cosmetic: a CC that is not
# in this map still yields a fully populated profile, labelled "cuda-smXY".
_CC_TO_ARCH: dict[tuple[int, int], str] = {
    (12, 1): "gb10",  # Grace Blackwell — primary target
    (12, 0): "rtx5090",  # Blackwell consumer
    (9, 0): "h100",  # Hopper
    (8, 9): "rtx4090",  # Ada Lovelace
    (8, 6): "rtx3090",  # Ampere consumer
    (8, 0): "a100",  # Ampere datacenter
}

# Minimum compute capability for native tensor-core support of each format.
_FP8_MIN_CC = (8, 9)  # Ada / Hopper and later
_FP4_MIN_CC = (10, 0)  # Blackwell and later (includes sm_120 / sm_121)


class DetectionStatus(StrEnum):
    OK = "ok"
    TORCH_MISSING = "torch-missing"
    NO_ACCELERATOR = "no-accelerator"
    UNNAMED_ARCH = "unnamed-arch"


@dataclass(frozen=True)
class Diagnostic:
    profile: HardwareProfile
    status: DetectionStatus
    detail: str = ""


def diagnose() -> Diagnostic:
    """Detect the best available accelerator and explain the result."""
    try:
        import torch
    except ImportError:
        return Diagnostic(
            profile=unknown_profile(),
            status=DetectionStatus.TORCH_MISSING,
            detail="`import torch` failed — install a torch build matching your hardware",
        )

    if torch.cuda.is_available():
        return _diagnose_cuda(torch)

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return _diagnose_mps(torch)

    return Diagnostic(
        profile=unknown_profile(),
        status=DetectionStatus.NO_ACCELERATOR,
        detail=(
            "torch is installed but no CUDA, ROCm, or MPS device is visible — "
            "running on CPU"
        ),
    )


def _diagnose_cuda(torch: object) -> Diagnostic:
    """Build a profile from a live CUDA (or ROCm) device."""
    cuda = torch.cuda  # type: ignore[attr-defined]
    major, minor = cuda.get_device_capability(0)
    props = cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)

    # ROCm reports through the CUDA API but sets torch.version.hip.
    is_rocm = getattr(torch.version, "hip", None) is not None  # type: ignore[attr-defined]
    backend = "rocm" if is_rocm else "cuda"

    # Ask the runtime rather than inferring from the capability tuple; this is
    # the check that keeps pre-Ampere and ROCm cards from attempting bf16.
    try:
        supports_bf16 = bool(cuda.is_bf16_supported())
    except Exception:  # older / exotic builds lack the helper
        supports_bf16 = major >= 8

    cc = (major, minor)
    arch = _CC_TO_ARCH.get(cc)
    named = arch is not None

    # GB10's unified pool is the whole 128 GB system memory. Detect it by the
    # integrated flag where torch exposes it, else by the known capability.
    unified = bool(getattr(props, "is_integrated", False)) or cc == (12, 1)

    profile = build_profile(
        arch=arch or f"{backend}-sm{major}{minor}",
        vram_gb=vram_gb,
        backend=backend,
        unified_memory=unified,
        supports_bf16=supports_bf16,
        supports_fp8=(not is_rocm) and cc >= _FP8_MIN_CC,
        supports_fp4=(not is_rocm) and cc >= _FP4_MIN_CC,
    )

    if named:
        return Diagnostic(profile=profile, status=DetectionStatus.OK)
    return Diagnostic(
        profile=profile,
        status=DetectionStatus.UNNAMED_ARCH,
        detail=(
            f"sm_{major}{minor} has no friendly name in forgekit, so the profile "
            f"is labelled {profile.arch!r}. VRAM and dtype support were measured "
            "from the device and are correct — this is informational only."
        ),
    )


def _diagnose_mps(torch: object) -> Diagnostic:
    """Build a profile for Apple Silicon.

    MPS shares system RAM, so `recommended_max_memory` (when available) is the
    honest number rather than any dedicated-VRAM figure.
    """
    vram_gb = 0.0
    recommended = getattr(torch.mps, "recommended_max_memory", None)  # type: ignore[attr-defined]
    if callable(recommended):
        try:
            vram_gb = float(recommended()) / (1024**3)
        except Exception:  # helper is version-dependent
            vram_gb = 0.0

    profile = build_profile(
        arch="mps",
        vram_gb=vram_gb,
        backend="mps",
        unified_memory=True,
        # M-series supports bf16 from macOS 14 / torch 2.3 onward. fp8/fp4 have
        # no MPS kernels at any version.
        supports_bf16=True,
    )
    return Diagnostic(profile=profile, status=DetectionStatus.OK)


def detect() -> HardwareProfile:
    """Return the profile of the best available accelerator, or a CPU fallback."""
    return diagnose().profile
