"""Lazy CUDA / arch detection.

`torch` is imported inside the function so CPU-only CI (which doesn't install
torch) still passes the rest of the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from forgekit.hw.profiles import GB10_128G, RTX3090, RTX4090, HardwareProfile, unknown_profile

# Compute-capability → arch label.
_CC_TO_ARCH: dict[tuple[int, int], str] = {
    (12, 1): "gb10",  # Grace Blackwell (sm_121)
    (8, 9): "rtx4090",  # Ada Lovelace
    (8, 6): "rtx3090",  # Ampere
}


class DetectionStatus(StrEnum):
    OK = "ok"
    TORCH_MISSING = "torch-missing"
    CUDA_UNAVAILABLE = "cuda-unavailable"
    UNRECOGNIZED_CC = "unrecognized-cc"


@dataclass(frozen=True)
class Diagnostic:
    profile: HardwareProfile
    status: DetectionStatus
    detail: str = ""


def diagnose() -> Diagnostic:
    """Detect hardware and return a diagnostic explaining the result."""
    try:
        import torch
    except ImportError:
        return Diagnostic(
            profile=unknown_profile(),
            status=DetectionStatus.TORCH_MISSING,
            detail="`import torch` failed — install a torch build matching your hardware",
        )

    if not torch.cuda.is_available():
        return Diagnostic(
            profile=unknown_profile(),
            status=DetectionStatus.CUDA_UNAVAILABLE,
            detail="torch is installed but `torch.cuda.is_available()` is False",
        )

    cc = torch.cuda.get_device_capability(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024**3)
    arch = _CC_TO_ARCH.get((cc[0], cc[1]))

    if arch == "gb10":
        return Diagnostic(profile=GB10_128G, status=DetectionStatus.OK)
    if arch == "rtx4090":
        return Diagnostic(profile=RTX4090, status=DetectionStatus.OK)
    if arch == "rtx3090":
        return Diagnostic(profile=RTX3090, status=DetectionStatus.OK)

    return Diagnostic(
        profile=HardwareProfile(arch="unknown", vram_gb=vram_gb, unified_memory=False),
        status=DetectionStatus.UNRECOGNIZED_CC,
        detail=f"compute capability sm_{cc[0]}{cc[1]} is not in the known arch map",
    )


def detect() -> HardwareProfile:
    """Return the hardware profile of CUDA device 0, or an 'unknown' profile."""
    return diagnose().profile
