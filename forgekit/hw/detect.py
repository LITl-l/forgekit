"""Lazy CUDA / arch detection.

`torch` is imported inside the function so CPU-only CI (which doesn't install
torch) still passes the rest of the test suite.
"""

from __future__ import annotations

from forgekit.hw.profiles import GB10_128G, RTX3090, RTX4090, HardwareProfile, unknown_profile

# Compute-capability → arch label.
_CC_TO_ARCH: dict[tuple[int, int], str] = {
    (12, 1): "gb10",  # Grace Blackwell (sm_121)
    (8, 9): "rtx4090",  # Ada Lovelace
    (8, 6): "rtx3090",  # Ampere
}


def detect() -> HardwareProfile:
    """Return the hardware profile of CUDA device 0, or an 'unknown' profile."""
    try:
        import torch
    except ImportError:
        return unknown_profile()

    if not torch.cuda.is_available():
        return unknown_profile()

    cc = torch.cuda.get_device_capability(0)
    arch = _CC_TO_ARCH.get((cc[0], cc[1]), "unknown")
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024**3)

    if arch == "gb10":
        return GB10_128G
    if arch == "rtx4090":
        return RTX4090
    if arch == "rtx3090":
        return RTX3090
    return HardwareProfile(arch="unknown", vram_gb=vram_gb, unified_memory=False)
