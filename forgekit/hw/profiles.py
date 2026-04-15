"""Hardware profile dataclass and suggested defaults.

Profiles are *hints* that plugins may consult for sensible batch / seq / quant
choices. The core does not enforce them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    arch: str  # e.g. "gb10", "rtx4090", "rtx3090", "unknown"
    vram_gb: float
    unified_memory: bool = False

    # Suggested defaults — plugins read these; they're hints only.
    suggested_micro_batch: int = 1
    suggested_seq_len: int = 2048
    suggested_quant_bits: int = 4


RTX3090 = HardwareProfile(
    arch="rtx3090",
    vram_gb=24.0,
    unified_memory=False,
    suggested_micro_batch=2,
    suggested_seq_len=2048,
    suggested_quant_bits=4,
)

RTX4090 = HardwareProfile(
    arch="rtx4090",
    vram_gb=24.0,
    unified_memory=False,
    suggested_micro_batch=4,
    suggested_seq_len=4096,
    suggested_quant_bits=4,
)

GB10_128G = HardwareProfile(
    arch="gb10",
    vram_gb=128.0,
    unified_memory=True,
    suggested_micro_batch=8,
    suggested_seq_len=8192,
    suggested_quant_bits=4,
)

PRESETS: dict[str, HardwareProfile] = {
    "rtx3090": RTX3090,
    "rtx4090": RTX4090,
    "gb10_128g": GB10_128G,
}


def unknown_profile() -> HardwareProfile:
    return HardwareProfile(arch="unknown", vram_gb=0.0, unified_memory=False)
