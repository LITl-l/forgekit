"""Hardware profile dataclass and capability-derived defaults.

Profiles are *hints* that plugins may consult for sensible batch / seq / quant
choices. The core does not enforce them.

Design note — why this is derived, not a lookup table
-----------------------------------------------------
An earlier version mapped three compute capabilities (sm_121 / sm_89 / sm_86)
to three hand-written presets and returned ``vram_gb=0.0`` for everything else.
That is wrong for the majority of machines forgekit targets: an RTX 3060, an
A100, a 5090, an AMD card, or an Apple M-series box all fell through to a
profile claiming zero VRAM, and then plugins sized their batches off it.

Profiles are now computed from *measured* VRAM plus *queried* dtype support.
Named presets are kept only as recognisable labels; the numbers come from the
device. Adding a new GPU requires no code change.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Accelerator backends forgekit knows how to size for.
Backend = str  # "cuda" | "rocm" | "mps" | "cpu"


@dataclass(frozen=True)
class HardwareProfile:
    arch: str  # e.g. "gb10", "rtx4090", "cuda-sm90", "mps", "cpu"
    vram_gb: float
    unified_memory: bool = False
    backend: Backend = "cpu"

    # Dtype support, queried from the device rather than assumed. Plugins should
    # branch on these instead of on `arch` — that is what keeps forgekit working
    # on hardware nobody has added a preset for.
    supports_bf16: bool = False
    supports_fp8: bool = False
    supports_fp4: bool = False

    # Suggested defaults — plugins read these; they're hints only.
    suggested_micro_batch: int = 1
    suggested_seq_len: int = 2048
    suggested_quant_bits: int = 4

    @property
    def usable_vram_gb(self) -> float:
        """Memory a training job may plan against.

        Unified-memory devices (GB10, Apple Silicon) report the whole system
        pool, but the OS and CPU-side allocations live there too, so budget
        70%. Discrete GPUs get 90% to leave room for fragmentation and the
        CUDA context.
        """
        return self.vram_gb * (0.70 if self.unified_memory else 0.90)

    @property
    def training_dtype(self) -> str:
        """The widest training dtype this device actually supports.

        Returns ``"bf16"``, ``"fp16"``, or ``"fp32"``. Trainers should call this
        rather than hardcoding ``bf16=True`` — pre-Ampere CUDA cards and most
        CPU setups will fault or silently fall back otherwise.
        """
        if self.supports_bf16:
            return "bf16"
        if self.backend in ("cuda", "rocm"):
            return "fp16"
        return "fp32"


def _suggest_for_vram(usable_gb: float) -> tuple[int, int]:
    """Map usable VRAM to (micro_batch, seq_len).

    Deliberately coarse. These are starting points that avoid an immediate OOM,
    not tuned throughput settings.
    """
    if usable_gb <= 0:  # unknown / CPU — stay minimal.
        return 1, 1024
    if usable_gb < 6:
        return 1, 1024
    if usable_gb < 11:
        return 1, 2048
    if usable_gb < 22:
        return 2, 2048
    if usable_gb < 44:
        return 4, 4096
    if usable_gb < 90:
        return 8, 8192
    return 16, 8192


def build_profile(
    *,
    arch: str,
    vram_gb: float,
    backend: Backend,
    unified_memory: bool = False,
    supports_bf16: bool = False,
    supports_fp8: bool = False,
    supports_fp4: bool = False,
) -> HardwareProfile:
    """Construct a profile, deriving batch/seq suggestions from usable VRAM."""
    probe = HardwareProfile(
        arch=arch,
        vram_gb=vram_gb,
        unified_memory=unified_memory,
        backend=backend,
        supports_bf16=supports_bf16,
        supports_fp8=supports_fp8,
        supports_fp4=supports_fp4,
    )
    micro_batch, seq_len = _suggest_for_vram(probe.usable_vram_gb)
    return replace(probe, suggested_micro_batch=micro_batch, suggested_seq_len=seq_len)


# --- Named presets ---------------------------------------------------------
# These exist so `--hardware.profile rtx4090` and test fixtures have stable
# names. Detection does not depend on them; it measures the real device.

RTX3090 = build_profile(
    arch="rtx3090", vram_gb=24.0, backend="cuda", supports_bf16=True
)
RTX4090 = build_profile(
    arch="rtx4090", vram_gb=24.0, backend="cuda", supports_bf16=True, supports_fp8=True
)
RTX5090 = build_profile(
    arch="rtx5090",
    vram_gb=32.0,
    backend="cuda",
    supports_bf16=True,
    supports_fp8=True,
    supports_fp4=True,
)
A100_80G = build_profile(
    arch="a100_80g", vram_gb=80.0, backend="cuda", supports_bf16=True
)
H100_80G = build_profile(
    arch="h100_80g", vram_gb=80.0, backend="cuda", supports_bf16=True, supports_fp8=True
)
GB10_128G = build_profile(
    arch="gb10",
    vram_gb=128.0,
    backend="cuda",
    unified_memory=True,
    supports_bf16=True,
    supports_fp8=True,
    supports_fp4=True,
)
APPLE_M_32G = build_profile(
    arch="mps", vram_gb=32.0, backend="mps", unified_memory=True, supports_bf16=True
)

PRESETS: dict[str, HardwareProfile] = {
    "rtx3090": RTX3090,
    "rtx4090": RTX4090,
    "rtx5090": RTX5090,
    "a100_80g": A100_80G,
    "h100_80g": H100_80G,
    "gb10_128g": GB10_128G,
    "apple_m_32g": APPLE_M_32G,
}


def unknown_profile() -> HardwareProfile:
    """Fallback when no accelerator can be identified — CPU-safe defaults."""
    return build_profile(arch="unknown", vram_gb=0.0, backend="cpu")
