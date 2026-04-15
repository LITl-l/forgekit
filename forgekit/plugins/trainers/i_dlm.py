"""I-DLM trainer — Introspective Diffusion Language Model gated-LoRA conversion.

Converts an autoregressive checkpoint into an introspective-diffusion model
suitable for Introspective Strided Decoding (ISD) at inference time.

⚠️ License unverified — upstream has not clarified redistribution terms. Do not
redistribute outputs of this plugin until the license question is resolved.
Gated behind the ``[i-dlm]`` extra and the
``FORGEKIT_ACCEPT_I_DLM_LICENSE=1`` environment variable.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class IDLMTrainer:
    name: ClassVar[str] = "i_dlm"

    def train(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError(
            "i_dlm trainer is a scaffold stub and is license-gated. "
            "See forgekit/plugins/trainers/i_dlm.py for terms."
        )
