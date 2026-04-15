"""I-DLM ISD exporter — Introspective Strided Decoding serving artifacts.

Emits the bits needed to serve an I-DLM checkpoint with ISD inference
(2.9–4.1× throughput vs. the AR baseline in the paper).

⚠️ License unverified — upstream has not clarified redistribution terms. Do not
redistribute outputs of this plugin until the license question is resolved.
Gated behind the ``[i-dlm]`` extra and the
``FORGEKIT_ACCEPT_I_DLM_LICENSE=1`` environment variable.
"""

from __future__ import annotations

from typing import ClassVar

from forgekit.stages import StageContext


class IDLMISDExporter:
    name: ClassVar[str] = "i_dlm_isd"

    def export(self, ctx: StageContext) -> StageContext:
        raise NotImplementedError(
            "i_dlm_isd exporter is a scaffold stub and is license-gated."
        )
