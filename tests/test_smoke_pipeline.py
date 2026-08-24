"""End-to-end smoke tests — the only tests that really execute a pipeline stage.

Everything else in this suite validates config and mocks the backend away. That
is fast and CPU-safe, but it cannot catch the failure mode this repository has
actually suffered: a plugin merged green against an upstream API that had moved.
Both compressor backends were pinned to libraries that are now archived, and the
qlora trainer called a TRL signature that no longer exists. No unit test noticed,
because none of them ran the code.

These are deselected by default (`addopts = -m 'not smoke'`). Run them with
`just smoke`, or `pytest -m smoke`. They need the ML extras and pull a few MB
from HuggingFace on first run.

Deliberately uses `hf-internal-testing` tiny random models: real architecture,
seconds to run, no meaningful download.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from forgekit.hw import detect as hw_detect
from forgekit.stages import StageContext

pytestmark = pytest.mark.smoke

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _have(*modules: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in modules)


needs_hf = pytest.mark.skipif(
    not _have("torch", "transformers", "datasets"),
    reason="needs `uv sync --extra trl` (torch + transformers + datasets)",
)


def _ctx(tmp_path: Path, **config: object) -> StageContext:
    return StageContext(
        recipe_name="smoke",
        model_path=TINY_MODEL,
        work_dir=tmp_path,
        hw=hw_detect.detect(),
        stage_config=dict(config),
    )


@needs_hf
def test_perplexity_evaluator_really_runs(tmp_path: Path) -> None:
    """Load a model, score a corpus, and produce a report with an interval."""
    import json

    from forgekit.plugins.evaluators.perplexity import PerplexityEvaluator

    ctx = _ctx(
        tmp_path,
        dataset="wikitext2",
        seq_len=64,
        stride=32,
        max_samples=300,
        bootstrap_resamples=200,
        device="cpu",
    )
    out = PerplexityEvaluator().evaluate(ctx)

    ppl = out.artifacts["perplexity"]
    assert math.isfinite(ppl) and ppl > 0, ppl

    report_path = Path(out.artifacts["perplexity_report_path"])
    assert report_path.exists()
    report = json.loads(report_path.read_text())

    assert report["n_windows"] >= 1
    assert report["n_scored_tokens"] > 0
    # With this many windows the bootstrap must have produced an interval, and
    # it must bracket the point estimate.
    if report["n_windows"] >= 5:
        assert report["ci_low"] is not None and report["ci_high"] is not None
        assert report["ci_low"] < report["perplexity"] < report["ci_high"]
        assert out.artifacts["perplexity_ci"][0] < ppl


@needs_hf
def test_perplexity_ci_narrows_with_more_data(tmp_path: Path) -> None:
    """A sanity check on the statistics: more windows, tighter interval."""
    from forgekit.plugins.evaluators.perplexity import PerplexityEvaluator

    def width(max_samples: int, sub: str) -> float:
        ctx = _ctx(
            tmp_path / sub,
            dataset="wikitext2",
            seq_len=64,
            stride=64,
            max_samples=max_samples,
            bootstrap_resamples=300,
            device="cpu",
            output_subdir=sub,
        )
        out = PerplexityEvaluator().evaluate(ctx)
        lo, hi = out.artifacts["perplexity_ci"]
        return hi - lo

    assert width(1200, "big") < width(200, "small")


@pytest.mark.skipif(
    not _have("gptqmodel", "transformers", "datasets"),
    reason="needs `uv sync --extra gptq`",
)
def test_gptq_compressor_really_runs(tmp_path: Path) -> None:
    """Quantize a tiny model with gptqmodel and produce a loadable directory."""
    from forgekit.plugins.compressors.gptq import GPTQCompressor

    ctx = _ctx(
        tmp_path,
        bits=4,
        group_size=16,  # tiny models have small hidden dims
        calibration={"num_samples": 8, "seq_len": 64},
    )
    out = GPTQCompressor().compress(ctx)

    out_dir = Path(out.artifacts["gptq_model_path"])
    assert out_dir.is_dir()
    assert any(out_dir.iterdir()), "quantized checkpoint directory is empty"
    assert out.model_path == str(out_dir)
    assert out.artifacts["quantized_model_path"] == str(out_dir)


@pytest.mark.skipif(
    not _have("auto_round", "transformers"),
    reason="needs `uv sync --extra autoround`",
)
def test_autoround_compressor_really_runs(tmp_path: Path) -> None:
    """Quantize a tiny model with AutoRound at a low iteration count."""
    from forgekit.plugins.compressors.autoround import AutoRoundCompressor

    ctx = _ctx(
        tmp_path,
        scheme="W4A16",
        format="auto_round",
        nsamples=8,
        iters=2,  # enough to exercise the search loop, not to get a good model
        seq_len=64,
    )
    out = AutoRoundCompressor().compress(ctx)

    out_dir = Path(out.artifacts["autoround_model_path"])
    assert out_dir.is_dir()
    assert any(out_dir.iterdir()), "quantized checkpoint directory is empty"
    assert out.artifacts["autoround_scheme"] == "W4A16"


def test_hardware_detection_runs_on_this_machine() -> None:
    """Whatever this box is, detection must return a coherent profile.

    No skip guard: this is the check that forgekit works away from GB10.
    """
    diag = hw_detect.diagnose()
    p = diag.profile

    assert p.arch
    assert p.backend in ("cuda", "rocm", "mps", "cpu")
    assert p.vram_gb >= 0.0
    assert p.suggested_micro_batch >= 1
    assert p.suggested_seq_len >= 1024
    assert p.training_dtype in ("bf16", "fp16", "fp32")
    # A device that reports FP4 must also report FP8 — no real hardware has the
    # newer format without the older one.
    if p.supports_fp4:
        assert p.supports_fp8
    # Only a live accelerator may claim a non-zero pool.
    if p.backend == "cpu":
        assert p.vram_gb == 0.0
