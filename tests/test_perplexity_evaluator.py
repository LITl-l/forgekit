"""Perplexity evaluator — CPU-safe unit tests.

These tests never import ``torch``, ``transformers``, or ``datasets``. They
exercise only registry resolution, config validation, dataset-alias resolution,
and the backend-availability guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.hw.profiles import unknown_profile
from forgekit.plugins.evaluators.perplexity import (
    PerplexityConfig,
    PerplexityEvaluator,
    _require_backend,
)
from forgekit.stages import StageContext


def test_registry_resolves_perplexity() -> None:
    cls = registry.get("evaluator", "perplexity")
    assert cls is PerplexityEvaluator
    assert cls.name == "perplexity"


def test_config_defaults() -> None:
    cfg = PerplexityConfig.model_validate({})
    assert cfg.dataset == "wikitext2"
    assert cfg.seq_len == 2048
    assert cfg.stride is None
    assert cfg.merge_adapter is True
    assert cfg.output_subdir == "perplexity"


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PerplexityConfig.model_validate({"bogus": 1})


def test_config_rejects_unknown_dataset_alias() -> None:
    with pytest.raises(ValidationError, match=r"unknown dataset alias"):
        PerplexityConfig.model_validate({"dataset": "not-a-real-dataset"})


def test_resolved_dataset_uses_alias_defaults() -> None:
    cfg = PerplexityConfig.model_validate({"dataset": "wikitext2"})
    assert cfg.resolved_dataset() == ("wikitext", "wikitext-2-raw-v1", "test", "text")


def test_resolved_dataset_custom_path_overrides_alias() -> None:
    cfg = PerplexityConfig.model_validate(
        {
            "dataset": "wikitext2",  # ignored because dataset_path is set
            "dataset_path": "my/dataset",
            "dataset_split": "validation",
            "text_column": "content",
        }
    )
    assert cfg.resolved_dataset() == ("my/dataset", None, "validation", "content")


def test_resolved_dataset_respects_explicit_overrides_on_alias() -> None:
    cfg = PerplexityConfig.model_validate(
        {"dataset": "wikitext2", "dataset_split": "validation"}
    )
    path, name, split, col = cfg.resolved_dataset()
    assert (path, name, split, col) == ("wikitext", "wikitext-2-raw-v1", "validation", "text")


def test_require_backend_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.evaluators.perplexity.importlib.util.find_spec",
        lambda _name: None,
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        _require_backend()


def test_require_backend_ok_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.evaluators.perplexity.importlib.util.find_spec",
        lambda _name: object(),
    )
    _require_backend()


def test_evaluate_missing_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.evaluators.perplexity.importlib.util.find_spec",
        lambda _name: None,
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={},
    )
    with pytest.raises(RuntimeError, match=r"required modules missing"):
        PerplexityEvaluator().evaluate(ctx)


def test_evaluate_invalid_config_raises_before_backend_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, Any] = {}

    def fake_find_spec(_name: str) -> Any:
        called["did"] = True
        return object()

    monkeypatch.setattr(
        "forgekit.plugins.evaluators.perplexity.importlib.util.find_spec", fake_find_spec
    )
    ctx = StageContext(
        recipe_name="t",
        model_path="dummy/model",
        work_dir=tmp_path,
        hw=unknown_profile(),
        stage_config={"bogus": True},
    )
    with pytest.raises(ValidationError):
        PerplexityEvaluator().evaluate(ctx)
    assert "did" not in called, "backend check must not run when config is invalid"


# --- bootstrap confidence interval -----------------------------------------


def test_weighted_mean_respects_token_weights() -> None:
    from forgekit.plugins.evaluators.perplexity import weighted_mean

    # A short trailing window must not count the same as a full one.
    assert weighted_mean([1.0, 3.0], [1, 3]) == pytest.approx(2.5)
    assert weighted_mean([1.0, 3.0], [1, 1]) == pytest.approx(2.0)


def test_weighted_mean_falls_back_when_weights_are_zero() -> None:
    from forgekit.plugins.evaluators.perplexity import weighted_mean

    assert weighted_mean([1.0, 3.0], [0, 0]) == pytest.approx(2.0)


def test_ci_is_none_below_window_threshold() -> None:
    from forgekit.plugins.evaluators.perplexity import bootstrap_perplexity_ci

    assert (
        bootstrap_perplexity_ci(
            [0.5] * 4, [10] * 4, resamples=100, ci_level=0.95, seed=0
        )
        is None
    )


def test_ci_is_none_when_resamples_disabled() -> None:
    from forgekit.plugins.evaluators.perplexity import bootstrap_perplexity_ci

    assert (
        bootstrap_perplexity_ci([0.5] * 20, [10] * 20, resamples=0, ci_level=0.95, seed=0)
        is None
    )


def test_ci_brackets_the_point_estimate() -> None:
    import math
    import random

    from forgekit.plugins.evaluators.perplexity import (
        bootstrap_perplexity_ci,
        weighted_mean,
    )

    rng = random.Random(1)
    nlls = [rng.gauss(2.0, 0.3) for _ in range(60)]
    weights = [rng.randint(50, 2048) for _ in range(60)]
    point = math.exp(weighted_mean(nlls, weights))

    ci = bootstrap_perplexity_ci(nlls, weights, resamples=1000, ci_level=0.95, seed=7)
    assert ci is not None
    low, high = ci
    assert low < point < high


def test_ci_collapses_when_every_window_agrees() -> None:
    import math

    from forgekit.plugins.evaluators.perplexity import bootstrap_perplexity_ci

    ci = bootstrap_perplexity_ci([0.5] * 20, [10] * 20, resamples=200, ci_level=0.95, seed=0)
    assert ci is not None
    assert ci[0] == pytest.approx(math.exp(0.5))
    assert ci[1] == pytest.approx(math.exp(0.5))


def test_ci_is_deterministic_for_a_fixed_seed() -> None:
    from forgekit.plugins.evaluators.perplexity import bootstrap_perplexity_ci

    nlls = [0.4, 0.6, 0.5, 0.7, 0.3, 0.55, 0.62, 0.48]
    weights = [100] * 8
    a = bootstrap_perplexity_ci(nlls, weights, resamples=300, ci_level=0.95, seed=11)
    b = bootstrap_perplexity_ci(nlls, weights, resamples=300, ci_level=0.95, seed=11)
    assert a == b


def test_higher_confidence_level_widens_the_interval() -> None:
    import random

    from forgekit.plugins.evaluators.perplexity import bootstrap_perplexity_ci

    rng = random.Random(3)
    nlls = [rng.gauss(2.0, 0.4) for _ in range(50)]
    weights = [512] * 50

    narrow = bootstrap_perplexity_ci(nlls, weights, resamples=1000, ci_level=0.80, seed=5)
    wide = bootstrap_perplexity_ci(nlls, weights, resamples=1000, ci_level=0.99, seed=5)
    assert narrow is not None and wide is not None
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_config_validates_ci_level_bounds() -> None:
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValidationError):
            PerplexityConfig.model_validate({"ci_level": bad})


def test_config_rejects_negative_resamples() -> None:
    with pytest.raises(ValidationError):
        PerplexityConfig.model_validate({"bootstrap_resamples": -1})


def test_bootstrap_defaults_are_on() -> None:
    cfg = PerplexityConfig.model_validate({})
    assert cfg.bootstrap_resamples == 1000
    assert cfg.ci_level == pytest.approx(0.95)
