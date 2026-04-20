"""QLoRA trainer — CPU-safe unit tests.

These tests never import `unsloth`, `trl`, `transformers`, `peft`, or `torch`.
They only exercise the plugin's config validation and backend-resolution logic.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from forgekit import registry
from forgekit.plugins.trainers.qlora import (
    QLoRAConfig,
    QLoRATrainer,
    _make_formatting_func,
    _resolve_backend,
)


def test_registry_resolves_qlora() -> None:
    cls = registry.get("trainer", "qlora")
    assert cls is QLoRATrainer
    assert cls.name == "qlora"


def test_config_requires_dataset() -> None:
    with pytest.raises(ValidationError):
        QLoRAConfig.model_validate({})


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        QLoRAConfig.model_validate(
            {"dataset": {"path": "tatsu-lab/alpaca"}, "bogus_field": 1}
        )


def test_config_defaults_populated() -> None:
    cfg = QLoRAConfig.model_validate({"dataset": {"path": "tatsu-lab/alpaca"}})
    assert cfg.backend == "auto"
    assert cfg.lr == pytest.approx(2e-4)
    assert cfg.steps == 200
    assert cfg.lora_r == 16
    assert cfg.dataset.split == "train"
    assert cfg.dataset.text_column == "text"


def test_resolve_backend_prefers_unsloth_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_find_spec(name: str) -> Any:
        return object() if name == "unsloth" else None

    monkeypatch.setattr(
        "forgekit.plugins.trainers.qlora.importlib.util.find_spec", fake_find_spec
    )
    assert _resolve_backend("auto") == "unsloth"


def test_resolve_backend_falls_back_to_trl(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_find_spec(name: str) -> Any:
        return object() if name == "trl" else None

    monkeypatch.setattr(
        "forgekit.plugins.trainers.qlora.importlib.util.find_spec", fake_find_spec
    )
    assert _resolve_backend("auto") == "trl"


def test_resolve_backend_raises_when_no_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.qlora.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match="no backend available"):
        _resolve_backend("auto")


def test_resolve_backend_explicit_unsloth_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.qlora.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"unsloth.*not installed"):
        _resolve_backend("unsloth")


def test_resolve_backend_explicit_trl_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "forgekit.plugins.trainers.qlora.importlib.util.find_spec", lambda _n: None
    )
    with pytest.raises(RuntimeError, match=r"trl.*not installed"):
        _resolve_backend("trl")


def test_dataset_prompt_without_completion_rejected() -> None:
    with pytest.raises(ValidationError, match="set together"):
        QLoRAConfig.model_validate(
            {
                "dataset": {
                    "path": "any",
                    "prompt_column": "question",
                },
            }
        )


def test_dataset_completion_without_prompt_rejected() -> None:
    with pytest.raises(ValidationError, match="set together"):
        QLoRAConfig.model_validate(
            {
                "dataset": {
                    "path": "any",
                    "completion_column": "answer",
                },
            }
        )


def test_dataset_prompt_completion_mode_detected() -> None:
    cfg = QLoRAConfig.model_validate(
        {
            "dataset": {
                "path": "any",
                "prompt_column": "question",
                "completion_column": "answer",
            },
        }
    )
    assert cfg.dataset.is_prompt_completion is True
    assert cfg.dataset.prompt_column == "question"
    assert cfg.dataset.completion_column == "answer"


def test_dataset_text_column_mode_by_default() -> None:
    cfg = QLoRAConfig.model_validate({"dataset": {"path": "any"}})
    assert cfg.dataset.is_prompt_completion is False
    assert cfg.dataset.text_column == "text"


def test_formatting_func_none_in_text_column_mode() -> None:
    cfg = QLoRAConfig.model_validate({"dataset": {"path": "any"}})
    assert _make_formatting_func(cfg, tokenizer=object()) is None


class _FakeTokenizer:
    """Minimal stand-in for an HF tokenizer with a chat template."""

    chat_template = "fake"

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool = False
    ) -> str:
        assert tokenize is False
        parts = [f"[{m['role']}]{m['content']}" for m in messages]
        return "".join(parts)


class _NoTemplateTokenizer:
    """Tokenizer that lacks a chat_template — represents base-model checkpoints."""

    chat_template = None


def test_formatting_func_uses_chat_template_when_present() -> None:
    cfg = QLoRAConfig.model_validate(
        {
            "dataset": {
                "path": "any",
                "prompt_column": "q",
                "completion_column": "a",
            },
        }
    )
    fmt = _make_formatting_func(cfg, _FakeTokenizer())
    assert fmt is not None
    out = fmt({"q": "hi", "a": "bye"})
    assert out == "[user]hi[assistant]bye"


def test_formatting_func_falls_back_when_no_template() -> None:
    cfg = QLoRAConfig.model_validate(
        {
            "dataset": {
                "path": "any",
                "prompt_column": "q",
                "completion_column": "a",
            },
        }
    )
    fmt = _make_formatting_func(cfg, _NoTemplateTokenizer())
    assert fmt is not None
    assert fmt({"q": "hi", "a": "bye"}) == "hi\n\nbye"
