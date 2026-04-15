"""Registry discovery tests — checks every stub is reachable by entry-point name."""

from __future__ import annotations

import pytest

from forgekit import registry

EXPECTED: dict[str, set[str]] = {
    "trainer": {
        "qlora",
        "sft",
        "dpo",
        "grpo",
        "full_finetune",
        "qat",
        "doc2lora",
        "i_dlm",
    },
    "compressor": {"onecompression", "gptq", "awq", "hqq", "bnb", "aqlm"},
    "evaluator": {"lm_eval_harness", "perplexity"},
    "exporter": {"gguf", "vllm", "mlx", "i_dlm_isd"},
}


@pytest.mark.parametrize("stage_kind,expected_names", list(EXPECTED.items()))
def test_every_stub_is_registered(stage_kind: str, expected_names: set[str]) -> None:
    group = registry.STAGE_GROUPS[stage_kind]
    found = set(registry.discover(group))
    missing = expected_names - found
    assert not missing, f"missing {stage_kind} plugins: {missing}"


def test_get_returns_class_with_matching_name() -> None:
    cls = registry.get("trainer", "qlora")
    assert cls.name == "qlora"


def test_get_unknown_stage_kind_raises_useful_error() -> None:
    with pytest.raises(KeyError, match="Unknown stage kind"):
        registry.get("nonsense", "qlora")


def test_get_unknown_plugin_name_lists_available() -> None:
    with pytest.raises(KeyError) as excinfo:
        registry.get("trainer", "does_not_exist")
    assert "qlora" in str(excinfo.value)
