"""CLI tests — actually invoke the commands.

Nothing in the suite previously executed the CLI, so a broken command surfaced
only when a user ran it. These need no ML dependencies: `doctor` degrades to the
torch-missing path, and `list-plugins` exercises the full entry-points discovery
chain (which transitively imports every registered plugin module).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from forgekit.cli import app

runner = CliRunner()


def test_version_runs() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "forgekit" in result.stdout


def test_doctor_runs_without_ml_deps() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    # The capability rows are what plugins branch on, so doctor must show them.
    for field in ("arch:", "backend:", "vram:", "dtypes:", "train as:"):
        assert field in result.stdout


def test_doctor_reports_optional_extras() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "optional extras" in result.stdout
    # Migrated backends must be the ones surfaced, not the dead ones.
    assert "auto-round" in result.stdout
    assert "gptqmodel" in result.stdout
    assert "auto-gptq" not in result.stdout
    assert "autoawq" not in result.stdout


def test_list_plugins_loads_every_entry_point() -> None:
    """Exercises `ep.load()` for all four groups — catches import-time breakage."""
    result = runner.invoke(app, ["list-plugins"])
    assert result.exit_code == 0
    for expected in ("qlora", "gptq", "awq", "autoround", "perplexity", "gguf"):
        assert expected in result.stdout


def test_list_recipes_validates_shipped_recipes(tmp_path: Path) -> None:
    repo_recipes = Path(__file__).resolve().parent.parent / "recipes"
    result = runner.invoke(app, ["list-recipes", "--dir", str(repo_recipes)])
    assert result.exit_code == 0
    # Every shipped recipe must parse; `list-recipes` prints "invalid" otherwise.
    assert "invalid" not in result.stdout


def test_list_recipes_missing_dir_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["list-recipes", "--dir", str(tmp_path / "nope")])
    assert result.exit_code == 1


def test_run_rejects_missing_recipe(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", str(tmp_path / "absent.yaml")])
    assert result.exit_code != 0


def test_run_fails_on_unknown_plugin(tmp_path: Path) -> None:
    """A typo'd plugin name must not succeed.

    Only asserts failure, not presentation: `_run_stage` currently catches just
    NotImplementedError, so a KeyError from the registry still surfaces as a
    traceback. Tightening that is a separate change.
    """
    recipe = tmp_path / "bad.yaml"
    recipe.write_text(
        "name: bad\n"
        "model: dummy/model\n"
        "data:\n  kind: hf_dataset\n  config: {}\n"
        "trainer:\n  kind: qlroa\n  config: {}\n"
    )
    result = runner.invoke(app, ["run", str(recipe), "--work-dir", str(tmp_path / "out")])
    assert result.exit_code != 0


@pytest.mark.parametrize("command", ["doctor", "list-plugins", "version"])
def test_commands_are_reachable(command: str) -> None:
    assert runner.invoke(app, [command]).exit_code == 0
