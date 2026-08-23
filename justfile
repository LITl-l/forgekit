# forgekit task runner.
#
# Every recipe assumes you are inside the Nix dev shell, which is where `uv`
# lives. Enter it once per terminal:
#
#     nix develop
#
# ...or prefix a single command: `nix develop --command just test`.
#
# The `lint`, `types`, and `test` recipes mirror .github/workflows/ci.yml
# step-for-step so local runs and CI cannot drift.

set shell := ["bash", "-uc"]

# Show available recipes.
default:
    @just --list

# --- environment -----------------------------------------------------------

# Install the dev toolchain (ruff, mypy, pytest). No ML deps.
sync:
    uv sync --extra dev

# Install dev + the extras needed to actually run a pipeline on this box.
# The default deliberately matches the CI `smoke` job's extras, so that
# `just sync-ml && just smoke` covers the same tests CI does — otherwise a
# smoke test guarded on a missing backend silently skips here and runs there.
# Override to taste: `just sync-ml EXTRAS="--extra trl --extra autoround"`
sync-ml EXTRAS="--extra trl --extra autoround --extra gptq":
    uv sync --extra dev {{EXTRAS}}

# Verify the lockfile still matches pyproject.toml (CI runs this too).
lock-check:
    uv lock --check

# --- quality gates ---------------------------------------------------------

# Lint (mirrors CI).
lint:
    uv run ruff check .

# Auto-fix what ruff can, then format.
fmt:
    uv run ruff check --fix .
    uv run ruff format .

# Type-check (mirrors CI).
types:
    uv run mypy forgekit/

# Unit tests — CPU-only, no ML deps required (mirrors CI).
test *ARGS:
    uv run pytest {{ARGS}}

# The full CI gate, in CI's order. Run this before opening a PR.
ci: lock-check lint types test

# --- integration -----------------------------------------------------------

# End-to-end smoke test against a tiny random model. Requires ML extras
# (`just sync-ml`) and downloads ~5 MB from HuggingFace on first run.
# This is the only test that proves the pipeline actually executes.
smoke:
    uv run pytest -m smoke --no-header -v

# Everything: unit tests plus the real-execution smoke test.
verify: ci smoke

# --- pipeline --------------------------------------------------------------

# Report detected hardware and which optional extras are importable.
doctor:
    uv run forgekit doctor

# List every registered plugin across all four stage groups.
plugins:
    uv run forgekit list-plugins

# List recipe YAMLs and whether each one still validates.
recipes:
    uv run forgekit list-recipes

# Run a recipe end-to-end: `just run recipes/qwen3_4b_qlora_gptq.yaml`
run RECIPE *ARGS:
    uv run forgekit run {{RECIPE}} {{ARGS}}

# Validate every recipe in recipes/ without executing any of them.
check-recipes:
    uv run forgekit list-recipes --dir recipes

# --- housekeeping ----------------------------------------------------------

# Remove pipeline outputs and tool caches. Does not touch the venv.
clean:
    rm -rf outputs/ .pytest_cache/ .ruff_cache/ .mypy_cache/
    find . -type d -name __pycache__ -prune -exec rm -rf {} +
