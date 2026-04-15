"""Plugin registry — entry_points-based discovery.

forgekit core never imports plugin modules directly. A plugin declares its
entry point in its (or forgekit's) `pyproject.toml` under one of:

    [project.entry-points."forgekit.trainers"]
    [project.entry-points."forgekit.compressors"]
    [project.entry-points."forgekit.evaluators"]
    [project.entry-points."forgekit.exporters"]

and this module loads them lazily via `importlib.metadata`.
"""

from __future__ import annotations

from functools import cache
from importlib.metadata import entry_points
from typing import Any

STAGE_GROUPS: dict[str, str] = {
    "trainer": "forgekit.trainers",
    "compressor": "forgekit.compressors",
    "evaluator": "forgekit.evaluators",
    "exporter": "forgekit.exporters",
}


@cache
def discover(group: str) -> dict[str, type[Any]]:
    """Load all plugin classes registered under the given entry_points group."""
    eps = entry_points(group=group)
    return {ep.name: ep.load() for ep in eps}


def get(stage_kind: str, name: str) -> type[Any]:
    """Resolve `(stage_kind, name)` → plugin class. Raises with the list of names on miss."""
    if stage_kind not in STAGE_GROUPS:
        raise KeyError(
            f"Unknown stage kind {stage_kind!r}. "
            f"Expected one of: {sorted(STAGE_GROUPS)}"
        )
    group = STAGE_GROUPS[stage_kind]
    plugins = discover(group)
    if name not in plugins:
        available = ", ".join(sorted(plugins)) or "(none registered)"
        raise KeyError(
            f"No {stage_kind} plugin named {name!r}. Registered: {available}"
        )
    return plugins[name]


def clear_cache() -> None:
    """Reset the entry_points cache (used by tests)."""
    discover.cache_clear()
