# Adding a plugin — the forgekit incubator contract

forgekit is a research incubator. A new paper becomes a runnable recipe by
adding one plugin class and one `pyproject.toml` entry. No core changes.

## Protocol to implement

Pick a stage and implement the matching Protocol from `forgekit.stages.*`:

| Stage       | Protocol            | Method      | Location                           |
| ----------- | ------------------- | ----------- | ---------------------------------- |
| data        | `DataPlugin`        | `prepare`   | `forgekit/stages/data.py`          |
| trainer     | `TrainerPlugin`     | `train`     | `forgekit/stages/trainer.py`       |
| compressor  | `CompressorPlugin`  | `compress`  | `forgekit/stages/compressor.py`    |
| evaluator   | `EvaluatorPlugin`   | `evaluate`  | `forgekit/stages/evaluator.py`     |
| exporter    | `ExporterPlugin`    | `export`    | `forgekit/stages/exporter.py`      |

Each method takes a `StageContext` (from `forgekit.stages`) and returns a
`StageContext`. The class must expose `name: ClassVar[str]` matching the
entry-point name.

```python
# forgekit/plugins/compressors/my_quant.py
from typing import ClassVar
from forgekit.stages import StageContext


class MyQuantCompressor:
    name: ClassVar[str] = "my_quant"

    def compress(self, ctx: StageContext) -> StageContext:
        # read ctx.stage_config, do the work, write to ctx.work_dir
        # return ctx with ctx.model_path updated to the quantized checkpoint
        ...
```

## Register via entry points

In `pyproject.toml`:

```toml
[project.entry-points."forgekit.compressors"]
my_quant = "forgekit.plugins.compressors.my_quant:MyQuantCompressor"
```

Groups are `forgekit.trainers`, `forgekit.compressors`, `forgekit.evaluators`,
`forgekit.exporters`. `data` is reserved and not yet an entry-points group.

## License checklist

A plugin may be merged only if its upstream dependencies are:

- **OSI-approved** (Apache-2.0, MIT, BSD, GPL/LGPL with clear linkage), **or**
- **Creative Commons BY / BY-SA** (not NC, not ND).

Non-commercial / no-derivatives licenses (e.g. CC BY-NC-ND, PHOTON) are
rejected. If the upstream license is unclear, the plugin is gated behind an
opt-in environment variable (see `FORGEKIT_ACCEPT_I_DLM_LICENSE` for the
pattern) and must not ship in `[all]`.

## Version pinning

- Pin upstream to a minor version (`>=X.Y,<X.(Y+1)`) unless the project
  guarantees semver. Plugins should break loudly, not silently.
- If the upstream is pre-1.0, pin to an exact commit in a git URL.

## PR template

Open the PR with:

- Link to the upstream paper / repo / license.
- A minimal recipe under `recipes/` that exercises the plugin end-to-end.
- A unit test that at minimum confirms `registry.get(<stage>, <name>)` resolves
  the class.
- A note in `README.md` under the recipes matrix if the plugin adds a new row.

## Design guidelines

- Delegate ML work to the upstream library — forgekit is glue, not training code.
- Make `config` fields explicit via a pydantic model inside the plugin module
  (validate on entry). Do not silently accept unknown keys.
- Treat `HardwareProfile` as a hint, not an enforcement surface. Pick sensible
  defaults per arch; don't raise on mismatch unless the plugin cannot run.
- No global state. Plugins are constructed per-run.
