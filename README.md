# forgekit

A research-incubator pipeline for local LLM development: data → fine-tune → compress → evaluate → serve. Primary target is NVIDIA GB10 (Grace Blackwell, 128 GB unified memory); it also runs on consumer GPUs (8–24 GB), ROCm, Apple Silicon, and CPU.

**Positioning.** forgekit is not another trainer wrapper. It is a plugin surface sized for the modern compression zoo: LoRA/QLoRA training, AutoRound/GPTQ/AWQ/HQQ/bitsandbytes/AQLM/OneCompression, GGUF/vLLM/MLX export. New efficiency papers land here as runnable `Recipe` plugins within about a week of upstream code release.

If you want a polished SFT UI, use Unsloth Studio. If you want the newest paper runnable end-to-end, use forgekit.

## Supported recipes

| Stage       | Plugins                                                                                    |
| ----------- | ------------------------------------------------------------------------------------------ |
| Trainer     | `qlora`, `sft`, `dpo`, `grpo`, `full_finetune`\*, `qat`\*, `doc2lora`, `i_dlm` *(gated)*\*  |
| Compressor  | **`autoround`**, `gptq`, `awq`, `hqq`, `bnb`, `aqlm`, `onecompression`                      |
| Evaluator   | `lm_eval_harness`, `perplexity`                                                             |
| Exporter    | `gguf`, `vllm`, `mlx`, `i_dlm_isd` *(gated)*\*                                              |

\* still a `NotImplementedError` stub.

Trainer plugins are named by *method*. The underlying backend (`unsloth`, `trl`, `torchtune`, `transformers`) is selected per recipe via `config.backend`, defaulting to whatever fits the detected hardware.

### Which compressor?

- **`autoround`** — start here below 4 bits. It learns the rounding decision with signed gradient descent rather than applying a fixed rule, which is roughly level with GPTQ/AWQ at W4 and clearly ahead at W3/W2. It can also emit every downstream format (`llm_compressor`, `auto_gptq`, `auto_awq`, GGUF) and supports `NVFP4`/`MXFP4`.
- **`gptq` / `awq`** — fine at 4 bits, and the right choice if you already have a working recipe.

See [docs/compression_landscape.md](docs/compression_landscape.md) for the evidence behind that ordering, and for what was deliberately *not* adopted.

## Quickstart

```bash
just sync            # dev toolchain only
just doctor          # what hardware and backends did forgekit find?
just plugins         # every registered plugin
just run recipes/qwen3_4b_qlora_autoround.yaml
```

`just` requires the Nix dev shell (`nix develop`), which is where `uv` lives. Run `just` with no arguments to list every recipe. Without `just`, each recipe is a one-line `uv run …` — read the `justfile`.

`doctor` reports the detected arch, backend (`cuda`/`rocm`/`mps`/`cpu`), measured VRAM, which dtypes the device supports (bf16/fp8/fp4), and which optional extras are importable.

### Installing backends

```bash
just sync-ml                                    # trl + autoround + gptq (matches CI smoke)
uv sync --extra dev --extra gptq --extra gguf   # or pick your own
#   there is no `--extra all`: no single environment holds every backend
```

**`unsloth` and `vllm` are deliberately not extras.** Both work — `backend: unsloth` and the vLLM exporter activate whenever the package is importable — but neither can share a dependency resolution with the mainstream stack:

| Package | Caps | Against |
| --- | --- | --- |
| `unsloth` | `transformers<=5.5`, `datasets<4.4`, `trl<=0.24` | `llm-compressor` needs `transformers>=5.9`; `trl>=1.10` needs `datasets>=4.7` |
| `vllm` | `compressed-tensors==0.17.0`, `protobuf<7` | `llm-compressor` pins `==0.18.0`; `gptqmodel` needs `protobuf>=7.34` |

Give each its own environment:

```bash
uv venv .venv-unsloth && uv pip install --python .venv-unsloth 'unsloth>=2026.8' -e .
uv venv .venv-serve   && uv pip install --python .venv-serve   'vllm>=0.27'
```

You don't need `vllm` installed to *export* for it — the exporter writes a serving config and only imports `vllm` for an optional smoke test. Quantize in one environment, serve in another.

## Recipe shape

```yaml
name: qwen3-4b-qlora-autoround
model: Qwen/Qwen3-4B
data:
  kind: hf_dataset
  config: {}          # NOTE: top-level `data:` is a no-op in v0
trainer:
  kind: qlora
  config:
    backend: auto
    dataset: { path: tatsu-lab/alpaca, prompt_column: instruction, completion_column: output }
    lr: 2e-4
    steps: 200
    # omit micro_batch_size / seq_len — forgekit derives them from measured VRAM
compressors:
  - kind: autoround
    config: { scheme: W3A16, format: llm_compressor }
evaluator:
  kind: perplexity
  config: { dataset: wikitext2, bootstrap_resamples: 1000 }
exporter:
  kind: vllm
  config: {}
```

See `recipes/` for runnable examples and `docs/adding_a_plugin.md` for the incubator contract.

## Hardware

GB10 is the primary target and keeps a named profile with tuned defaults. Everything else is supported by *measuring* it rather than looking it up: VRAM comes from the driver and dtype support is queried, so an unrecognised card gets a correct profile instead of a zeroed placeholder.

Plugins branch on capabilities (`supports_bf16`, `supports_fp8`, `supports_fp4`), not on arch names. Two consequences worth knowing:

- Training dtype follows the device. A pre-Ampere or ROCm card gets fp16 instead of faulting on a hardcoded bf16.
- Requesting `scheme: NVFP4` on hardware without FP4 tensor cores is a hard error, not a silent slow path — the checkpoint would be dequantized to bf16 on every forward pass. Override with `allow_unsupported_scheme: true` when quantizing here to deploy elsewhere.

## On reading the evaluation numbers

The `perplexity` evaluator reports a bootstrap confidence interval alongside the point estimate. Use it: at realistic window counts the interval is around ±7%, so two methods differing by 0.1 perplexity are not distinguishable.

More importantly, **perplexity measures fluency, not capability**. Compression degrades reasoning well before it moves perplexity — quantized models lose accuracy while producing *longer* chains of thought. For any claim about whether a compressed model still reasons, use `lm_eval_harness`. Details and citations in [docs/compression_landscape.md](docs/compression_landscape.md).

## Development

```bash
just ci        # lockfile + ruff + mypy + pytest, exactly as CI runs them
just smoke     # end-to-end tests that really execute pipeline stages
just fmt       # ruff --fix + format
```

`just test` is CPU-only and needs no ML dependencies. `just smoke` needs the ML extras and pulls a few MB from HuggingFace; it is the only thing that catches upstream API drift, and it also runs weekly in CI.

## License

Apache-2.0. Plugins must be OSI-approved or CC BY / CC BY-SA; PHOTON-style non-commercial licenses are excluded. I-DLM plugins are gated behind the `[i-dlm]` extra and `FORGEKIT_ACCEPT_I_DLM_LICENSE=1` until upstream clarifies.
