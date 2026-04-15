# forgekit

A research-incubator pipeline for local LLM development: data → fine-tune → compress → evaluate → serve. Runs on consumer GPUs (8–24 GB) and NVIDIA GB10 (Grace Blackwell, 128 GB unified memory).

**Positioning.** forgekit is not another trainer wrapper. It is a plugin surface sized for the modern compression zoo: LoRA/QLoRA training, GPTQ/AWQ/HQQ/bitsandbytes/AQLM/OneCompression, TriAttention at inference, GGUF/vLLM/MLX export. New efficiency papers land here as runnable `Recipe` plugins within about a week of upstream code release.

If you want a polished SFT UI, use Unsloth Studio. If you want the newest paper runnable end-to-end, use forgekit.

## Supported recipes

| Stage       | Plugins (stubs — follow-up PRs land implementations)                                  |
| ----------- | ------------------------------------------------------------------------------------- |
| Trainer     | `qlora`, `sft`, `dpo`, `grpo`, `full_finetune`, `qat`, `doc2lora`, `i_dlm` *(gated)*  |
| Compressor  | `onecompression`, `gptq`, `awq`, `hqq`, `bnb`, `aqlm`                                 |
| Evaluator   | `lm_eval_harness`, `perplexity`                                                       |
| Exporter    | `gguf`, `vllm`, `mlx`, `i_dlm_isd` *(gated)*                                          |

Trainer plugins are named by *method*. The underlying backend (`unsloth`, `trl`, `torchtune`, `transformers`) is selected per recipe via `config.backend`, defaulting to whatever fits the detected hardware.

## Quickstart

```bash
uv sync --extra dev
uv run forgekit doctor
uv run forgekit list-plugins
uv run forgekit run recipes/qwen3_4b_qlora_gptq.yaml
```

`doctor` reports the detected CUDA arch (e.g. `gb10` / `rtx4090` / `rtx3090`), VRAM, and which optional extras are importable. Install extras as needed:

```bash
uv sync --extra qlora --extra gptq --extra lm-eval --extra gguf
```

## Recipe shape

```yaml
name: qwen3-4b-qlora-gptq
model: Qwen/Qwen3-4B
data:
  kind: hf_dataset
  config: { path: tatsu-lab/alpaca, split: train }
trainer:
  kind: qlora
  config: { backend: unsloth, lr: 2e-4, steps: 200 }
compressors:
  - kind: gptq
    config: { bits: 4, group_size: 128 }
evaluator:
  kind: perplexity
  config: { dataset: wikitext2 }
exporter:
  kind: gguf
  config: { quant: q4_k_m }
```

See `recipes/` for runnable examples and `docs/adding_a_plugin.md` for the incubator contract.

## License

Apache-2.0. Plugins must be OSI-approved or CC BY / CC BY-SA; PHOTON-style non-commercial licenses are excluded. I-DLM plugins are gated behind the `[i-dlm]` extra and `FORGEKIT_ACCEPT_I_DLM_LICENSE=1` until upstream clarifies.
