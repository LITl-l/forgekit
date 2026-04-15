# forgekit — proposal

## Why

As of April 2026 there is still no OSS tool that unifies the full local-LLM
development loop — LoRA/QLoRA training + the modern compression zoo (GPTQ, AWQ,
HQQ, bitsandbytes, AQLM, OneCompression) + evaluation + GGUF/vLLM/MLX export —
on consumer GPUs (8–24 GB) and on NVIDIA GB10 (Grace Blackwell, 128 GB unified
memory). Unsloth Studio is closest but skips the compression step. Axolotl is
trainer-focused. `llm-compressor` is compressor-focused. None are research
incubators.

## What

A plugin pipeline driven by a single YAML `Recipe`. Five stage kinds —
`data`, `trainer`, `compressor`, `evaluator`, `exporter` — each defined as a
`typing.Protocol`. Plugins register via `importlib.metadata` entry points and
are discovered at runtime; there is no central registry file to edit when
adding a plugin.

Trainer plugins are named by *method* (`qlora`, `sft`, `dpo`, `grpo`,
`full_finetune`, `qat`, `doc2lora`, `i_dlm`). The underlying backend
(`unsloth` / `trl` / `torchtune` / `transformers`) is a `config.backend` field
with a hardware-aware default.

## Differentiator

Bleeding-edge techniques ship as plugins within ~a week of upstream code
release. The first three waves:

1. QLoRA + GPTQ + GGUF (baseline parity).
2. **OneCompression** (mixed-precision PTQ + AutoBit, Fujitsu, MIT).
3. **Doc-to-LoRA** (Sakana AI hypernetwork → adapter from plain docs).

Further candidates tracked in the plan: TriAttention (Apache-2.0) at serve
time, I-DLM gated-LoRA conversion + Introspective Strided Decoding (gated on
upstream license clarification), TurboQuant (watching for release). PHOTON is
excluded — CC BY-NC-ND is incompatible with an OSS plugin surface.

## Scope of the scaffold PR

Core `forgekit` package (`recipe`, `stages/*`, `registry`, `hw/*`, `cli`),
plugin stubs that raise `NotImplementedError`, three example recipes, tests
for schema / registry / hw-detect, CPU-only CI, dev-shell flake, docs. Zero ML
code. Follow-up PRs land implementations one plugin at a time.

See `docs/adding_a_plugin.md` for the incubator contract.
