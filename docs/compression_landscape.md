# Compression landscape — August 2026

Why forgekit's backends changed, what the current accuracy-per-bit frontier
looks like, and which popular ideas were deliberately *not* adopted.

This is a decision record, not a survey. Every claim here either changed the
code in this repo or explains why something stayed the same.

---

## 1. Two backends died and forgekit did not notice

| Backend | Status | Replacement |
| --- | --- | --- |
| `auto-gptq` | Archived **2025-04-11**; removed from `transformers` | `gptqmodel` |
| `autoawq` | Officially deprecated and unmaintained | `llm-compressor` (vLLM project) |

forgekit's two flagship compressors imported both. Neither is a pin bump:

* `AutoGPTQForCausalLM.from_pretrained(path, quantize_config=…)` →
  `GPTQModel.load(path, quant_config)`; `save_quantized` → `save`.
* **Calibration data changed shape.** `auto-gptq` took pre-tokenized
  `{"input_ids", "attention_mask"}` dicts. `gptqmodel` takes a list of raw
  strings and tokenizes internally. `llm-compressor` takes the opposite — a
  *pre-tokenized* dataset via `oneshot()`. forgekit's old shared
  `_build_calibration_examples` matched neither.
* `llm-compressor` emits **compressed-tensors**, which vLLM loads natively.
  That is strictly better for this repo, which has a vLLM exporter.

The broader consolidation: `llm-compressor` is absorbing the fragmented
`AutoGPTQ` / `AutoAWQ` / `AutoFP8` world into one library, and now covers GPTQ,
AWQ, AutoRound, RTN, SmoothQuant, SparseGPT, QuIP and SpinQuant.

**Why nothing caught this:** every test in the suite mocked the backend away.
Twelve `feat: implement X` PRs merged on green CI without executing the code
they landed. See `tests/test_smoke_pipeline.py`.

### The pins were also two major versions stale

Independently of the dead backends, the extras that *were* populated pinned a
2024-era stack that excludes every current release:

| Pin | Current | Effect |
| --- | --- | --- |
| `transformers>=4.43,<5` | 5.x | excludes all of 5.x |
| `datasets>=2.19,<3` | 4.x–5.x | excludes everything current |
| `peft>=0.11,<0.13` | 0.20 | excluded |
| `accelerate>=0.30,<1` | 1.x | excluded |
| `trl>=0.9,<0.12` | 1.x | excluded |

That last one matters beyond resolution: `qlora.py` called
`SFTTrainer(tokenizer=…, max_seq_length=…)` with `TrainingArguments`. TRL 1.x
replaced `TrainingArguments` with `SFTConfig`, moved `max_seq_length` to
`SFTConfig.max_length`, and dropped the trainer-level `tokenizer=` and
`dataset_text_field=` arguments. The flagship trainer could not have run.

### Conflicts the resolver found

Running `uv lock` — as opposed to reading requirement files — surfaced a set of
hard incompatibilities. Each one below took an iteration to find:

* **`unsloth` caps `transformers<=5.5` and `datasets<4.4`**, while
  `llm-compressor` needs `transformers>=5.9` and `trl>=1.10` needs
  `datasets>=4.7`. There is no single version set.
* **`unsloth` caps `trl<=0.24`**, so the unsloth path is on TRL 0.x while
  everything else is on 1.x. `_sft_config` detects which field name to use
  rather than guessing.
* **`llm-compressor` 0.13.0 caps `transformers<=5.14.1`.**
* **`vllm` 0.27 pins `compressed-tensors==0.17.0`**; `llm-compressor` pins
  `==0.18.0`.
* **`vllm` pins `protobuf<7`** (transitively, via cutlass) while
  `gptqmodel>=7.3.1` requires `protobuf>=7.34`.

The shape of this is two isolated islands — `unsloth` and `vllm` — that cannot
share an environment with the mainstream quantization stack, or with each other.

Declaring them as conflicting extras under `[tool.uv] conflicts` is expressible,
and was tried: it needs ~25 explicit pairs (uv does not infer them through a
shared extra) and forks the resolution combinatorially — `uv lock` did not finish
in 10 minutes. Removing both extras instead makes the lock resolve in **0.6
seconds**. They are installed into their own environments; see the README.

Neither island costs much in practice:

* Pick one training backend per environment; `--extra trl` is the portable one.
* The `vllm` **exporter** writes a serving config and never imports `vllm`
  except for an optional serve smoke-test. Quantize in one environment, serve
  in another — which is how vLLM is normally deployed regardless.

There is also no `all` extra, for the same reason: no single environment holds
every backend, so an extra promising one would be a lie.

---

## 2. What actually buys accuracy per bit

Ranked by value to this repo.

### AutoRound — adopted, new `autoround` plugin

GPTQ and AWQ round with a fixed rule and correct the error afterwards.
AutoRound (Intel, Apache-2.0) *learns* the rounding decision and the clipping
range together via signed gradient descent over a few hundred steps.

* **W4**: roughly level with GPTQ/AWQ.
* **W3 / W2**: consistently ahead — reported up to ~2.1× relative accuracy over
  baselines at INT2, which is where fixed-rule methods collapse.
* **No inference overhead** — the output is an ordinary quantized checkpoint.

It also subsumes a lot of the zoo: it can emit `auto_round`, `llm_compressor`,
`auto_gptq`, `auto_awq`, or GGUF, supports `NVFP4`/`MXFP4` schemes, and applies
rotation internally. One plugin, every downstream format.

**Verdict:** the single best addition available. Use `gptq`/`awq` at 4 bits if
you have a working recipe; use `autoround` below 4 bits.

### Rotation (QuaRot / SpinQuant / ButterflyQuant) — adopted indirectly

Rotating weights and activations by an orthogonal transform destroys the
outlier channels that wreck low-bit quantization. SpinQuant (learned rotations)
beats QuaRot (random Hadamard) by up to 45% of the gap to full precision on
hard-to-quantize models; ButterflyQuant makes the learning cheap via Givens
parameterization.

**Verdict:** no separate plugin. AutoRound applies rotation internally, and
`llm-compressor` ships SpinQuant. A standalone rotation stage would duplicate
both.

### FP4 formats (NVFP4 / MXFP4) — exposed, deliberately not defaulted

NVFP4 (16-element blocks, FP8 scale) is more accurate than MXFP4 (32-element
blocks, power-of-two scale); MXFP4 reportedly needed ~36% more tokens to reach
the same pretraining loss. NVFP4 PTQ of DeepSeek-R1 stays within ~1% of FP8.
The throughput case is real *where the kernels are*: INT4 has to dequantize to
16-bit before the matmul, so NVFP4 has measured ~2.35× over INT4 on hardware
with FP4 tensor cores.

**But the GB10 story is genuinely unresolved.** A benchmark published
2026-04-21 (updated 05-06) found FP8 beating NVFP4 by ~32% on GB10 — 53.8 vs
40.8 tok/s on Qwen-3.6-35B-A3B — because sm_121 lacked the
`cvt.rn.satfinite.e2m1x2.f32` PTX instruction that sm_120 and sm_100 have,
forcing a Marlin fallback that decompresses FP4→BF16. Native SM120/121 CUTLASS
NVFP4 GEMM then merged around 2026-05-20, with further vLLM support landing in
June.

So the honest answer as of this writing is **version-dependent, and it changes
under you**.

**Verdict:** forgekit exposes `scheme: NVFP4` / `MXFP4` and refuses them on
hardware without FP4 tensor cores (`_check_scheme_supported`), with an
`allow_unsupported_scheme` escape hatch for cross-targeting. It does not make
FP4 a default, and it does not encode a guess about which format wins. Measure
on your box. Fewer bits is not automatically faster.

### Accuracy recovery after PTQ — not adopted yet

Recover-LoRA and similar train a small adapter on the *already quantized* model
to reclaim accuracy, with no retraining of the base. NVIDIA has a related
eigenspace low-rank approach that needs no retraining at all.

**Verdict:** worth doing, but it needs a pipeline shape forgekit does not have —
`compress → recover → eval`, i.e. a trainer stage running *after* a compressor.
The current `RecipeSpec` hardcodes one trainer before N compressors. Left as
follow-up; it is a schema change, not a plugin.

---

## 3. Evaluation is the real bottleneck — and the default was misleading

Every shipped recipe ended with `evaluator: {kind: perplexity}`. That is the
weakest possible check on the primary thing forgekit does.

* Aggressive PTQ **degrades reasoning while lengthening chains of thought**:
  across 28 model–quantization pairs, Spearman ρ ≈ **-0.73** between accuracy
  loss and CoT growth. In up to **52%** of failures the quantized model reaches
  the correct answer mid-trace and then abandons it.
* None of that moves perplexity much. Perplexity measures fluency.

So forgekit's default evaluator was structurally blind to the primary failure
mode of forgekit's primary feature.

### What changed

The perplexity evaluator now reports a **bootstrap confidence interval** beside
the point estimate:

* Resampling is over evaluation *windows*, token-weighted so a short trailing
  window does not count like a full one.
* Percentile bootstrap on mean NLL, exponentiated at the endpoints (`exp` is
  monotonic, so the transformed interval is valid for perplexity).
* Percentile, not CLT — normal-approximation intervals are unreliable below a
  few hundred datapoints, and window counts are routinely in the tens.

This matters more than it sounds. On a realistic window count the interval is
around **±7%**. Comparisons of quantization methods on perplexity gaps of 0.1
are, at that resolution, comparisons of noise.

The docstring and the shipped recipe now say plainly: use perplexity as a
regression signal, use `lm_eval_harness` for any capability claim.

Broader context: confidence intervals in LLM evaluation are systematically too
narrow, output variance can exceed data-sampling variance even at temperature
0, and NIST AI 800-3 (2026) now provides formal guidance on uncertainty
quantification for model comparison.

---

## 4. Deliberately not adopted

**LoRA variants (DoRA, PiSSA, LoRA-GA, MiLoRA, VeRA).** The 2026 evidence is
that once the learning rate is tuned, these land within noise of vanilla LoRA —
reported gaps of 0.52% (Gemma-3-1B, math) and 0.43%/1.75% (Llama-2-7B), with
several papers concluding vanilla LoRA at rank 16–32 is the right default.
DoRA closes maybe half the gap to full fine-tuning for 5–10% more VRAM.

Building six more trainer plugins would add surface area for an effect smaller
than the LR sensitivity that dominates it. **A learning-rate sweep would buy
more than all of them.**

**Speculative decoding (EAGLE-3).** Genuinely strong — 3–6× lossless, draft
head ~1–2% of base parameters, and it has a data-scaling law. But it is a
*serving-time* technique, and forgekit's pipeline ends at export. It belongs
behind the vLLM/SGLang exporter, not as a compressor. Follow-up.

**Pruning + distillation (Minitron).** Width pruning plus distillation reaches
strong accuracy-per-parameter with up to 40× fewer tokens than training from
scratch. But it needs a distillation *training* loop and a teacher model —
a new stage kind, not a compressor. Follow-up.

---

## 5. Open questions worth measuring, not guessing

1. **NVFP4 vs FP8 vs W4A16 on GB10**, at the current driver and vLLM version.
   The published answer has flipped once already.
2. **Does AutoRound's W3 advantage survive to task accuracy**, or only to
   perplexity? Section 3 predicts these can diverge — this is exactly the
   question perplexity cannot answer.
3. **Where does the accuracy cliff sit per model family?** AutoRound makes W2
   viable in principle; whether it is viable for a given 4B model is empirical.

All three are pipeline runs, not literature questions. That is the argument for
the smoke test and the confidence intervals: forgekit's value is measuring
these on the user's hardware, not encoding someone else's benchmark.

---

## Sources

Backends and tooling:
[AutoGPTQ deprecation](https://github.com/AutoGPTQ/AutoGPTQ/discussions/758) ·
[GPTQModel](https://github.com/ModelCloud/GPTQModel) ·
[AutoAWQ](https://github.com/casper-hansen/AutoAWQ) ·
[llm-compressor](https://github.com/vllm-project/llm-compressor) ·
[llm-compressor Q1 2026 roadmap](https://github.com/vllm-project/llm-compressor/issues/2262)

Quantization methods:
[AutoRound](https://github.com/intel/auto-round) ·
[SignRound / AutoRound paper](https://arxiv.org/abs/2309.05516) ·
[SpinQuant](https://arxiv.org/abs/2405.16406) ·
[ButterflyQuant](https://arxiv.org/pdf/2509.09679) ·
[GPTQ](https://arxiv.org/abs/2210.17323) ·
[AWQ](https://arxiv.org/abs/2306.00978)

FP4 formats and GB10:
[NVFP4 vs MXFP4 guide](https://www.spheron.network/blog/nvfp4-vs-mxfp4-gpu-cloud-4bit-quantization-guide/) ·
[NVFP4 across the model lifecycle](https://ai-infrastructure.net/nvfp4-quantization/) ·
[NVFP4 is a trap on GB10 (FP8 wins)](https://ai-muninn.com/en/blog/dgx-spark-nvfp4-trap-gb10-fp8-wins) ·
[vLLM on DGX Spark](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark) ·
[SM121 native NVFP4 support thread](https://forums.developer.nvidia.com/t/sm121-gb10-native-nvfp4-compute-seeking-guidance-on-software-support/364607)

Evaluation:
[Quantized reasoning models overthink](https://arxiv.org/abs/2606.00206) ·
[Quantization hurts reasoning?](https://arxiv.org/html/2504.04823v1) ·
[Adding error bars to evals](https://www.alphaxiv.org/overview/2411.00640) ·
[Don't use the CLT with few datapoints](https://arxiv.org/pdf/2503.01747) ·
[Hidden measurement error in LLM pipelines](https://arxiv.org/pdf/2604.11581)

PEFT and other techniques:
[Which LoRA? empirical study](https://arxiv.org/pdf/2606.10428) ·
[Learning rate matters: vanilla LoRA may suffice](https://arxiv.org/pdf/2602.04998) ·
[EAGLE-3 on vLLM](https://vllm.ai/blog/2026-07-13-eagle-3-amd-instinct) ·
[Minitron: pruning and distillation](https://arxiv.org/pdf/2408.11796) ·
[Recover-LoRA](https://arxiv.org/pdf/2606.04238)
