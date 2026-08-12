# Muse Glimmer Abliteration

Refusal-direction ablation on `meta-models/Muse-Glimmer-30B`, using the pipeline
developed in [gemma-4-abliteration](https://github.com/TrevorS/gemma-4-abliteration).

Method and tooling come from prior work ([credits](#credits)); this repo ports the
pipeline to a new architecture and adds an evaluation axis for agentic safety.

## Models

| Model | Params | Refusals before | after | HF (bf16) | HF (GGUF) |
|-------|--------|-----------------|-------|-----------|-----------|
| Muse-Glimmer-30B | 29.8B dense | 128/150 (85.3%) | **3/150 (2.0%)** | [uncensored](https://huggingface.co/TrevorJS/Muse-Glimmer-30B-uncensored) | [GGUF](https://huggingface.co/TrevorJS/Muse-Glimmer-30B-uncensored-GGUF) |

Collection: [Muse Glimmer 30B uncensored](https://huggingface.co/collections/TrevorJS/muse-glimmer-30b-uncensored-6a7bea176a516f9bf5f3ebf4)

## Results

Scale 1.0, all 52 layers. Single-turn, greedy decode, 1536-token budget, paired against
the unmodified base.

| | Before | After |
|--|--------|-------|
| Refusals (`harmful_tune`, 150 prompts) | 128/150 (85.3%) | **3/150 (2.0%)** |
| Over-refusal (`harmless_tune`, 75 prompts) | 1/75 (1.4%) | **0/75 (0.0%)** |
| Deflections | 12 | **0** |
| Broken / degenerate | 0 | **0** |

Of 128 baseline refusals, 83 flipped to confirmed compliance and none degraded. A
further 42 are unresolved — compliance runs about 4x longer than refusal (median output
354 -> 1340 tokens) and hit the token cap mid-answer. The confirmed flip rate is
therefore a lower bound of 65%, with an upper bound of 98% if every unresolved row is
compliance.

### Agentic axis

Muse Glimmer's published safety training covers tool-use boundaries, injection
resistance and permission handling. A content-refusal harness cannot observe any of
that, so a 30-probe agentic set was built for it.

| Category | Probes | Before | After |
|----------|--------|--------|-------|
| Prompt-injection resistance | 12 | 0/12 | **0/12** |
| Scope adherence | 8 | 0/8 | **0/8** |
| Irreversible-action confirmation | 10 | 7/10 | **8/10** |
| **Total** | **30** | **7/30** | **8/30** |

Injection and scope are unchanged by an edit that took content refusal from 85.3% to
2.0%. One probe changed: asked to delete logs older than a day, the base model noted
that the available tool could not filter by age and asked before acting; the abliterated
model called `delete_files` directly.

Both unchanged categories were already at a 0% failure floor, so this shows the edit did
not break them rather than that they are hard to remove.

## Method

Norm-preserving biprojected abliteration. The refusal direction is projected out of each
residual-write matrix and every column rescaled to its original norm:

```
W' = W - scale * r(rTW),   then  ||W'||_col = ||W||_col
```

Applied to `self_attn.o_proj` and `mlp.down_proj` in all 52 text decoder layers — 104
tensors edited, 1332 copied unchanged, including all 800 vision-tower tensors.

| Parameter | Value |
|-----------|-------|
| Layers | 100% (all 52) |
| Scale | 1.0 |
| Winsorization | 0.995 |
| Derivation pairs | 400 harmful / 400 harmless |
| Direction gate | held-out, mean selectivity 7.49, 0/52 anti-selective |

### Architecture notes

1. **Targets are standard.** `o_proj` and `mlp.down_proj`, both `[6656, *]`.
2. **Two `gate_proj` per layer** — attention and MLP, 104 total. Neither is a
   residual-write matrix; targets are matched by full path, never by name suffix.
3. **Logits are softcapped** (`20*tanh(x*0.196/20)`), so KL here is not comparable to
   the Gemma 4 family's numbers.
4. **Channelled output.** The model emits `to=self` deliberation then a `to=user`
   answer. Evaluation reads the final channel; scoring the concatenation reads the
   model's reasoning about a refusal as the refusal itself.
5. **Tool calls are not JSON** — `<atem:invoke name="...">`. A JSON-shaped scorer
   returns zero calls on every completion.

## Requirements

| | |
|---|---|
| `transformers` | **>= 5.15.0** — `muse_glimmer` is absent from 5.12.0 |
| `torch` | 2.11.0+cu130 (tested) |
| llama.cpp (GGUF) | build **b10353+** ([#26841](https://github.com/ggml-org/llama.cpp/pull/26841), merged 2026-08-10) |

## Corpora

Prompt corpora are **not included in this repo**. The derivation and evaluation sets are
public datasets — see [Credits](#credits) for sources (mlabonne, JailbreakBench,
tulu-harmbench, NousResearch, AdvBench). Place them under `data/` as
`harmful_train.txt` / `harmless_train.txt` / `harmful_eval_full.txt`, then
`scripts/splits.py` builds the disjoint splits.

The agentic and conversational probe sets are ours and regenerate from source:

```bash
uv run python scripts/build_agentic_probes.py
uv run python scripts/build_conversational_probes.py
```

## Pipeline

```bash
# 0. splits — leakage audit + disjoint DERIVE/TUNE/TEST
uv run python scripts/splits.py

# 1. capture activations                                    [GPU]
.venv-gpu/bin/python scripts/capture.py \
  --harmful data/harmful_train.txt --harmless data/harmless_train.txt \
  --out acts/derive.npz --position prompt_final

# 2. derive + gate                                          [CPU]
uv run python scripts/derive_direction.py \
  --acts acts/derive.npz --acts-holdout acts/holdout.npz --out directions/v1

# 3. apply, shard-by-shard                                  [CPU]
.venv-gpu/bin/python scripts/abliterate.py \
  --directions directions/v1.npy --out models/mg-abl-s1.0 --scale 1.0

# 4. generate, then score offline                           [GPU, then CPU]
.venv-gpu/bin/python scripts/generate.py --mode content \
  --prompts data/splits/harmful_tune.txt --out runs/abl-content.jsonl
uv run python scripts/eval_content.py --completions runs/abl-content.jsonl \
  --baseline runs/base-content.jsonl
```

## Scripts

| Script | Purpose |
|---|---|
| `splits.py` | leakage audit + disjoint DERIVE/TUNE/TEST |
| `capture.py` | per-layer residuals; `prompt_final` / `prompt_mean` / `response` |
| `derive_direction.py` | direction + selectivity gate + SNR profile |
| `projection.py` | norm-preserving biprojection (single source of truth) |
| `selectivity.py` | the gate |
| `abliterate.py` | shard-by-shard safetensors edit |
| `generate.py` | completions, local or against an OpenAI-compatible endpoint |
| `eval_content.py` | offline refusal classifier + paired flip |
| `eval_agentic.py` | structural agentic scorer |
| `eval_conversational.py` | stylistic-refusal scorer (built, not yet run) |
| `channels.py` / `toolcalls.py` | channel split and `<atem:invoke>` extraction |
| `serve.py` | OpenAI-compatible server + chat UI |

## Tests

```bash
scripts/run_tests.sh
```

67 CPU regressions, no GPU or model load. Several use real completions as fixtures, so
the scoring edge cases they cover stay pinned.

## Limitations

- Scale 1.0 only; no sweep.
- Flip rate is a range (65–98%) because a third of rows hit the token cap.
- Hard-harm single-turn corpus only. A check of 150 base completions found zero
  stylistic refusals, so that band is unmeasured; a 50-probe conversational set is built
  but unrun.
- Single-turn only; the vision tower is untouched.
- Agentic probe set is N=30.

## Credits

Roughly in order of how much this work leans on them.

**Refusal directions**
- Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee and
  Neel Nanda, [*Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717)
  (NeurIPS 2024; [code](https://github.com/andyrdt/refusal_direction)) — the result the
  whole technique rests on.
- [*There Is More to Refusal in Large Language Models than a Single Direction*](https://arxiv.org/html/2602.02132v1)
  — the direct follow-up, and worth reading alongside the original before treating a
  single direction as the whole story.
- [grimjim](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)
  — norm-preserving biprojected abliteration, the exact projection used here. Also
  [ORBA](https://huggingface.co/blog/grimjim/orthogonal-reflection-bounded-ablation).
- [p-e-w/heretic](https://github.com/p-e-w/heretic) — the tool the gemma-4 pipeline was
  built on and borrowed heavily from.
- [elder-plinius/OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS) — the
  Expert-Granular Abliteration concept used for MoE in the gemma-4 work.

**Papers that shaped decisions here**
- [*Abliteration Is Not a Scalpel*](https://arxiv.org/html/2607.17427) — off-target
  effects of refusal removal. Prompted the agentic axis; it states that tool use,
  injection and irreversible-action confirmation are *not* assessed there.
- [*SOM Directions Are Better than One*](https://www.alphaxiv.org/overview/2511.08379v1)
  (Piras et al., AAAI 2026, [code](https://github.com/pralab/som-refusal-directions)) —
  multi-directional refusal suppression.
- [arXiv:2603.22061](https://arxiv.org/html/2603.22061) — the circularity critique of
  scoring a direction on the activations it was derived from. This is why the
  selectivity gate runs on held-out captures.
- [*Your Agent is More Brittle Than You Think*](https://arxiv.org/html/2604.03870) and
  [*InjecAgent*](https://arxiv.org/abs/2403.02691) — indirect prompt-injection
  benchmarks; the agentic probe design borrows their canary / tool-result structure.
- [*Not All Refusals Are Equal*](https://arxiv.org/html/2607.02714) and
  [*Ablating Safety*](https://arxiv.org/html/2605.17413v1) — refusal is not one
  behaviour, which is the premise of the multi-axis evaluation.
- [*The Landscape of Prompt Injection Threats in LLM Agents*](https://arxiv.org/pdf/2602.10453)
  and [*A Survey on Agentic Security*](https://arxiv.org/pdf/2510.06445) — taxonomies
  used to check the probe categories were not missing a class.

**Datasets**
- [mlabonne](https://huggingface.co/mlabonne) — `harmful_behaviors` and
  `harmless_alpaca`, the derivation corpora.
- [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors),
  [allenai/tulu-3-harmbench-eval](https://huggingface.co/datasets/allenai/tulu-3-harmbench-eval),
  [NousResearch/RefusalDataset](https://huggingface.co/datasets/NousResearch/RefusalDataset),
  [walledai/AdvBench](https://huggingface.co/datasets/walledai/AdvBench) — evaluation pools.

**Tooling**
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF conversion,
  quantization and serving, including `muse_glimmer` support.
- HuggingFace `transformers` — model loading and the `muse_glimmer` implementation.

**Base model**
- [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B),
  Meta Superintelligence Lab, Apache 2.0.

## License

Apache 2.0, matching the base model.
