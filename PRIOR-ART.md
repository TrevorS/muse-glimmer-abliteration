# Prior art — surveyed 2026-08-10

What exists, what it settles, and what it leaves open for this repo. Read before
proposing a method; several things on the `IDEAS.md` backlog are already answered, and
one item that `ds4-refusal` marked closed is **not** actually closed.

## The headline: nobody has measured what we're about to measure

**Muse Glimmer has not been abliterated.** Searched HF for abliterated/uncensored
variants — only the official repos (`meta-models/Muse-Glimmer-30B`, `-GGUF`,
`-assistant`, `-ExecuTorch-PTE`) and `unsloth/Muse-Glimmer-30B{,-GGUF}` quants exist.
First mover on this target.

**And the agentic axis is explicitly unexamined by the closest paper.**
[*Abliteration Is Not a Scalpel: Off-Target Effects of Refusal Removal on Decision
Disposition Across Model Families*](https://arxiv.org/html/2607.17427) is the strongest
evidence that abliteration has off-target behavioral effects — and it states outright
that it does **not** assess tool use, prompt injection, or irreversible-action
confirmation. That is our gap, named by the paper closest to it.

Its result also raises the stakes. On 21,600 stock-direction calls (60 equities × 18
weeks × 5 samples), abliterated variants were systematically more optimistic:
**+12.2 pp** more up-bets on Gemma-4-26B-A4B-it, **+7.4 pp** on Qwen3-30B-A3B, plus
longer self-justification (+4.0 / +7.4 words) and fewer uncertainty markers.
Confidence moved in *opposite* directions across the two families (−0.008 Gemma,
+0.109 Qwen, non-overlapping CIs) — so the off-target effect is real but not uniform,
and cannot be predicted from one family.

Note it tested **Gemma-4-26B-A4B-it**, the same model `../gemma-4-abliteration`
shipped. Its stated gaps — "no mechanistic grounding yet", single abliteration method,
deferred activation-space analysis — are places our activation-space tooling could
speak directly.

## Correction to a closed item

`ds4-refusal/FINDINGS-20260807.md` §9 closes **rank-k subspace** on the grounds that
SVD of the harmful-minus-benign departures gives σ1 = 69–83% of variance with
cos(σ1, meandiff) = 1.000, so refusal is "~78% rank-1".

That argument does not close
[**SOM Directions Are Better than One**](https://www.alphaxiv.org/overview/2511.08379v1)
(Piras et al., AAAI 2026; [code](https://github.com/pralab/som-refusal-directions)).
The method is not a subspace decomposition:

- Train a 4×4 hexagonal SOM (16 neurons) on harmful-prompt activations at one layer.
- Each neuron gives a direction `r_ι = w_ι − ν`, ν = the harmless centroid.
- Bayesian Optimization picks `k ∈ [2,7]` of them; ablate uniformly across all layers.

These are **centroids of a clustered manifold, mutually correlated** — the paper
explicitly contrasts this with orthogonal decomposition. An SVD showing one dominant
*orthogonal* component says nothing about whether the harmful set is multi-modal in the
clustering sense. The two analyses are compatible and measure different things.

Results: **59.11% ASR (k=7) vs 0.00% for the single-direction baseline on Llama2-7B**,
and 25.79% on Mistral-7B-RR where competitors largely fail; 8 models tested. Stated
limits: BO search is expensive, layer strategy is undeveloped (uniform ablation), and a
single harmless centroid may miss benign-side structure.

**Implication.** The ~12–22% stubborn residual in `ds4-refusal` was attributed to a
*distributional* limit ("our corpus is missing refusal modes") after rank-k was ruled
out. SOM says the missing modes may be recoverable as *cluster* structure the
diff-of-means average washes out — which is the same diagnosis with a method attached.
Worth trying here, and worth revisiting on ds4.

## Direction-extraction methods

| Work | Idea | Bearing on us |
|---|---|---|
| Arditi et al. 2024 | diff-of-means single direction | the baseline; what `derive_direction.py` implements |
| [Heretic](https://github.com/p-e-w/heretic) | Optuna over ablation params | 3/100 refusals @ KL 0.16 on Gemma-3-12B vs 0.45–1.04 manual |
| [SOM directions](https://www.alphaxiv.org/overview/2511.08379v1) | multi-direction from clustered manifold | see above — the live lead |
| [ORBA](https://huggingface.co/blog/grimjim/orthogonal-reflection-bounded-ablation) | orthogonal reflection, bounded | grimjim; same lineage as our biprojection |
| [arXiv:2603.22061](https://arxiv.org/html/2603.22061) | topic-matched contrast baselines fail | already cited in FINDINGS §2 for the circularity critique |
| DeepRefusal | probabilistic refusal ablation | not yet read |

## Off-target / capability effects

- [**Abliteration Is Not a Scalpel**](https://arxiv.org/html/2607.17427) — decision
  disposition shifts far from any refusal. See above.
- [**Willing but Unable: Separating Refusal from Capability in Code LLMs via
  Abliteration**](https://arxiv.org/pdf/2606.05396) — uses abliteration as an
  *instrument* to separate "won't" from "can't". Directly relevant to reading our
  `broken` class: a post-ablation failure may be incapacity, not refusal.
- [**Not All Refusals Are Equal**](https://arxiv.org/html/2607.02714) — safety
  alignment fails unevenly across cybersecurity tasks; refusal is not one behavior.
- [**Ablating Safety**](https://arxiv.org/html/2605.17413v1) — mechanisms for removing
  alignment for security applications.

Common thread, and it matches our two-axis design: **refusal is not a single behavior,
and removing it does not affect all behaviors equally.**

## Agentic safety / injection — for probe design

Our 30-probe pilot is hand-written. Established benchmarks to borrow structure from
before scaling it up:

- [**InjecAgent**](https://arxiv.org/abs/2403.02691) — 1,054 test cases, 17 user tools,
  62 attacker tools. The obvious source for expanding beyond N=30.
- [**Your Agent is More Brittle Than You Think**](https://arxiv.org/html/2604.03870) —
  indirect injection in agentic LLMs; expanded action spaces, privilege exposure.
- [**The Landscape of Prompt Injection Threats in LLM Agents**](https://arxiv.org/pdf/2602.10453)
  — taxonomy, useful for making sure our three categories aren't missing a class.
- [**A Survey on Agentic Security**](https://arxiv.org/pdf/2510.06445) — threats and defenses.

None of these test an **abliterated** model. That intersection is empty.

## Tooling facts established

- **llama.cpp supports this arch.** Merged 10 Aug 2026 (#26841, `62bf73d`), first in
  release **`b10353`**; `b10344` and older refuse to load. Present in the local
  checkout at `src/llama-model.cpp` (`LLM_ARCH_MUSE_GLIMMER`).
- **Official GGUFs exist** — `meta-models/Muse-Glimmer-30B-GGUF` ships
  `muse-glimmer-30B-kquant-{17gb,dynamic}.gguf`, plus `mmproj-kquant.gguf` (perception
  encoder, required for image input, `--mmproj`) and `dflash-kquant.gguf` (speculative
  decoding, optional, `-md`). unsloth ships BF16 / Q8_0 / UD-IQ2_XXS…IQ3_M.
- **Consequence: the `ds4-refusal` runtime-steering path is available here.** Control
  vectors on a GGUF, no 60 GB weight bake, no requantization. `derive_direction.py`
  already knows how to emit a llama.cpp control-vector GGUF. This is a cheaper first
  experiment than the weight edit and should probably come first.

## Not found in this search

- Multi-turn evaluation of directional **ablation** (additive persona steering has it —
  arXiv:2605.10664 — ablation does not). Still open, as `FINDINGS` §4 concluded.
- Position-resolved selectivity as a diagnostic. Still open, as `FINDINGS` §5 concluded.
- Any measurement of abliteration's effect on **injection resistance, tool-use
  boundaries, or irreversible-action confirmation.** Empty intersection — this repo's
  main question.
- Any measurement of whether text-derived directions transfer to the **vision** pathway
  of a multimodal model.
