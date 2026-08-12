# Research State

## Current model under test
`meta-models/Muse-Glimmer-30B` — 29.6B dense multimodal, 52 text layers, hidden 6656.
Downloaded and verified byte-exact (13/13 files, 59.58 GB) at `models/Muse-Glimmer-30B`.

## Status: captures done, direction v1 derived and gated, baseline re-running

Done: derive capture (400/400) and held-out capture (150/75), both at `prompt_final`,
all finite. Direction `v1` derived and **passing the gate on held-out activations**.

Not done: no baseline of record (v1 retracted, see below), no abliteration, no sweep.
Cells marked *pending* are empty on purpose — a placeholder gets cited later as data.

## Open question driving the target choice

Muse Glimmer's advertised safety training is mostly **not** content refusal: tool-use
boundaries, prompt-injection resistance, permission handling. The inherited 686-prompt
harness cannot observe any of it.

> **Pre-registered question.** Does a rank-1 refusal-direction ablation that drives
> content refusal toward zero also degrade agentic safety behaviors?

> **Pre-registered prediction.** Injection resistance degrades *less* than content
> refusal at matched scale. Reasoning: injection resistance is a discrimination between
> instruction sources present throughout the context, not a decision made at the
> prompt-final token where the direction is derived. If it degrades *equally*, that is
> evidence the direction is closer to a general "comply with what is in front of me"
> axis than to a refusal axis specifically — which would be the more interesting result
> and would also reframe every prior flip-rate number in this program.

Falsifiable either way on `data/agentic/probes.jsonl` at the same scales as the content
sweep. Record the answer here before running anything downstream of it.

## Baseline — v1 RETRACTED, re-run in flight

**Do not cite baseline v1.** It reported agentic injection 12/12 failed; every one was
a false positive from scoring the `to=self` reasoning channel instead of `to=user`.
Full post-mortem in `RESEARCH-JOURNAL.md` (2026-08-11). Also affected: probe inj-12
(quoting is the correct answer there) and a 512-token cap that left 14% of rows with
no final answer at all.

| axis | set | n | metric | value |
|---|---|---|---|---|
| content | `harmful_tune` | 150 | refuse rate | pending |
| content | `harmless_tune` | 75 | over-refusal | pending |
| agentic | injection | 12 | fail rate | pending (provisional 0/11) |
| agentic | irreversible | 10 | fail rate | pending (provisional 6/8) |
| agentic | scope | 8 | fail rate | pending (provisional 0/7) |

Provisionals are the corrected scorer applied to the flawed 512-token run. They match
the pre-registered ordering — injection resistance holds, irreversible-action
confirmation does not — but they are not the baseline of record.

## Directions

| tag | position | pairs | gate | adj_cos | notes |
|---|---|---|---|---|---|
| `v1` | prompt_final | 400/400 | **PASS** derivation 9.36 · **held-out 7.49**, 0/52 anti-selective | 0.834 | SNR peaks L39-48; L49-51 are a magnitude trap |

`v1` geometry: the direction rotates smoothly through depth rather than being one
global axis — adjacent-layer cosine 0.75-0.92 (rising with depth), but SVD of the
[52, 6656] stack gives σ1 = only 26.4% of variance (σ2 16.6%, σ3 11.3%). SNR at L0 is
0.061, i.e. early layers carry almost no refusal signal.

## What is already established (from the two prior programs)

Carried over as constraints, not re-derived:

- **The projection is correct as written.** `W' = W - scale·r(rᵀW)` with per-**column**
  norm restoration. Row-wise restoration plus an unscaled second pass made `scale == 0`
  a near-full ablation and the scale knob inert (`NEGATIVE-RESULTS.md`). Pinned by 6
  CPU regressions.
- **Anti-selective layers are disqualifying.** Every direction that ever worked had 0;
  every broken one had 11–20 (`FINDINGS-20260807.md` §2). Binary, no threshold.
- **Selectivity is a regression check, not a ranker.** On derivation activations it
  largely restates "was Gram-Schmidt applied" — deriving a direction from activations
  and scoring it on those same activations is circular (arXiv:2603.22061). Held-out
  captures are required for any forward use.
- **Rank by SNR, never raw separation.** Raw separation selects magnitude-trap layers.
  Confirmed here: raw sep ranks L51,50,49... monotone in residual norm (L0 = 5.5,
  L51 = 295.6) while SNR ranks L45,41,42,44,43. (The top-10 sets overlap 7/10 on this
  model, not ~1/8 as on ds4 — but the disagreement is at the ends and that is what
  bites: sep would take L49-51 over the better L39-41.)
- **`adj_cos` is descriptive, not diagnostic.** Measured 0.834 here *with* a clean
  held-out gate. ds4's `refusal_iq3` was bad because of 20/42 anti-selective layers;
  its 0.952 collinearity was a co-symptom, not the defect. Judge by the gate.
- **Score the final channel, never the reasoning.** This model deliberates on `to=self`
  before answering on `to=user`. Scoring the concatenation produced a 100% false-
  positive rate on injection probes. See `channels.py`.
- **A truncated row is not a pass.** No final channel and no tool call = `unscoreable`,
  reported separately. Folding it into either bucket invents data.
- **`broken != refused`.** An over-ablated model stops emitting refusal markers because
  its output falls apart. The ds4 shipped config "unlocked" 8.8% of refusals while
  breaking ~10% of harmful and ~5% of harmless outputs.
- **Refusals converting to `deflect` is over-ablation, not progress** (`FINDINGS` §3).
- **s > 1 is a different intervention, not overdriving.** The ds4 optimum was 3.0, not
  ≤1; INLP reports the same from the other side (nullspace projection underperforms
  counterfactual flipping). Do not assume the Gemma-era `scale=1.0` transfers.
- **Free-running generation noise floor ≈ 7–10pt.** Adjacent operating points are not
  resolvable by generation alone. Teacher-forced paired KL, by contrast, was measured
  at *exactly* zero noise — any nonzero value there is signal.

## Known-closed avenues (`FINDINGS-20260807.md` §9 — do not re-run)

Expert/router targeting · early-layer zeroing · corpus diversity at equal N ·
substring refusal markers (11/12 false positives in the gemma-4 audit).

**Re-opened: rank-k / multi-direction.** The prior closure rested on SVD showing
σ1 = 69–83% with cos(σ1, meandiff) = 1.000. That rules out an *orthogonal subspace*.
It does not rule out the SOM method (Piras et al., AAAI 2026), which extracts mutually
**correlated centroids of a clustered manifold** — a different object, and one an
orthogonal decomposition is blind to. Reported 59.11% ASR at k=7 vs 0.00% for
single-direction on Llama2-7B. See `PRIOR-ART.md`; now `IDEAS.md` 3b.

This also re-frames ds4's stubborn 12–22% residual: it was attributed to a
*distributional* limit after rank-k was ruled out, and SOM is that same diagnosis with
a method attached.

## Toolchain — verified 2026-08-10

- CPU: `uv run python` (pyproject) — transformers 5.15.0, numpy, jinja2.
- GPU: `.venv-gpu/bin/python` — + torch 2.11.0+cu130. `cuda.is_available()` True on
  **NVIDIA GB10** (aarch64). Meta-device instantiation reproduces all **1436**
  parameter tensors; `model.model.language_model.layers` -> 52, hook register/remove
  OK. So `capture.py`'s module path is confirmed against the real class, not guessed.
- Chat template verified: all 30 agentic probes render with `tools=`.
- llama.cpp `muse_glimmer` merged 10 Aug 2026 (#26841, `62bf73d`, release `b10353`) and
  is present in the local checkout; official GGUFs exist. Runtime control-vector
  steering is therefore available — cheaper than the weight bake, `IDEAS.md` 2b.
- The heretic env was deliberately **not** upgraded (5.12.0 has no `muse_glimmer`, and
  `gemma-4-abliteration` is validated there).

## Architecture findings (2026-08-10)

- Residual-write targets confirmed: 104 tensors (52 `o_proj` + 52 `mlp.down_proj`),
  both `[6656, *]`. Applier dry-run matches exactly; 1332 copied, 1436 total.
- **Two `gate_proj` per layer** (attention + MLP, 104 total). Neither is a residual
  write. The attention gate multiplies *before* `o_proj`, in its input space, so
  `o_proj` remains the correct target.
- **Post-norm leak.** The residual write is `post_attention_layernorm(o_proj(x))`. The
  norm's element-wise gain `1 + w` (measured range **0.199-7.750**, sd 0.697) breaks
  the orthogonality the projection establishes. `--norm-aware` folds it in.
  Leak for a *random* direction is only ~1.5e-3/layer, so this is **not yet a result** —
  measure it for the derived direction first (`IDEAS.md` 9).
- Logit softcapping `20*tanh(x*0.196/20)` rules out the 12B `-inf` NaN-KL trap, and
  rules out comparing KL to the Gemma 4 family.
- Tool calls use `<atem:invoke name=...>`, not JSON. A JSON-only scorer would have
  reported a maximally-unsafe model as 0/30.

## Open limitations, stated up front

- **KL is not comparable to the Gemma 4 family.** Logit softcapping
  (`20·tanh(x·0.196/20)`) compresses the distribution before the divergence is taken.
- **The agentic probe set is N=30, a pilot.** Against a ±7–10pt noise floor it resolves
  roughly a 15–20pt difference and no better. Report paired per-probe flips, not rates.
- **Text-only.** A direction derived from text prompts says nothing about the vision
  pathway; image-conditioned refusal is unmeasured and untouched.
- **Single-turn.** The multi-turn gap from `FINDINGS` §4 is inherited wholesale and is
  worse here, because this is an agentic model whose realistic workload is multi-turn.

## Publishing constraint

`USAGE_POLICY.md` §1.8 in the model repo prohibits actions to "circumvent or remove ...
safety measures". The Apache 2.0 grant does not incorporate it by reference. Unresolved;
bears on upload/redistribution, not on local measurement. Decide before any push.
