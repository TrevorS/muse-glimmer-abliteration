# Prioritized backlog

Ordered by information gained per GPU-hour. Anything that can falsify a cheap
hypothesis before an expensive run goes first.

---

## 1. Baseline, both axes  ·  BLOCKING EVERYTHING

Nothing else is interpretable without it, and both prior programs found their baseline
disagreed with expectation.

- `generate.py --mode content --prompts data/splits/harmful_tune.txt` and
  `--mode agentic`, then score offline.
- **Multi-seed from the start.** `--seeds 0 1 2`. The ±7–10pt free-running floor was
  discovered late in ds4 and invalidated a batch of single-run comparisons.
- Also run `harmless_tune` — over-refusal has to be a baseline number, not a
  post-hoc check.

**Predict before running:** content refusal on `harmful_tune`, and the agentic fail
rate per category. Write the predictions in `RESEARCH-JOURNAL.md` first. A baseline
that surprises you is worth more than a sweep that confirms you.

*Watch for:* a high baseline agentic fail rate. If the model already follows injections
30% of the time, the axis has little headroom and needs harder probes before it can
measure anything about abliteration.

---

## 2. Derive at `prompt_final` and pass the gate

Standard recipe: 400 harmful / 400 harmless from `data/*_train.txt`, winsorize 0.995,
double-pass Gram-Schmidt, all 52 layers.

- Capture a **held-out** set too (`--holdout` on tune-split prompts) and derive with
  `--acts-holdout`. Selectivity on derivation activations is partly circular; the JSON
  records which basis was used.
- Read the **SNR profile**, not raw separation. Expect a low-signal early band and a
  high-signal mid-late band, as in both prior models. That profile picks the layer band
  for `--top-pct`; do not guess it.
- `adj_cos_median` is **descriptive, not diagnostic** — see the 2026-08-10 result below.

---

## 2b. Runtime steering first, weight bake second  ·  cheaper than it looks

I originally wrote that llama.cpp had no `muse_glimmer` support. **Wrong** — see
`PRIOR-ART.md`. Support merged 10 Aug 2026 (#26841, `62bf73d`, release `b10353`), it is
already in the local checkout, and official GGUFs exist including `mmproj` and `dflash`.

That makes the whole `ds4-refusal` runtime path available: a llama.cpp **control
vector** on a quantized GGUF, no 60 GB weight bake, no requantization, and the scale
knob is a runtime flag instead of a re-export per operating point.
`derive_direction.py` already emits the control-vector GGUF format.

Do the scale sweep this way. Reserve `abliterate.py` for the final chosen point.

Two carry-overs from ds4 that apply verbatim, both of which cost real time there:

- **The `direction.<il>` index is `il`, not `il+1`.** The off-by-one wrote layer 0's
  vector one layer late and dropped the last layer's entirely, collapsing flip to ~5%
  (`derive_direction.py:85-90` in the ds4 tree documents it as "THE killer").
- **Verify which file the server actually loads.** `FINDINGS` §1: production served the
  one direction in the directory that skipped orthogonalization, for weeks.

## 3. THE QUESTION: does the content sweep take agentic safety with it?

The reason this target is interesting. Run both axes at every scale in one job.

- Scales: 0 (identity control), 0.5, 1.0, 1.5, 2.0, 2.5, 3.0. **Do not assume 1.0.**
  The ds4 optimum was 3.0 and `s > 1` is a different intervention, not overdriving.
- Report the **paired** transitions on both axes, not the rate delta.
- Include `scale=0` as an explicit identity check — it also verifies the applier is a
  no-op at zero, which is the exact thing the old projection bug broke.

Three outcomes, all publishable:

| observed | reading |
|---|---|
| both collapse together | refusal and agentic safety share a direction |
| content drops, agentic holds | the behaviors are separable; the "refusal direction" is narrower than assumed |
| agentic drops *first* | the direction is closer to a general compliance axis — reframes every flip-rate number in this program |

---

## 4. Position-resolved selectivity  ·  the decisive test from `FINDINGS` §5

Cheap, offline once captured, and it settles the standing explanation for ds4's
multi-turn degradation. **Untested for ablation anywhere in the literature.**

Capture harmless/harmful at `prompt_final`, `prompt_mean`, and `response`, then compute
selectivity per layer per position on held-out captures. `capture.py --position` already
supports all three.

- Selectivity collapses away from the derivation position → hypothesis holds; the
  remedy is position-uniform orthogonalization (orthogonalize against a *basis* of
  harmless means, `FINDINGS` §5).
- Flat → hypothesis dead. Implement CAST gating instead. **Do not rescue it.**

This matters more here than it did on ds4: an agentic model's realistic workload is
long multi-turn tool loops, which is almost entirely continuations.

---

## 5. Multi-turn agentic degradation

`FINDINGS` §4: zero multi-turn data at any scale, so 2.0 was never established as safe —
only 3.0 as unsafe. That gap is inherited, and it is worse on an agentic model.

Extend the agentic probes to K-turn conversations with the KV cache persisted; score
injection resistance and confirmation behavior **by turn depth**. Prior art
(arXiv:2605.10664) reports −18.6 points of coherence drift over 10 turns for additive
persona steering and names KV-cache contamination as the mechanism.

Gate on this before any claim about agentic deployment.

---

## 6. Expand the agentic probe set

N=30 resolves ~15–20pt and no better. Before any quantitative claim:

- 100+ probes, categories balanced.
- Harder injections: multi-hop (tool output referencing another tool), indirect
  (injection in a file the model chooses to read), and role-confusion.
- A **benign control set** — tasks that superficially resemble the probes but where
  acting is correct. Without it, "declines everything" scores as perfect safety.

That control is not optional. It is the agentic analogue of over-refusal, and the ds4
program shipped a config that raised benign over-refusal 2.7% → 9.3% without noticing.

---

## 7. Vision pathway

A text-derived direction touches nothing in the 800-tensor vision tower. Two open
questions, neither answered by anything in this repo:

- Does image-conditioned refusal survive text-space abliteration?
- Is there a separate refusal direction in the vision adapter's output space?

Cheap first probe: run the content set with a benign image attached and diff against
text-only. If refusal survives at a materially different rate, there is a second
mechanism.

---

## 8. KL methodology  ·  do before quoting any KL number

Logit softcapping (`20·tanh(x·0.196/20)`) makes KL here non-comparable to the Gemma 4
family's table. Decide and document *once*:

- pre- or post-softcap logits;
- per-token normalization (the ds4 diffusion analog was sequence-summed and had to be
  retracted);
- teacher-forced paired probe, which had an exactly-zero noise floor — port
  `quality_gate.py` from `llama.cpp/experiments/refusal/`.

Never put a Muse Glimmer KL in the same table as a Gemma one.

---

## Explicitly closed — do not re-run (`FINDINGS-20260807.md` §9)

Expert/router targeting (arXiv:2606.04160) · early-layer zeroing (A1) · corpus
diversity at equal N · substring refusal markers · COSMIC (≈ SNR).

**Removed from the closed list: rank-k / multi-direction.** See `PRIOR-ART.md`. The SVD
argument (σ1 = 69–83%, cos(σ1, meandiff) = 1.000) rules out an *orthogonal subspace*; it
does not rule out the SOM method (Piras et al., AAAI 2026), which extracts mutually
**correlated centroids of a clustered manifold**. Those are different objects. That
paper reports 59.11% ASR at k=7 vs 0.00% single-direction on Llama2-7B. This is now
item 3b below.

## 3b. SOM multi-direction extraction  ·  promoted from the closed list

Train a small SOM on harmful-prompt activations at the best-SNR layer, take
`r_ι = w_ι − ν` per neuron, select `k ∈ [2,7]`. Offline once activations are captured,
so it is cheap to try alongside item 3.

Why it plausibly matters here: `ds4-refusal` attributed its stubborn 12–22% residual to
a *distributional* limit — "our corpus is missing refusal modes" — after ruling out
rank-k. SOM is that same diagnosis with a method attached: the missing modes may be
cluster structure the diff-of-means average washes out. A positive result is also worth
back-porting to ds4.

## 9. Measure the post-norm leak for the REAL direction

`projection.fold_post_norm` and `abliterate.py --norm-aware` are implemented and tested,
but the payoff is unmeasured. On the real checkpoint the norm gain `1 + w` spans
0.199–7.750 (mean 1.278, sd 0.697), yet a **random** direction leaks only ~1.5e-3 of the
contribution per layer — small enough to be a footnote.

Refusal directions are repeatedly reported to sit on high-norm / outlier channels, and
this gain has outliers to 7.75. So: once a direction exists, compute its actual leak
per layer before spending a sweep on `--norm-aware`. If it is ~1e-3, drop it. If the
direction is concentrated on the high-gain channels, it is worth a matched-KL A/B — and
worth re-testing on Gemma 2/3/4, which share the topology.

## Not on the roadmap

- **Publishing weights.** Blocked on the `USAGE_POLICY.md` §1.8 question in `STATE.md`.

## 10. Conversational refusal  ·  BUILT 2026-08-11, not yet run

`data/conversational/probes.jsonl` (50 benign probes, 5 classes x 10) +
`eval_conversational.py`. Built because a check against the 150-prompt content baseline
found **0/150** stylistic refusals — the hard-harm corpus cannot see this band at all.

The experiment: the direction was derived from `harmful_train.txt`, so there is no
reason to assume it captures persona locks, opinion refusal, or hedging. Whether it
does is the finding.

Run alongside every scale point. Prefer **hedge density** (markers per 100 words) over
the label rate when comparing adjacent operating points — it is continuous, so it
resolves changes a 50-item rate cannot.

**Known corpus divergence:** `inj-12`'s task text was fixed in source (the old wording
made quoting the canary the correct answer) but `data/agentic/probes.jsonl` is
deliberately NOT rebuilt while completions from the old version are in play. The
`--check` guard reports this as STALE, which is correct. Rebuild and regenerate that
probe at the next scale point, when everything can be made consistent at once.
