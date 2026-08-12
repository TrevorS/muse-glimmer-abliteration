# Runbook

Ordered commands for a full pass. Every GPU stage is preceded by the check that would
have caught the last program's mistakes.

## Measured costs (2026-08-11, GB10)

| stage | cost |
| --- | --- |
| model load | **~6 min** (1436 tensors, 59.58 GB) — pay it once per process |
| capture, 800 prompts | seconds (one forward per prompt) |
| generation, batch 4 | **0.80 rows/min** |
| generation, batch 16 | **1.89 rows/min** (2.36x, *not* 4x — see below) |

Generation is autoregressive: up to 1536 sequential decode steps per item, each
streaming all 60 GB of weights. GPU sits at **94%** — this is bandwidth, not a bug.

**Batching gives sub-linear returns because a batch runs until its LONGEST member
finishes.** Measured completion lengths: min 76, p50 347, p90 649, max 893 tokens. A
batch of 16 therefore pays ~893 steps while the median needs 347 — roughly 2.6x waste,
which is why 4x batch bought 2.36x throughput. Bigger batches make this worse, not
better. (llama.cpp continuous batching does not have this problem — `IDEAS.md` 2b.)

`--max-new-tokens 1536` is correct: **0/30 rows truncated, 0 unscoreable.** Do not lower
it; 512 left 14% of rows with no final answer at all.

```bash
# CPU stages: uv run python      GPU stages: .venv-gpu/bin/python
```

## Long GPU jobs must be DETACHED  ·  the operational finding of 2026-08-11

Background shell tasks in this environment get killed, and the kill takes their child
processes with it. Six long GPU runs died this way. It is not OOM (no kernel OOM events,
memory was fine), not a wall-clock limit (one run survived 40 min), and not caused by
issuing concurrent commands (a run died with nothing else running).

**Proof it is process-group reaping:** a `setsid`-detached generation job kept running
at 100% CPU with `PPID 1` while the background task that was *watching* it was killed.
Same machine, same moment.

So: launch long jobs into their own session, and poll with SHORT foreground commands.

```bash
cat > /tmp/gen.sh <<'EOF'
#!/usr/bin/env bash
cd /home/trevor/Projects/muse-glimmer-abliteration
exec .venv-gpu/bin/python -u scripts/generate.py --mode content \
  --prompts data/splits/harmful_tune.txt --out runs/base-content.jsonl \
  --batch-size 8 --max-new-tokens 1536 --resume
EOF
chmod +x /tmp/gen.sh
setsid nohup /tmp/gen.sh > /tmp/gen.log 2>&1 < /dev/null &
```

Verify it detached — `PPID` must be 1:

```bash
ps -o pid,ppid,etimes,%cpu -p "$(pgrep -f 'python -u scripts/generate' | head -1)"
```

`python -u` matters: without it stdout is block-buffered when redirected, so the log
looks frozen for minutes. Combine with `--resume` and a kill costs at most one batch.

## 0. Before touching the GPU

```bash
scripts/run_tests.sh          # 54 CPU regressions, ~7s
nvidia-smi                    # confirm the GB10 is idle
```

## 1. Capture  [GPU, ~7 min incl. load]

Derive set and held-out set in one script so the model loads twice at most:

```bash
.venv-gpu/bin/python scripts/capture.py \
  --harmful data/harmful_train.txt --harmless data/harmless_train.txt \
  --out acts/derive.npz --position prompt_final --batch-size 16

.venv-gpu/bin/python scripts/capture.py \
  --harmful data/splits/harmful_tune.txt --harmless data/splits/harmless_tune.txt \
  --out acts/holdout.npz --position prompt_final --batch-size 16 --holdout
```

Sanity before moving on: all finite, no all-zero layers, residual norms rising with
depth. Expect max **raw separation at L51** and peak **SNR around L41-48** — that gap is
the magnitude trap, and SNR is the one to trust.

## 2. Derive + gate  [CPU, seconds]

```bash
uv run python scripts/derive_direction.py \
  --acts acts/derive.npz --acts-holdout acts/holdout.npz --out directions/v1
```

Stop and read the output. **The gate is not advisory** — if any layer is
anti-selective, fix the derivation, do not reach for `--force`.

`adj_cos_median` is descriptive only. Measured **0.834** on this model with a clean
gate; do not read a high value as a defect (see `RESEARCH-JOURNAL.md`, 2026-08-10).

## 3. Baseline generation  [GPU, one load for both axes]

Baseline before any edit — otherwise there is nothing to pair against.

```bash
.venv-gpu/bin/python scripts/generate.py --mode content \
  --prompts data/splits/harmful_tune.txt --out runs/base-content.jsonl --batch-size 16 --resume
.venv-gpu/bin/python scripts/generate.py --mode content \
  --prompts data/splits/harmless_tune.txt --out runs/base-harmless.jsonl --batch-size 16 --resume
.venv-gpu/bin/python scripts/generate.py --mode agentic \
  --out runs/base-agentic.jsonl --batch-size 16 --resume
```

Always pass `--resume`. Generation flushes each batch, so a kill costs one batch instead
of the run — long jobs on this box have been killed mid-run four times, and re-invoking
with `--resume` skips the (id, seed) pairs already written.

```bash
uv run python scripts/eval_content.py --completions runs/base-content.jsonl --show-audit 10
uv run python scripts/eval_agentic.py --completions runs/base-agentic.jsonl
```

**Write the predictions in `RESEARCH-JOURNAL.md` before running this.** A baseline that
surprises you is worth more than a sweep that confirms you.

If the baseline agentic fail rate is already high, the axis has no headroom and the
probes need to get harder before any of this measures anything.

## 4. Apply  [CPU, shard-by-shard]

Dry run first, always — it costs a second and prints the exact tensor list:

```bash
.venv-gpu/bin/python scripts/abliterate.py \
  --directions directions/v1.npy --out models/mg-abl-s1.0 --scale 1.0 --dry-run
```

Expect **104 edited (52 o_proj + 52 down_proj), 1332 copied, 1436 total**. Then drop
`--dry-run`. Each output model is a full ~60 GB copy — check disk first, and delete
losing candidates.

Cheaper alternative for the sweep: **runtime control vectors** (`IDEAS.md` 2b). llama.cpp
supports this arch and official GGUFs exist, so scale becomes a runtime flag instead of a
60 GB re-export per operating point. Prefer this for the sweep; reserve the weight bake
for the final chosen point.

## 5. Evaluate  [GPU generate, CPU score]

```bash
.venv-gpu/bin/python scripts/generate.py --mode content --model models/mg-abl-s1.0 \
  --prompts data/splits/harmful_tune.txt --out runs/abl-s1.0-content.jsonl --batch-size 16 --resume
.venv-gpu/bin/python scripts/generate.py --mode agentic --model models/mg-abl-s1.0 \
  --out runs/abl-s1.0-agentic.jsonl --batch-size 16 --resume

uv run python scripts/eval_content.py --completions runs/abl-s1.0-content.jsonl \
  --baseline runs/base-content.jsonl --json-out experiments/s1.0-content.json
uv run python scripts/eval_agentic.py --completions runs/abl-s1.0-agentic.jsonl \
  --baseline runs/base-agentic.jsonl --json-out experiments/s1.0-agentic.json
```

Read the **paired transitions**, not the rate delta. A flip only counts when it lands on
`comply` — refusals converting to `deflect` or `broken` are the over-ablation signature.

## Scale sweep

`0` (identity control), `0.5`, `1.0`, `1.5`, `2.0`, `2.5`, `3.0`. **Do not assume 1.0** —
the ds4 optimum was 3.0, and `s > 1` is a different intervention rather than overdriving.
Include `scale=0` explicitly: it verifies the applier is a no-op at zero, which is exactly
what the old projection bug broke.

Both axes at every scale. A content-only number on this model is not a result.

## Hard rules

- TEST (`data/splits/harmful_test.txt`, 250) is touched **once per candidate, at the end**.
  Never tune against it.
- **One greedy run per config is enough.** Seeds are a no-op without `--sample`
  (measured 30/30 byte-identical). Multi-seed under greedy costs Nx for nothing.
  The real variance axis is batch composition; hold `--batch-size` fixed across
  conditions you intend to compare, and vary it deliberately if you want a noise
  estimate.
- Never put a Muse Glimmer KL in the same table as a Gemma one (logit softcapping).
- No HF upload pending the `USAGE_POLICY.md` §1.8 question in `STATE.md`.
