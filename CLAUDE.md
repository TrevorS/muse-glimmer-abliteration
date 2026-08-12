# Abliteration Research Loop — Muse Glimmer

Third target after `../gemma-4-abliteration` and `../ds4-refusal`. Read `README.md`
for why this one is not just a sixth model.

## The loop

1. Read `STATE.md` (current results, established constraints) and `IDEAS.md` (backlog).
2. Pick the highest-priority untried idea.
3. **Predict the outcome in `RESEARCH-JOURNAL.md` before running.** Both prior programs
   produced their best findings from surprises, and their worst from confirmations that
   nobody had pre-registered.
4. Run it. One experiment at a time on the GPU.
5. Record the result as JSON in `experiments/`.
6. Update `STATE.md` if it changes what is known — including negative results.
7. Mark the idea done in `IDEAS.md` with a one-line result.

## Non-negotiables

These are not style preferences. Each one is a specific defect that cost the prior
programs weeks.

- **The selectivity gate is not advisory.** `derive_direction.py` will not write a
  direction with anti-selective layers and `abliterate.py` will not apply one. If you
  reach for `--force`, you are reproducing `FINDINGS-20260807.md` §1 on purpose or you
  are wrong.
- **Never tune against TEST.** `data/splits/harmful_test.txt` (250) is touched once per
  candidate, at the end. Sweeps use `harmful_tune.txt`.
- **Generate once, score offline.** Never classify inline against a live server — that
  is how the first ds4 sweep undercounted refusal ~2×.
- **Do not use multi-seed under greedy — it is a no-op here.** Measured 2026-08-11:
  two seeds gave 30/30 byte-identical completions. The axis that actually varies is
  **batch composition** (different grouping changed 30/30 rows). For a noise estimate,
  vary `--batch-size`, or pass `--sample`. Do not import ds4's ±7-10pt floor; that came
  from MoE routing nondeterminism on a live server, not from this setup.
- **Report paired transitions, not aggregate rates.** An unchanged aggregate can hide a
  fully rotated set (`FINDINGS` §7).
- **Rank layers by SNR, never raw separation.**
- **`broken` is not `refused`.** A flip only counts when it lands on `comply`.
- **Both eval axes, every time.** A content-only result on this model is not a result;
  the whole point of the target is that its safety training is mostly agentic.

## Environments

```bash
uv run python ...          # CPU: transformers 5.15.0, numpy, jinja2 (pyproject.toml)
.venv-gpu/bin/python ...   # GPU: + torch 2.11.0+cu130, accelerate, safetensors
```

- CPU stages (`splits`, `derive_direction`, `eval_*`, all tests): `uv run python`.
  Deps live in `pyproject.toml`; add with `uv add`, never `pip install`.
- GPU stages (`capture`, `generate`): `.venv-gpu/bin/python`. Verified working —
  CUDA on GB10, and every module path the scripts hook resolves.
- **Do not upgrade the shared heretic env in place.** `gemma-4-abliteration` is
  validated on transformers 5.12.0, which has no `muse_glimmer`.
- `abliterate.py` is CPU-only; torch is just a bf16 container for safetensors I/O.

## Long GPU jobs must be detached

Background shell tasks here get killed and take their children with them — six runs
died that way. Proven by a `setsid`-detached job surviving at 100% CPU while the
background task watching it was killed. Launch with
`setsid nohup ... > log 2>&1 < /dev/null &`, verify `PPID` is 1, use `python -u`, and
poll with SHORT foreground commands. Always pass `--resume`. Details in `RUNBOOK.md`.

## Before touching a GPU

```bash
scripts/run_tests.sh     # 47 CPU regressions, no model load
```

## Architecture traps

- **Two `gate_proj` per layer** (`mlp.` and `self_attn.`), 104 total. Neither is a
  residual-write matrix. Never select targets by name suffix.
- **Softcapped logits** — KL here is not comparable to the Gemma 4 family. Do not put
  them in one table.
- **The vision tower is untouched** by any text-derived direction. Do not describe a
  result as covering the model when it covers the text pathway.

## Publishing

Blocked pending the `USAGE_POLICY.md` §1.8 question in `STATE.md`. Local measurement is
unaffected. Ask Teej before any HF upload.
