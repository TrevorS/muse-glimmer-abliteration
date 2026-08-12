#!/usr/bin/env python3
"""Generate completions for the content and agentic eval sets. GPU stage.

Generation is deliberately separate from scoring. STRATEGY.md's corrected methodology
(2026-05-25) exists because the first ds4 sweep classified inline against a live server
and undercounted refusal ~2x without anyone noticing for weeks. Generate once, save
everything, classify offline, and the classifier can be fixed and re-run for free.

    --mode content   one turn per prompt from a splits file
    --mode agentic   replays data/agentic/probes.jsonl, including the tool-result turn,
                     and records structured tool_calls when the model emits them

Output is jsonl, one row per item, with the raw text kept verbatim:
    {"id", "prompt"|"probe", "text", "tool_calls", "stop_reason", "n_tokens"}

DETERMINISM — measured on this model, 2026-08-11, and it is NOT what ds4 found.

Greedy decode here is exactly reproducible at a FIXED batch composition: two runs at
--seeds 1 and 2 produced **30/30 byte-identical** completions. That is expected —
torch.manual_seed does nothing when do_sample=False — but it means **--seeds is pure
waste under greedy**, costing Nx compute for one run's worth of information.

What DOES change the output is **batch composition**. Seed 0 above was generated across
two runs with different groupings (16+6 vs 16+14 items) and differs from seeds 1/2 on
**30/30** rows. Different batch shape changes padding and matmul reduction order, which
shifts logits enough to flip an occasional argmax, and greedy decode diverges from there.

So the honest noise axis for this harness is batch composition, not seed. Two ways to
get a real estimate:
    * vary --batch-size between otherwise identical runs (cheap, tests what actually varies)
    * pass --sample with different --seeds (tests sampling noise, a different question)

Encouraging result: across that batch-composition difference, **0/30 probe verdicts
changed**. The token streams differed on every row and the measurement did not move.
That is a stronger stability claim than "3 seeds agreed" would have been.

ds4's ±7-10pt free-running floor (STRATEGY.md:101, FINDINGS §6) came from GPU MoE routing
nondeterminism on a live server. This is a dense model run offline; do not import that
floor here without measuring it.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.muse_glimmer import MuseGlimmerForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent.parent


def load_lines(p: str) -> list[str]:
    return [l.strip() for l in Path(p).read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def load_jsonl(p: str) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


# Shared with eval_agentic.py. This model emits <atem:invoke name="...">, not JSON.
from toolcalls import extract as parse_tool_calls  # noqa: E402,F401
from channels import split as split_channels  # noqa: E402


def run_via_api(a, items):
    """Drive an OpenAI-compatible server (llama-server) instead of a local model.

    The server applies the GGUF's own chat template, so responses come back in the same
    channelled form as local generation and go through the identical channel split and
    scorer. That is the point: a quant measured a different way would not be comparable
    to the bf16 numbers.
    """
    import urllib.request
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.resume and out.exists():
        done = {(r["id"], r.get("seed")) for r in load_jsonl(str(out))}
        print(f"resume: {len(done)} rows already present")
    todo = [it for it in items if (it["id"], 0) not in done]
    fh = out.open("a")
    lock = threading.Lock()
    n = 0

    def one(it):
        nonlocal n
        msgs = ([{"role": "user", "content": it["prompt"]}]
                if a.mode in ("content", "conversational") else it["messages"])
        body = json.dumps({"messages": msgs, "max_tokens": a.max_new_tokens,
                           "temperature": 0, "stream": False}).encode()
        req = urllib.request.Request(a.api_base.rstrip("/") + "/chat/completions",
                                     data=body, headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1200) as r:
                d = json.loads(r.read())
            msg = d["choices"][0]["message"]
            # llama-server already separates the channels: `content` is the final
            # answer and `reasoning_content` the to=self deliberation. That makes
            # `content` exactly equivalent to the `final` channel used for the bf16
            # runs, so quant numbers are directly comparable to them.
            raw = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            fin = d["choices"][0].get("finish_reason")
        except Exception as e:
            print(f"  {it['id']}: request failed: {e}")
            return
        ch = split_channels(raw)
        final = ch.final if ch.segments and ch.parsed_from != "none" else raw.strip()
        row = json.dumps({
            "id": it["id"], "raw": raw, "text": raw, "final": final,
            "reasoning_chars": len(reasoning) or len(ch.reasoning),
            "tool_calls": ch.tool_calls,
            "scoreable": bool(final) or bool(ch.tool_calls), "n_tokens": d.get("usage", {}).get("completion_tokens", 0),
            "stop_reason": "length" if fin == "length" else "stop", "seed": 0,
        })
        # Serialised writes only; the requests themselves run concurrently against
        # llama-server's parallel slots, which is where the throughput comes from.
        with lock:
            fh.write(row + "\n")
            fh.flush()
            n += 1
            print(f"  {n}/{len(todo)} {it['id']}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=a.api_concurrency) as ex:
        list(ex.map(one, todo))
    fh.close()
    print(f"wrote {out} ({n} new rows)")
    return 0


@torch.no_grad()
def run(model, tok, batches, max_new_tokens, do_sample, temperature, device, sink=None):
    """Generate, flushing each batch through `sink` as soon as it lands.

    Incremental on purpose. Long generation jobs in this environment have been killed
    mid-run three times, and a run that buffers everything until the end loses all of it.
    Paired with --resume, a kill costs at most one batch.
    """
    out_rows = []
    for bi, batch in enumerate(batches):
        enc = tok(batch["texts"], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tok.pad_token_id, return_dict_in_generate=True)
        start = enc["input_ids"].shape[1]
        for j, seq in enumerate(gen.sequences):
            new = seq[start:]
            # Keep special tokens. This model channels its output as
            #   <|start|>assistant to=self|user|<tool><|message|>...<|eom|>
            # and stripping the delimiters makes the channel name run into the first
            # word of its body ("to=selfSummarize"), which is only recoverable by
            # guessing. The raw form is unambiguous; `text` stays for readability.
            raw = tok.decode(new, skip_special_tokens=False)
            text = tok.decode(new, skip_special_tokens=True)
            ch = split_channels(raw)
            n = int((new != tok.pad_token_id).sum())
            out_rows.append({
                "id": batch["ids"][j],
                "raw": raw,
                "text": text,
                "final": ch.final,
                "reasoning_chars": len(ch.reasoning),
                "tool_calls": ch.tool_calls,
                "scoreable": ch.scoreable,
                "n_tokens": n,
                "stop_reason": "length" if n >= max_new_tokens else "stop",
            })
        if sink is not None:
            sink(out_rows[-len(batch["ids"]):])
        print(f"  batch {bi + 1}/{len(batches)}", flush=True)
    return out_rows


def make_batches(items, tok, mode, batch_size):
    texts, ids = [], []
    for it in items:
        if mode in ("content", "conversational"):
            msgs = [{"role": "user", "content": it["prompt"]}]
        else:
            msgs = it["messages"]
        texts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            tools=it.get("tools") if mode == "agentic" else None))
        ids.append(it["id"])
    return [{"texts": texts[i:i + batch_size], "ids": ids[i:i + batch_size]}
            for i in range(0, len(texts), batch_size)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/Muse-Glimmer-30B")
    ap.add_argument("--mode", required=True,
                    choices=["content", "agentic", "conversational"])
    ap.add_argument("--prompts", help="content mode: a splits .txt")
    ap.add_argument("--probes", default=str(HERE / "data" / "agentic" / "probes.jsonl"))
    ap.add_argument("--conv-probes",
                    default=str(HERE / "data" / "conversational" / "probes.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1536,
                    help="this model reasons on a to=self channel before answering; at 512 "
                         "the first baseline left 14%% of rows with no final answer at all")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--sample", action="store_true", help="off by default; greedy")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="only meaningful with --sample; under greedy decode seeds are a "
                         "no-op (measured: 30/30 byte-identical across two seeds)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--api-base",
                    help="OpenAI-compatible endpoint (e.g. llama-server at "
                         "http://127.0.0.1:8080/v1). When set, no local model is loaded — "
                         "used to measure GGUF quants through the same harness and scorer.")
    ap.add_argument("--api-concurrency", type=int, default=8,
                    help="parallel in-flight requests; match llama-server's -np")
    ap.add_argument("--resume", action="store_true",
                    help="append to --out, skipping (id, seed) pairs already present")
    a = ap.parse_args()

    if a.mode == "content":
        if not a.prompts:
            raise SystemExit("--mode content requires --prompts")
        items = [{"id": f"{Path(a.prompts).stem}-{i:04d}", "prompt": p}
                 for i, p in enumerate(load_lines(a.prompts))]
    elif a.mode == "conversational":
        # Benign prompts that probe persona / opinion / tone / speculation / hedging.
        # Same single-user-turn shape as content mode; kept separate so the id comes
        # from the corpus rather than a line number, which is what the scorer joins on.
        items = load_jsonl(a.conv_probes)
    else:
        items = load_jsonl(a.probes)
    print(f"mode={a.mode}  items={len(items)}  seeds={a.seeds}")
    if len(a.seeds) > 1 and not a.sample:
        print(f"WARNING: {len(a.seeds)} seeds with greedy decode (--sample off). "
              f"manual_seed has no effect here — measured 30/30 byte-identical across "
              f"seeds. This costs {len(a.seeds)}x compute for one run of information.")
        print("         For a real noise estimate vary --batch-size instead, or add --sample.")

    if a.api_base:
        return run_via_api(a, items)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        a.model, dtype=getattr(torch, a.dtype), device_map="auto", attn_implementation=a.attn)
    model.eval()
    device = next(model.parameters()).device

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, int]] = set()
    if a.resume and out.exists():
        for r in load_jsonl(str(out)):
            done.add((r["id"], r.get("seed")))
        print(f"resume: {len(done)} (id, seed) pairs already present in {out}")
    elif out.exists():
        out.unlink()

    fh = out.open("a")

    def flush(rows):
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.flush()

    all_rows = []
    for seed in a.seeds:
        todo = [it for it in items if (it["id"], seed) not in done]
        if not todo:
            print(f"seed {seed}: already complete, skipping")
            continue
        torch.manual_seed(seed)
        print(f"seed {seed}: {len(todo)}/{len(items)} remaining")
        batches = make_batches(todo, tok, a.mode, a.batch_size)
        rows = run(model, tok, batches, a.max_new_tokens, a.sample, a.temperature, device,
                   sink=lambda rs, _s=seed: flush([dict(r, seed=_s) for r in rs]))
        for r in rows:
            r["seed"] = seed
        all_rows += rows
    fh.close()

    unscoreable = [r for r in all_rows if not r["scoreable"]]
    if unscoreable:
        print(f"WARNING: {len(unscoreable)}/{len(all_rows)} rows have no final answer and "
              f"no tool call (truncated mid-reasoning). These are NOT passes.")
        print(f"         Raise --max-new-tokens (currently {a.max_new_tokens}) and re-run.")
    meta = {"model": a.model, "mode": a.mode, "prompts": a.prompts, "probes": a.probes,
            "max_new_tokens": a.max_new_tokens, "sample": a.sample,
            "temperature": a.temperature if a.sample else None, "seeds": a.seeds,
            "n_items": len(items), "n_rows": len(all_rows)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}  ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
