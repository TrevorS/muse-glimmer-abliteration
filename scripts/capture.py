#!/usr/bin/env python3
"""Capture per-layer residual activations for harmful/harmless prompts. GPU stage.

Stage 1 of the pipeline (CAPTURE -> derive -> apply). Writes an .npz with
G [n, L, d] harmful and B [m, L, d] harmless activations that derive_direction.py
consumes unchanged.

POSITION SCOPE — the reason this script has a `--position` flag at all.

FINDINGS-20260807.md §5 identifies the open gap in the previous program: every
direction was derived, and its selectivity measured, at ONE position type — the final
prompt token, which is also where Gram-Schmidt zeroes the harmless mean. Nothing
establishes that the direction stays inert on benign *continuations*, and a long
conversation is almost entirely continuations. That is the standing explanation for the
multi-turn degradation Teej observed at scale 3.0 on ds4.

§5 also names the decisive test, and it is cheap: capture at prompt-final, mid-response,
and turn depths, then compute selectivity per layer per position on HELD-OUT captures.
Selectivity collapses away from the derivation position -> hypothesis holds. Flat ->
hypothesis dead, implement CAST instead, do not rescue it.

So this supports:
    --position prompt_final   last prompt token (the classic derivation position)
    --position prompt_mean    mean over prompt tokens
    --position response       positions inside a generated continuation (--gen-tokens)

Capture the derivation set at prompt_final to reproduce prior art; capture a held-out
set at each position to run the §5 test. Pass --holdout to tag the output metadata so
derive_direction.py's --acts-holdout can never be pointed at a derivation capture by
accident.

MUSE GLIMMER SPECIFICS
  * text tower is `model.model.language_model` (a MuseGlimmerTextModel); the vision
    tower is a sibling and is NOT touched. Text-only capture says nothing about
    image-conditioned refusal — see IDEAS.md.
  * 52 layers, hidden 6656.
  * left padding, so the final prompt token is index -1 for every row in the batch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.muse_glimmer import MuseGlimmerForConditionalGeneration


def load_prompts(p: str) -> list[str]:
    return [l.strip() for l in Path(p).read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def text_layers(model):
    """The 52 decoder layers of the text tower."""
    return model.model.language_model.layers


def build_inputs(tok, prompts: list[str], device):
    chats = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
             for p in prompts]
    enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def capture(model, tok, prompts, position, batch_size, gen_tokens, device):
    layers = text_layers(model)
    L = len(layers)
    buf: dict[int, torch.Tensor] = {}
    # Reduce inside the hook, not after. Keeping the full [B, S, 6656] for all 52
    # layers is 52*B*S*6656*2 bytes — ~0.5 GB at B=8/S=93 today, but it scales with
    # sequence length and would be the first thing to OOM if the corpus or the
    # generation budget grows. The reduced form is 52*B*6656*2 = ~5 MB.
    reduce_mode = {"prompt_final": "last", "prompt_mean": "masked_mean"}.get(position, "keep")
    mask_ref: dict[str, torch.Tensor] = {}

    def mk_hook(i):
        def hook(_module, _args, out):
            h = out[0] if isinstance(out, tuple) else out
            if reduce_mode == "last":
                buf[i] = h[:, -1, :].detach().float()
            elif reduce_mode == "masked_mean":
                m = mask_ref["m"]
                buf[i] = ((h.detach().float() * m).sum(1) / m.sum(1))
            else:
                buf[i] = h.detach()
        return hook

    handles = [l.register_forward_hook(mk_hook(i)) for i, l in enumerate(layers)]
    rows = []
    try:
        for s in range(0, len(prompts), batch_size):
            chunk = prompts[s:s + batch_size]
            enc = build_inputs(tok, chunk, device)

            if position == "response":
                if gen_tokens < 1:
                    raise SystemExit("--position response needs --gen-tokens >= 1")
                out = model.generate(**enc, max_new_tokens=gen_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id,
                                     return_dict_in_generate=True)
                # Re-run prompt+continuation once so every layer's hook fires over the
                # response positions in a single forward. Cleaner than stitching
                # per-step decode captures, and it matches how the ablation is applied
                # (teacher-forced over the whole sequence).
                seq = out.sequences
                start = enc["input_ids"].shape[1]
                # Rows that hit EOS early are pad-filled to the batch max. Marking those
                # positions as attended would average real activations together with
                # padding — and short refusals finish first, so the contamination would
                # land hardest on exactly the rows the direction is derived from.
                gen_mask = (seq[:, start:] != tok.pad_token_id).long()
                attn = torch.cat([enc["attention_mask"], gen_mask], dim=1)
                buf.clear()
                model(input_ids=seq, attention_mask=attn)
                m = gen_mask.unsqueeze(-1).float()
                denom = m.sum(1).clamp(min=1.0)
                acts = np.stack([((buf[i][:, start:, :].float() * m).sum(1) / denom).cpu().numpy()
                                 for i in range(L)], axis=1)
            else:
                if reduce_mode == "masked_mean":
                    mask_ref["m"] = enc["attention_mask"].unsqueeze(-1).float()
                model(**enc)
                acts = np.stack([buf[i].cpu().numpy() for i in range(L)], axis=1)

            rows.append(acts.astype(np.float32))
            buf.clear()
            print(f"  {min(s + batch_size, len(prompts))}/{len(prompts)}", end="\r", flush=True)
    finally:
        for h in handles:
            h.remove()
    print()
    return np.concatenate(rows, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/Muse-Glimmer-30B")
    ap.add_argument("--harmful", required=True)
    ap.add_argument("--harmless", required=True)
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--position", default="prompt_final",
                    choices=["prompt_final", "prompt_mean", "response"])
    ap.add_argument("--gen-tokens", type=int, default=0, help="for --position response")
    ap.add_argument("--limit", type=int, default=0, help="cap prompts per side (0 = all)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--holdout", action="store_true",
                    help="tag this capture as held-out (required for honest selectivity)")
    a = ap.parse_args()

    harmful, harmless = load_prompts(a.harmful), load_prompts(a.harmless)
    if a.limit:
        harmful, harmless = harmful[:a.limit], harmless[:a.limit]
    print(f"harmful={len(harmful)}  harmless={len(harmless)}  position={a.position}")

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        a.model, dtype=getattr(torch, a.dtype), device_map="auto", attn_implementation=a.attn)
    model.eval()
    device = next(model.parameters()).device
    L = len(text_layers(model))
    print(f"loaded: {L} text layers, hidden {model.config.text_config.hidden_size}, device {device}")

    print("capturing harmful:")
    G = capture(model, tok, harmful, a.position, a.batch_size, a.gen_tokens, device)
    print("capturing harmless:")
    B = capture(model, tok, harmless, a.position, a.batch_size, a.gen_tokens, device)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, G=G, B=B)
    meta = {"model": a.model, "position": a.position, "gen_tokens": a.gen_tokens,
            "harmful_file": a.harmful, "harmless_file": a.harmless,
            "shapes": {"G": list(G.shape), "B": list(B.shape)}, "holdout": a.holdout}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}  G={G.shape} B={B.shape}  holdout={a.holdout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
