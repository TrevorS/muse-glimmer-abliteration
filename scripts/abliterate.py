#!/usr/bin/env python3
"""Apply norm-preserving biprojection to Muse Glimmer's weights. Shard-by-shard, CPU.

Stage 3 of the pipeline (capture -> derive -> APPLY). This is a pure safetensors edit:
each shard is streamed, the targeted tensors are rewritten, everything else is copied
unchanged. No model instantiation, NO GPU, and peak memory is one shard rather than the
60 GB a full load would take. Same motivation as gemma-4's export_31b.py.

torch is imported only as a bfloat16-capable container for safetensors I/O — numpy has
no bf16 dtype, so the numpy safetensors backend cannot open this checkpoint at all. All
arithmetic still happens in numpy float32 inside projection.py, which is what the CPU
regression suite pins. Runs fine in the heretic env (torch 2.11.0+cu130); that env's
transformers is too old for the muse_glimmer *architecture*, but nothing here
instantiates the model, so it does not matter.

TARGETS — the two residual-write matrices per text decoder layer:

    model.language_model.layers.{i}.self_attn.o_proj.weight    [6656, 4096]
    model.language_model.layers.{i}.mlp.down_proj.weight       [6656, 19968]

Both have out_features == hidden_size == 6656, which is the space the refusal direction
lives in. That is the whole requirement for the edit to be meaningful; projection.py
raises if a direction of the wrong dimension is passed.

DO NOT WIDEN THE TARGET SET BY NAME GLOB. This architecture has TWO tensors per layer
called `gate_proj`:

    model.language_model.layers.{i}.mlp.gate_proj.weight         [19968, 6656]  MLP gate
    model.language_model.layers.{i}.self_attn.gate_proj.weight   [4096, 6656]   attention gate

104 of them in the checkpoint. Neither is a residual-write matrix (both have
out_features != 6656), so neither is a valid target, and a suffix-match on "gate_proj"
would silently hit both. The vision tower (800 tensors) is never touched: a direction
derived from text prompts says nothing about the image pathway.

LAYER BAND. `--top-pct` mirrors gemma-4's finding that the Unified 12B carried its
refusal signal only in the upper layers (L15-47, abliterate 70% not 100%). Default here
is 100%, but derive_direction.py's SNR profile is what should pick the band — not raw
separation, which is a magnitude trap (STRATEGY.md:147).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from projection import biproject, fold_post_norm  # noqa: E402

O_PROJ = re.compile(r"^model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight$")
DOWN_PROJ = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.weight$")

# The post-sublayer norm each target's output passes through before the residual add.
# See projection.fold_post_norm for why this matters.
POST_NORM_FOR = {
    "o_proj": "model.language_model.layers.{i}.post_attention_layernorm.weight",
    "down_proj": "model.language_model.layers.{i}.post_feedforward_layernorm.weight",
}

# Copied verbatim, never edited. Listed so a future reader can see these were considered.
NEVER_TOUCH = re.compile(
    r"^model\.vision_tower\.|^model\.vision_adapter\.|^model\.vision_projection\."
    r"|\.self_attn\.gate_proj\.weight$|\.mlp\.gate_proj\.weight$|\.mlp\.up_proj\.weight$"
    r"|\.self_attn\.[qkv]_proj\.weight$|layernorm|^lm_head|embed_tokens|^model\.language_model\.norm"
)


def target_layer(name: str) -> tuple[int, str] | None:
    m = O_PROJ.match(name)
    if m:
        return int(m.group(1)), "o_proj"
    m = DOWN_PROJ.match(name)
    if m:
        return int(m.group(1)), "down_proj"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/Muse-Glimmer-30B")
    ap.add_argument("--directions", required=True, help=".npy [L, hidden] from derive_direction.py")
    ap.add_argument("--out", required=True, help="output model directory")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--top-pct", type=float, default=100.0,
                    help="edit only the top N%% of layers (upper band); 100 = all")
    ap.add_argument("--layers", help="explicit comma/range list, e.g. '15-47' (overrides --top-pct)")
    ap.add_argument("--norm-aware", action="store_true",
                    help="fold the post-sublayer RMSNorm gain into the direction, so the "
                         "layer's RESIDUAL CONTRIBUTION is orthogonal to r rather than "
                         "just o_proj/down_proj's raw output (see projection.fold_post_norm)")
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    a = ap.parse_args()

    src = Path(a.model)
    dirs = np.load(a.directions)
    if dirs.ndim != 2:
        raise SystemExit(f"directions must be [L, hidden], got {dirs.shape}")
    L, D = dirs.shape

    meta_path = Path(a.directions).with_suffix(".json")
    if meta_path.exists():
        dm = json.loads(meta_path.read_text())
        if not dm.get("gate_passed", True):
            print(f"REFUSING: {meta_path} records gate_passed=false "
                  f"(forced={dm.get('gate_forced')}). Re-derive; do not apply a "
                  f"direction with anti-selective layers.")
            return 1
        print(f"direction gate: PASSED on {dm.get('gate_basis')} activations")
    else:
        print(f"WARNING: no {meta_path.name} beside the direction — gate status unknown")

    if a.layers:
        want = set()
        for part in a.layers.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                want |= set(range(int(lo), int(hi) + 1))
            else:
                want.add(int(part))
    else:
        first = int(round(L * (1.0 - a.top_pct / 100.0)))
        want = set(range(first, L))
    print(f"directions {dirs.shape}; editing {len(want)} of {L} layers: "
          f"{min(want)}..{max(want)}  scale={a.scale}")

    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))

    # Pre-load the post-norm gains when --norm-aware. Small ([6656] each), and pulling
    # them up front keeps the edit loop a single streaming pass per shard.
    norms: dict[tuple[int, str], np.ndarray] = {}
    if a.norm_aware:
        need: dict[str, list[tuple[int, str, str]]] = {}
        for layer in sorted(want):
            for kind, tpl in POST_NORM_FOR.items():
                key = tpl.format(i=layer)
                shard = weight_map.get(key)
                if shard is None:
                    raise SystemExit(f"--norm-aware: {key} not in the checkpoint index")
                need.setdefault(shard, []).append((layer, kind, key))
        for shard, items in need.items():
            with safe_open(str(src / shard), framework="pt") as f:
                for layer, kind, key in items:
                    norms[(layer, kind)] = f.get_tensor(key).float().numpy()
        print(f"--norm-aware: loaded {len(norms)} post-norm gains "
              f"(residual contribution, not raw projection output, is made orthogonal to r)")

    out = Path(a.out)
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and not f.name.endswith(".safetensors"):
                shutil.copy2(f, out / f.name)

    edited, skipped, copied = [], [], 0
    for shard in shards:
        print(f"\n{shard}")
        tensors = {}
        with safe_open(str(src / shard), framework="pt") as f:
            for k in f.keys():
                tl = target_layer(k)
                # Dry run inspects shapes only — no tensor data is read, so the plan
                # comes back in a second instead of streaming 60 GB.
                shape = tuple(f.get_slice(k).get_shape())

                if tl is None:
                    if not NEVER_TOUCH.search(k) and ".layers." in k:
                        skipped.append(k)
                    if not a.dry_run:
                        tensors[k] = f.get_tensor(k)
                    copied += 1
                    continue
                layer, kind = tl
                if layer not in want:
                    if not a.dry_run:
                        tensors[k] = f.get_tensor(k)
                    copied += 1
                    continue
                if shape[0] != D:
                    raise SystemExit(
                        f"{k}: out_features {shape[0]} != direction dim {D}. "
                        "This is not a residual-write matrix; refusing to edit it.")
                if not a.dry_run:
                    t = f.get_tensor(k)
                    r = dirs[layer]
                    if a.norm_aware:
                        r = fold_post_norm(r, norms[(layer, kind)])
                    edit = biproject(t.float().numpy(), r, a.scale)
                    tensors[k] = torch.from_numpy(edit.astype(np.float32)).to(t.dtype)
                edited.append((k, kind, layer, shape))
                if sys.stdout.isatty():
                    print(f"  edit {kind:10s} L{layer:<3d} {str(shape):>16s}", end="\r", flush=True)
        print(f"  {len(edited)} edited so far, {copied} copied      ")
        if not a.dry_run:
            save_file(tensors, str(out / shard), metadata={"format": "pt"})
        del tensors

    n_o = sum(1 for e in edited if e[1] == "o_proj")
    n_d = sum(1 for e in edited if e[1] == "down_proj")
    print(f"\n{'DRY RUN — ' if a.dry_run else ''}edited {len(edited)} tensors "
          f"({n_o} o_proj + {n_d} down_proj across {len(want)} layers); copied {copied}")
    if skipped:
        print(f"NOTE: {len(skipped)} per-layer tensors neither targeted nor on the "
              f"never-touch list — check these are intentional:")
        for k in sorted(set(re.sub(r'\.\d+\.', '.N.', s) for s in skipped)):
            print(f"    {k}")

    if not a.dry_run:
        (out / "abliteration.json").write_text(json.dumps({
            "base_model": str(src), "directions": a.directions, "scale": a.scale,
            "top_pct": a.top_pct, "layers": sorted(want), "norm_aware": a.norm_aware,
            "edited": [{"tensor": k, "kind": kd, "layer": l} for k, kd, l, _ in edited],
        }, indent=2))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
