#!/usr/bin/env python3
"""Derive a per-layer refusal direction from captured activations. Pure numpy, no GPU.

Stage 2 of the pipeline (capture -> DERIVE -> apply). Generalized from
ds4-refusal's 43x4096 hardcoded version: shape comes from the .npz, so this runs
unchanged on Muse Glimmer's 52x6656 text tower.

    d[l] = unit( gram_schmidt( winsorized_mean(G[:,l]) - winsorized_mean(B[:,l]),
                               winsorized_mean(B[:,l]) ) )

Positive apply scale SUPPRESSES refusal.

THE GATE. Unlike the ds4 version, `anti_selective_layers == 0` is enforced here and
the direction is NOT written when it fails. In ds4-refusal that check was added only
after a `--no-orthogonalize` direction reached production and served for weeks
(FINDINGS-20260807.md §1): it unlocked 8.8% of refusals while breaking ~10% of harmful
and ~5% of harmless outputs, and raised benign over-refusal 2.7% -> 9.3%. The gate is
one second and needs no model load. It runs before anything touches a GPU.

Pass `--acts-holdout` to score selectivity on captures the direction was NOT derived
from. On derivation activations the metric is partly circular (selectivity.py explains
why); the JSON records which basis was used so no downstream reader has to guess.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from projection import gram_schmidt, winsorized_mean  # noqa: E402
from selectivity import format_report, selectivity_profile, summarize  # noqa: E402


def load_acts(path: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    missing = {"G", "B"} - set(z.files)
    if missing:
        raise SystemExit(f"{path}: missing array(s) {sorted(missing)}; found {z.files}")
    G, B = z["G"].astype(np.float32), z["B"].astype(np.float32)
    if G.ndim != 3 or B.ndim != 3:
        raise SystemExit(f"expected [n, L, d] arrays, got G {G.shape} B {B.shape}")
    if G.shape[1:] != B.shape[1:]:
        raise SystemExit(f"G/B layer-dim mismatch: {G.shape[1:]} vs {B.shape[1:]}")
    return G, B


def derive(G: np.ndarray, B: np.ndarray, winsorize: float, orthogonalize: bool):
    """Returns (dirs [L,d], per-layer separation, per-layer SNR)."""
    _, L, D = G.shape
    dirs = np.zeros((L, D), dtype=np.float32)
    seps, snrs = [], []
    for l in range(L):
        mg = winsorized_mean(G[:, l], winsorize)
        mb = winsorized_mean(B[:, l], winsorize)
        d = mg - mb
        sep = float(np.linalg.norm(d))
        # SNR = separation / typical residual norm. STRATEGY.md:147 — raw separation
        # is a magnitude trap: the last layers have huge residual norms and rank top
        # by raw sep while carrying little refusal signal. Rank by this instead.
        resid = float(np.linalg.norm(np.concatenate([G[:, l], B[:, l]], axis=0), axis=1).mean())
        seps.append(sep)
        snrs.append(sep / resid if resid > 1e-8 else 0.0)
        if orthogonalize:
            d = gram_schmidt(d, mb)
        n = float(np.linalg.norm(d))
        dirs[l] = (d / n if n > 1e-8 else d).astype(np.float32)
    return dirs, seps, snrs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--acts", required=True, help=".npz with G [n,L,d] and B [m,L,d]")
    ap.add_argument("--acts-holdout", help=".npz of held-out captures for an honest selectivity score")
    ap.add_argument("--out", required=True, help="output basename (writes .npy and .json)")
    ap.add_argument("--winsorize", type=float, default=0.995)
    ap.add_argument("--no-orthogonalize", action="store_true",
                    help="skip Gram-Schmidt. This is the shipped-noortho defect; the gate will fail.")
    ap.add_argument("--force", action="store_true",
                    help="write the direction even if the selectivity gate fails (records gate_forced=true)")
    a = ap.parse_args()

    G, B = load_acts(a.acts)
    _, L, D = G.shape
    print(f"loaded G={G.shape} B={B.shape}  ->  {L} layers x {D} dims")

    dirs, seps, snrs = derive(G, B, a.winsorize, not a.no_orthogonalize)

    adj = [float(np.dot(dirs[l], dirs[l + 1])) for l in range(L - 1)]
    adj_cos = float(np.median(np.abs(adj))) if adj else 0.0

    # Gate on derivation acts always; on held-out too when supplied.
    prof_derive = selectivity_profile(G, B, dirs)
    print(format_report(prof_derive, "derivation"))
    reports = {"derivation": summarize(prof_derive)}
    gate_basis = "derivation"

    if a.acts_holdout:
        Gh, Bh = load_acts(a.acts_holdout)
        if Gh.shape[1:] != (L, D):
            raise SystemExit(f"holdout shape {Gh.shape[1:]} != derivation {(L, D)}")
        prof_hold = selectivity_profile(Gh, Bh, dirs)
        print(format_report(prof_hold, "held-out"))
        reports["holdout"] = summarize(prof_hold)
        gate_basis = "holdout"

    passed = all(r["passed"] for r in reports.values())

    top_snr = sorted(range(L), key=lambda i: -snrs[i])[:10]
    top_sep = sorted(range(L), key=lambda i: -seps[i])[:10]
    # adj_cos is DESCRIPTIVE, not a pass/fail signal. An earlier version of this line
    # called a high value "a tell for a bad direction", generalizing from ds4's
    # refusal_iq3 (adj_cos_median 0.952). But that direction also had 20/42
    # anti-selective layers — the anti-selectivity was the defect and the collinearity
    # rode along with it. Muse Glimmer measures 0.834 with 0/52 anti-selective on
    # HELD-OUT activations and mean selectivity 7.49, i.e. high alignment and a clean
    # direction at the same time. Judge the direction by the gate; read this for shape.
    print(f"\nadj_cos_median={adj_cos:.4f}   (descriptive: how fast the direction rotates "
          f"layer-to-layer; NOT a pass/fail signal — the gate is)")
    print(f"top-10 by SNR: {top_snr}")
    print(f"top-10 by raw sep: {top_sep}   (overlap {len(set(top_snr) & set(top_sep))}/10 — "
          "rank by SNR, not sep)")

    if not passed and not a.force:
        print(f"\nGATE FAILED on {gate_basis} activations — direction NOT written.")
        print("An anti-selective layer removes more benign signal than refusal signal at the")
        print("exact quantity the ablation operates on. Fix the derivation (is Gram-Schmidt on?)")
        print("rather than compensating with scale. Use --force only to reproduce a known defect.")
        return 1

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out.with_suffix(".npy")), dirs)

    meta = {
        "format": "muse-glimmer-refusal-direction-v1",
        "shape": [L, D],
        "pairs": int(min(G.shape[0], B.shape[0])),
        "winsorize": a.winsorize,
        "orthogonalize": not a.no_orthogonalize,
        "adj_cos_median": adj_cos,
        "sep_per_layer": [round(s, 4) for s in seps],
        "snr_per_layer": [round(s, 4) for s in snrs],
        "top10_by_snr": top_snr,
        "top10_by_sep": top_sep,
        "acts": a.acts,
        "acts_holdout": a.acts_holdout,
        "gate_basis": gate_basis,
        "gate_passed": passed,
        "gate_forced": bool(a.force and not passed),
        "selectivity": reports,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    status = "PASSED" if passed else "FORCED (FAILING)"
    print(f"\nwrote {out.with_suffix('.npy')} / {out.with_suffix('.json')}   gate: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
