#!/usr/bin/env python3
"""Selectivity — the regression check that would have caught the shipped-noortho bug.

From ds4-refusal/FINDINGS-20260807.md §2:

    selectivity(l) = mean|<harmful_acts, d_l>| / mean|<harmless_acts, d_l>|

`<x, d>` is exactly what the ablation removes (`x' = x - s*<x,d>*d`), so this is not
a proxy — it is the operator's own operand. `sel < 1` means layer l deletes more
benign signal than refusal signal. Every direction that ever worked in that program
had 0 anti-selective layers; every broken one had 11-20. Binary, no threshold to tune.

WHAT THIS IS NOT (§2, "Limit (and it is real)"). Gram-Schmidt zeroes <mb, d> by
construction, and for the broken variant that term was 98.5% of the denominator. So on
the DERIVATION activations this metric largely restates "was the Gram-Schmidt applied."
arXiv:2603.22061 makes the general form of the critique: deriving a direction from
activations and scoring it on those same activations is circular.

Therefore:
  * on derivation acts  -> a REGRESSION CHECK on a known defect. Gate on it, do not rank on it.
  * on held-out acts    -> a real measurement. `derive_direction.py --acts-holdout` routes here
                           and the JSON records which one was used.

Never use selectivity to compare two methods. SRA, sparsification, and response-side
directions all optimize related quantities and would score well by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class LayerSelectivity:
    layer: int
    harmful: float   # mean |<x, d>| over harmful activations
    harmless: float  # mean |<x, d>| over harmless activations
    ratio: float     # harmful / harmless; < 1.0 is anti-selective

    @property
    def anti_selective(self) -> bool:
        return self.ratio < 1.0


def layer_selectivity(G_l: np.ndarray, B_l: np.ndarray, d_l: np.ndarray, layer: int) -> LayerSelectivity:
    """Selectivity of direction `d_l` at one layer.

    G_l: [n, d] harmful activations.  B_l: [m, d] harmless.  d_l: [d] direction.
    """
    d = np.asarray(d_l, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(d))
    d = d / n if n > 1e-8 else d
    harmful = float(np.abs(np.asarray(G_l, dtype=np.float64) @ d).mean())
    harmless = float(np.abs(np.asarray(B_l, dtype=np.float64) @ d).mean())
    ratio = harmful / harmless if harmless > 1e-12 else float("inf")
    return LayerSelectivity(layer=layer, harmful=harmful, harmless=harmless, ratio=ratio)


def selectivity_profile(G: np.ndarray, B: np.ndarray, dirs: np.ndarray) -> list[LayerSelectivity]:
    """Per-layer selectivity. G/B: [n, L, d]; dirs: [L, d]."""
    L = dirs.shape[0]
    if G.shape[1] != L or B.shape[1] != L:
        raise ValueError(f"layer mismatch: G {G.shape}, B {B.shape}, dirs {dirs.shape}")
    return [layer_selectivity(G[:, l], B[:, l], dirs[l], l) for l in range(L)]


def gate(profile: list[LayerSelectivity]) -> tuple[bool, list[int]]:
    """The hard gate: `anti_selective_layers == 0`.

    Returns (passed, anti_selective_layer_indices). One second, no model load — this
    is meant to run before a direction is ever allowed near a GPU.
    """
    bad = [s.layer for s in profile if s.anti_selective]
    return (not bad), bad


def summarize(profile: list[LayerSelectivity], top: int = 5) -> dict:
    ratios = np.array([s.ratio for s in profile], dtype=np.float64)
    finite = ratios[np.isfinite(ratios)]
    passed, bad = gate(profile)
    worst = sorted(profile, key=lambda s: s.ratio)[:top]
    return {
        "passed": passed,
        "n_layers": len(profile),
        "anti_selective_layers": bad,
        "n_anti_selective": len(bad),
        "mean_selectivity": float(finite.mean()) if finite.size else float("nan"),
        "min_selectivity": float(finite.min()) if finite.size else float("nan"),
        "worst_layers": [asdict(s) for s in worst],
    }


def format_report(profile: list[LayerSelectivity], label: str = "") -> str:
    s = summarize(profile)
    head = f"selectivity [{label}]" if label else "selectivity"
    lines = [
        f"{head}: mean {s['mean_selectivity']:.2f}  min {s['min_selectivity']:.2f}  "
        f"anti-selective {s['n_anti_selective']}/{s['n_layers']}"
    ]
    if not s["passed"]:
        lines.append(f"  ANTI-SELECTIVE LAYERS: {s['anti_selective_layers']}")
        for w in s["worst_layers"]:
            if w["ratio"] < 1.0:
                lines.append(
                    f"    L{w['layer']:<3d} removes {w['harmless']:8.2f} from harmless, "
                    f"{w['harmful']:8.2f} from harmful  -> ratio {w['ratio']:.3f}  INVERTED"
                )
    return "\n".join(lines)
