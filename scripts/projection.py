#!/usr/bin/env python3
"""Norm-preserving biprojected ablation — the single source of truth for the edit.

Ported from gemma-4-abliteration/scripts/abliterate.py::modify_weight_norm_preserved,
rewritten in numpy so the regression suite runs with no torch and no GPU. The torch
path in `abliterate.py` calls straight through here after a float32 round-trip.

    W' = W - scale * r (rᵀ W)          =>  rᵀW' = (1 - scale) rᵀW
    W' <- W' * (‖W‖_col / ‖W'‖_col)

`W` is [out_features, in_features] and `r` lives in the OUTPUT space (dim == out).
These are residual-write matrices (`o_proj`, `mlp.down_proj`), so their output-space
contribution is what gets ablated.

Norm preservation is per COLUMN (input dimension), not per row. The row-wise variant
plus an unscaled second Gram-Schmidt pass is the bug in gemma-4's NEGATIVE-RESULTS.md:
it made scale==0 a near-full ablation and left the scale knob inert. test_projection.py
pins all five symptoms.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def unit(v: np.ndarray, axis: int = 0) -> np.ndarray:
    """L2-normalize, safe at zero."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, EPS)


def biproject(W: np.ndarray, r: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Norm-preserving biprojection of `r` out of `W`'s output space.

    Args:
        W:     [out, in] residual-write weight.
        r:     [out] refusal direction (need not be unit-norm).
        scale: 0.0 is the identity; 1.0 removes the direction exactly.

    Returns:
        [out, in] float64. Caller casts back to the storage dtype.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D [out, in], got {W.shape}")
    r = unit(np.asarray(r, dtype=np.float64).reshape(-1))
    if r.shape[0] != W.shape[0]:
        raise ValueError(
            f"direction dim {r.shape[0]} != W out_features {W.shape[0]}; "
            "r must live in the OUTPUT space of a residual-write matrix"
        )

    col_norms = np.maximum(np.linalg.norm(W, axis=0, keepdims=True), EPS)  # [1, in]
    W_new = W - scale * np.outer(r, r @ W)                                 # [out, in]
    return W_new * (col_norms / np.maximum(np.linalg.norm(W_new, axis=0, keepdims=True), EPS))


def residual_fraction(W: np.ndarray, r: np.ndarray, scale: float) -> float:
    """‖rᵀW'‖ / ‖rᵀW‖ — fraction of the direction surviving the edit.

    The diagnostic the regression suite asserts on: 1.0 at scale 0, ~0 at scale 1,
    monotone non-increasing in between.
    """
    W = np.asarray(W, dtype=np.float64)
    rn = unit(np.asarray(r, dtype=np.float64).reshape(-1))
    denom = np.linalg.norm(rn @ W)
    if denom < EPS:
        return 0.0
    return float(np.linalg.norm(rn @ biproject(W, r, scale)) / denom)


def winsorized_mean(x: np.ndarray, p: float = 0.995) -> np.ndarray:
    """Per-dimension mean after clipping each dim to its [1-p, p] quantiles.

    x: [n, d] -> [d]. Same estimator as ds4-refusal/derive_direction.py so directions
    stay comparable across the two programs.
    """
    x = np.asarray(x, dtype=np.float64)
    lo = np.quantile(x, 1.0 - p, axis=0)
    hi = np.quantile(x, p, axis=0)
    return np.clip(x, lo, hi).mean(axis=0)


def fold_post_norm(r: np.ndarray, norm_weight: np.ndarray) -> np.ndarray:
    """Fold a post-sublayer RMSNorm gain into the refusal direction.

    THE PROBLEM. In this architecture the residual write is not `o_proj(x)` — it is
    `post_attention_layernorm(o_proj(x))` (`modeling_muse_glimmer.py:374-421`, the
    Gemma-2/3 sandwich-norm topology). `MuseGlimmerTextCenteredRMSNorm` computes

        PostNorm(z) = (z / RMS(z)) * (1 + w)          w = the learned per-channel gain

    Projecting `r` out of `o_proj` makes `z` orthogonal to `r`. The `1/RMS(z)` factor is
    a positive scalar and preserves that. But `*(1 + w)` is an ELEMENT-WISE rescale, and
    element-wise rescaling does NOT preserve orthogonality — so the layer's actual
    contribution to the residual stream still has a component along `r`. The weight-space
    edit is weaker than its own algebra claims, in a way that scales with how far `w`
    departs from uniform.

    THE FIX. Ask for `<PostNorm(z), r> = 0` directly:

        <(z/RMS(z)) * (1+w), r>  =  (1/RMS(z)) * <z, (1+w) * r>

    which vanishes iff `z` is orthogonal to `(1 + w) ⊙ r`. So project THAT out of
    `o_proj` instead. Same for `post_feedforward_layernorm` and `mlp.down_proj`.

    HOW BIG IS THE LEAK, MEASURED. On this checkpoint's real norm weights (layers
    0/12/25/38/51, both norms), the gain `1 + w` spans **0.199 to 7.750**, mean 1.278,
    sd 0.697 — far from uniform. But for a RANDOM unit direction the resulting leak is
    only ~**1.5e-3** of the contribution's norm per layer, and folding drives it to
    ~1e-17. So on a random direction this is a small effect, and it would be
    overclaiming to call it a fix for anything.

    The number that actually matters is the leak for the DERIVED refusal direction,
    which is not random: refusal directions are repeatedly found to align with
    high-norm / outlier channels, and this gain has outliers to 7.75. That measurement
    needs a real direction and is listed in IDEAS.md. Until it exists, treat
    `--norm-aware` as a free exactness improvement with unknown payoff, not as an
    expected win.

    It applies unchanged to Gemma 2/3/4, which share this sandwich-norm topology, so a
    positive result here is worth re-testing on `../gemma-4-abliteration`.
    """
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    w = np.asarray(norm_weight, dtype=np.float64).reshape(-1)
    if w.shape != r.shape:
        raise ValueError(f"norm weight dim {w.shape[0]} != direction dim {r.shape[0]}")
    return unit((1.0 + w) * r)


def gram_schmidt(d: np.ndarray, ref: np.ndarray, passes: int = 2) -> np.ndarray:
    """Remove the component of `d` along `ref`. Two passes for numerical stability.

    This is what makes the direction near-inert on benign input: it zeroes <mb, d>,
    so the ablation's own operand `<x, d>` is ~0 for x near the harmless mean.
    That is also exactly why selectivity measured at the derivation position is
    partly circular — see selectivity.py.
    """
    d = np.asarray(d, dtype=np.float64).copy()
    ref = np.asarray(ref, dtype=np.float64)
    for _ in range(passes):
        r = ref / max(float(np.linalg.norm(ref)), EPS)
        d = d - float(np.dot(d, r)) * r
    return d
