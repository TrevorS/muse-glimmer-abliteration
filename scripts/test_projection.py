#!/usr/bin/env python3
"""Regression tests for the norm-preserving refusal projection.

Guards the bug in gemma-4-abliteration/NEGATIVE-RESULTS.md, where the second
Gram-Schmidt pass dropped the `scale` factor and norm preservation used the wrong
axis. Symptoms: even `scale == 0` performed a near-full ablation, and the scale knob
was inert (every scale produced the same edit).

Shapes mirror Muse Glimmer's two residual-write matrices:
    o_proj        [6656, 4096]    (hidden, n_heads*head_dim)   — used at full size
    mlp.down_proj [6656, 19968]   (hidden, intermediate)       — narrowed 4x here

`down_proj` is narrowed because `biproject` promotes to float64 internally, so the real
[6656, 19968] shape costs ~1 GB per intermediate and ~4 GB peak per call — enough to get
the suite OOM-killed on a busy machine (observed: SIGTERM, exit 143). Every property
under test is about ORIENTATION (which axis the direction lives on, which axis the norm
is restored along), and orientation bugs surface identically at any width as long as
out_features != in_features and the two matrices differ. What must stay exact is
`out_features == 6656`, since that is the dimension the direction has to match.

Pure numpy on CPU — no torch, no model.

    python scripts/test_projection.py     # standalone, prints a table
    pytest scripts/test_projection.py     # CI-style
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from projection import biproject, fold_post_norm, residual_fraction, unit  # noqa: E402

HIDDEN, ATTN_IN = 6656, 4096          # real
MLP_IN = 19968 // 4                   # narrowed; see the module docstring
SHAPES = {"o_proj": (HIDDEN, ATTN_IN), "down_proj": (HIDDEN, MLP_IN)}

# Scale-sweep tests call biproject 7+ times. Those assert on the shape of the
# residual-vs-scale curve, which is width-independent, so they run on a small matrix;
# the orientation-sensitive tests above keep the real out_features.
SWEEP = (832, 512)


def _fixture(shape=(HIDDEN, ATTN_IN), seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape), rng.standard_normal(shape[0])


def test_identity_at_scale_zero():
    for name, shape in SHAPES.items():
        W, r = _fixture(shape)
        assert np.allclose(biproject(W, r, 0.0), W, atol=1e-8, rtol=1e-8), \
            f"{name}: scale=0 must be the identity"


def test_full_ablation_at_scale_one():
    for name, shape in SHAPES.items():
        W, r = _fixture(shape)
        assert residual_fraction(W, r, 1.0) < 1e-6, f"{name}: scale=1 must remove the direction"


def test_residual_monotonic_in_scale():
    W, r = _fixture(SWEEP)
    scales = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    res = [residual_fraction(W, r, s) for s in scales]
    assert abs(res[0] - 1.0) < 1e-8, "scale=0 must leave the direction untouched"
    for a, b in zip(res, res[1:]):
        assert b <= a + 1e-9, f"residual must be non-increasing in scale: {res}"
    assert res[0] - res[-1] > 0.9, f"scale must actually move the residual: {res}"


def test_column_norms_preserved():
    for name, shape in SHAPES.items():
        W, r = _fixture(shape)
        Wp = biproject(W, r, 0.7)
        n0 = np.linalg.norm(W, axis=0)
        n1 = np.linalg.norm(Wp, axis=0)
        assert np.allclose(n0, n1, atol=1e-8, rtol=1e-8), f"{name}: per-column norms must be preserved"


def test_scale_changes_the_edit():
    """Distinct scales must produce distinct matrices (the original bug did not)."""
    W, r = _fixture(SWEEP)
    a = biproject(W, r, 0.3)
    b = biproject(W, r, 0.7)
    rel = np.linalg.norm(a - b) / np.linalg.norm(W)
    assert rel > 1e-3, f"scale=0.3 and 0.7 must differ, got relative diff {rel:.2e}"


def _post_norm(z, w, eps=1e-8):
    """MuseGlimmerTextCenteredRMSNorm: (z / RMS(z)) * (1 + w)."""
    return (z / np.sqrt((z ** 2).mean() + eps)) * (1.0 + w)


def test_fold_post_norm_is_identity_at_unit_gain():
    """w == 0 means gain 1, so the direction must be unchanged."""
    r = np.random.default_rng(0).standard_normal(HIDDEN)
    assert np.allclose(fold_post_norm(r, np.zeros(HIDDEN)), unit(r), atol=1e-12)


def test_fold_post_norm_rejects_dim_mismatch():
    r = np.random.default_rng(0).standard_normal(HIDDEN)
    try:
        fold_post_norm(r, np.zeros(ATTN_IN))
    except ValueError:
        return
    raise AssertionError("must reject a norm gain that does not match the direction dim")


def test_plain_projection_leaks_through_the_post_norm():
    """The defect fold_post_norm exists to fix.

    Projecting r out of o_proj makes its raw output orthogonal to r, but the residual
    write is post_attention_layernorm(o_proj(x)), and the norm's element-wise gain
    re-introduces a component along r. Asserted as a real leak, so that if a future
    refactor makes it vanish this test tells us rather than silently passing.
    """
    rng = np.random.default_rng(7)
    W = rng.standard_normal((HIDDEN, ATTN_IN))
    r = unit(rng.standard_normal(HIDDEN))
    w = rng.standard_normal(HIDDEN) * 0.5          # non-uniform learned gain
    x = rng.standard_normal(ATTN_IN)

    z = biproject(W, r, 1.0) @ x
    assert abs(z @ r) < 1e-8, "sanity: raw projection output must be orthogonal to r"

    contribution = _post_norm(z, w)
    leak = abs(contribution @ r) / np.linalg.norm(contribution)
    assert leak > 1e-3, f"expected the post-norm gain to re-introduce r, got leak {leak:.2e}"


def test_norm_aware_projection_zeroes_the_residual_contribution():
    """With the gain folded in, the LAYER'S RESIDUAL WRITE is orthogonal to r."""
    rng = np.random.default_rng(7)
    W = rng.standard_normal((HIDDEN, ATTN_IN))
    r = unit(rng.standard_normal(HIDDEN))
    w = rng.standard_normal(HIDDEN) * 0.5

    Wp = biproject(W, fold_post_norm(r, w), 1.0)
    for seed in range(5):                            # must hold for ANY input
        x = np.random.default_rng(100 + seed).standard_normal(ATTN_IN)
        contribution = _post_norm(Wp @ x, w)
        leak = abs(contribution @ r) / np.linalg.norm(contribution)
        assert leak < 1e-9, f"seed {seed}: residual contribution must be orthogonal to r, leak {leak:.2e}"


def test_rejects_direction_in_wrong_space():
    """A direction sized to the INPUT dim is the classic transposed-weight mistake."""
    W = np.random.default_rng(0).standard_normal((HIDDEN, MLP_IN))
    bad = np.random.default_rng(1).standard_normal(MLP_IN)
    try:
        biproject(W, bad, 1.0)
    except ValueError:
        return
    raise AssertionError("must reject a direction that does not live in the output space")


def _main() -> int:
    print(f"Fixture: W[{SWEEP[0]},{SWEEP[1]}], random refusal direction\n")
    print(f"{'scale':>6}  {'residual ||rTW\'||/||rTW||':>26}  {'col-norm err':>13}")
    print("-" * 50)
    W, r = _fixture(SWEEP)
    n0 = np.linalg.norm(W, axis=0)
    for s in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        Wp = biproject(W, r, s)
        colerr = np.abs(np.linalg.norm(Wp, axis=0) - n0).max()
        print(f"{s:>6.2f}  {residual_fraction(W, r, s):>26.6f}  {colerr:>13.2e}")

    tests = [
        test_identity_at_scale_zero,
        test_full_ablation_at_scale_one,
        test_residual_monotonic_in_scale,
        test_column_norms_preserved,
        test_scale_changes_the_edit,
        test_fold_post_norm_is_identity_at_unit_gain,
        test_fold_post_norm_rejects_dim_mismatch,
        test_plain_projection_leaks_through_the_post_norm,
        test_norm_aware_projection_zeroes_the_residual_contribution,
        test_rejects_direction_in_wrong_space,
    ]
    print()
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
