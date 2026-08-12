#!/usr/bin/env python3
"""Regression tests for the selectivity gate.

Pins the defect from ds4-refusal/FINDINGS-20260807.md §1-2: a direction derived
WITHOUT Gram-Schmidt against the harmless mean is anti-selective — it removes more
benign signal than refusal signal at `<x, d>`, the ablation's own operand. That
direction shipped and served for weeks. The gate catches it in one second with no
model load, so these tests are the thing standing between a bad derivation and a GPU.

Two synthetic fixtures, both with known ground truth:

`_fixture` — benign case. harmful = harmless + delta, with delta drawn independently
of mu_b. In high dimension two random vectors are near-orthogonal (cos ~ 1/sqrt(D)),
so Gram-Schmidt has almost nothing to remove and BOTH derivations pass. This is the
control: it shows the gate does not simply fire on every un-orthogonalized direction.

`_fixture_magnitude_trap` — the defect. The harmful mean is *attenuated* along the
dominant shared axis: mu_g = beta*mu_b + delta_perp with beta < 1. Now the raw
difference-of-means is dominated by -(1-beta)*mu_b, so an un-orthogonalized direction
is essentially -unit(mu_b) — it points along the benign mean. It then removes
||mu_b|| from harmless but only beta*||mu_b|| from harmful: ratio ~ beta < 1,
anti-selective. That is the L42 situation in §2 (removes 68.35 from harmless, 8.95
from harmful, ratio 0.13) and the reason the noortho direction shipped broken.
Gram-Schmidt recovers delta_perp and the ratio inverts to >> 1.

Pure numpy on CPU.

    python scripts/test_selectivity.py
    pytest scripts/test_selectivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derive_direction import derive  # noqa: E402
from selectivity import gate, layer_selectivity, selectivity_profile, summarize  # noqa: E402

N, L, D = 128, 8, 256


def _fixture(seed=0, mu_scale=30.0, delta_scale=1.0):
    """Control: refusal offset drawn independently of the harmless mean."""
    rng = np.random.default_rng(seed)
    mu_b = rng.standard_normal((L, D)) * mu_scale
    delta = rng.standard_normal((L, D)) * delta_scale
    B = mu_b[None] + rng.standard_normal((N, L, D))
    G = (mu_b + delta)[None] + rng.standard_normal((N, L, D))
    return G.astype(np.float32), B.astype(np.float32)


def _fixture_magnitude_trap(seed=1, mu_scale=30.0, beta=0.85, delta_scale=1.0):
    """The defect: harmful mean attenuated along the dominant shared axis.

    mu_g = beta*mu_b + delta_perp, beta < 1. The raw diff-of-means is then dominated
    by -(1-beta)*mu_b, so without Gram-Schmidt the direction points along the BENIGN
    mean and deletes more harmless signal than harmful.
    """
    rng = np.random.default_rng(seed)
    mu_b = rng.standard_normal((L, D)) * mu_scale
    delta = rng.standard_normal((L, D)) * delta_scale
    # component of delta orthogonal to mu_b, per layer — the true refusal signal
    for l in range(L):
        u = mu_b[l] / np.linalg.norm(mu_b[l])
        delta[l] -= np.dot(delta[l], u) * u
    mu_g = beta * mu_b + delta
    B = mu_b[None] + rng.standard_normal((N, L, D))
    G = mu_g[None] + rng.standard_normal((N, L, D))
    return G.astype(np.float32), B.astype(np.float32)


def test_orthogonalized_direction_passes_gate():
    for name, fx in (("control", _fixture), ("magnitude-trap", _fixture_magnitude_trap)):
        G, B = fx()
        dirs, _, _ = derive(G, B, winsorize=0.995, orthogonalize=True)
        passed, bad = gate(selectivity_profile(G, B, dirs))
        assert passed, f"{name}: orthogonalized direction must pass; anti-selective layers {bad}"


def test_unorthogonalized_direction_fails_gate():
    """The shipped-noortho defect must be caught on the trap fixture."""
    G, B = _fixture_magnitude_trap()
    dirs, _, _ = derive(G, B, winsorize=0.995, orthogonalize=False)
    passed, bad = gate(selectivity_profile(G, B, dirs))
    assert not passed, "un-orthogonalized direction under an attenuated shared axis must FAIL"
    assert len(bad) == L, f"expected every layer anti-selective, got {len(bad)}/{L}"


def test_control_fixture_does_not_false_positive():
    """The gate must not fire merely because Gram-Schmidt was skipped.

    When the refusal offset is near-orthogonal to the harmless mean to begin with,
    skipping orthogonalization is harmless and the gate must stay quiet — otherwise
    it is just re-reporting a flag, not measuring geometry.
    """
    G, B = _fixture()
    dirs, _, _ = derive(G, B, winsorize=0.995, orthogonalize=False)
    passed, bad = gate(selectivity_profile(G, B, dirs))
    assert passed, f"control fixture must not trip the gate; got anti-selective {bad}"


def test_inverted_layer_is_detected_exactly():
    """A hand-built inverted layer: ratio < 1 and named in the gate output."""
    rng = np.random.default_rng(3)
    d = rng.standard_normal(D)
    d /= np.linalg.norm(d)
    # harmless projects hard onto d, harmful barely does -> ratio << 1
    B_l = np.outer(rng.normal(50.0, 1.0, N), d) + rng.standard_normal((N, D))
    G_l = np.outer(rng.normal(2.0, 1.0, N), d) + rng.standard_normal((N, D))
    s = layer_selectivity(G_l, B_l, d, layer=42)
    assert s.anti_selective, f"expected anti-selective, ratio={s.ratio:.3f}"
    assert s.ratio < 0.2, f"expected strong inversion, ratio={s.ratio:.3f}"


def test_ratio_is_scale_invariant_in_direction():
    """Selectivity must not depend on ||d|| — it is normalized internally.

    Tolerance is float32-relative: `dirs` is stored float32, so rescaling and
    re-normalizing round-trips through that precision.
    """
    G, B = _fixture()
    dirs, _, _ = derive(G, B, 0.995, True)
    a = layer_selectivity(G[:, 0], B[:, 0], dirs[0], 0).ratio
    b = layer_selectivity(G[:, 0], B[:, 0], dirs[0] * 137.0, 0).ratio
    assert abs(a - b) <= 1e-5 * max(abs(a), 1.0), \
        f"ratio must be invariant to direction scaling: {a} vs {b}"


def test_summary_counts_match_profile():
    G, B = _fixture_magnitude_trap()
    dirs, _, _ = derive(G, B, 0.995, False)
    prof = selectivity_profile(G, B, dirs)
    s = summarize(prof)
    assert s["n_anti_selective"] == len([p for p in prof if p.anti_selective])
    assert s["n_layers"] == L
    assert s["passed"] == (s["n_anti_selective"] == 0)


def _main() -> int:
    print(f"Fixtures: {N} pairs x {L} layers x {D} dims\n")
    print(f"{'fixture':>16}  {'derivation':>16}  {'mean sel':>9}  {'min sel':>9}  {'anti-sel':>10}")
    print("-" * 70)
    for fname, fx in (("control", _fixture), ("magnitude-trap", _fixture_magnitude_trap)):
        G, B = fx()
        for label, ortho in (("orthogonalized", True), ("no-orthogonalize", False)):
            dirs, _, _ = derive(G, B, 0.995, ortho)
            s = summarize(selectivity_profile(G, B, dirs))
            flag = "" if s["passed"] else "  <-- GATE FAILS"
            print(f"{fname:>16}  {label:>16}  {s['mean_selectivity']:>9.2f}  "
                  f"{s['min_selectivity']:>9.2f}  {s['n_anti_selective']:>4d}/{s['n_layers']:<4d}{flag}")

    tests = [
        test_orthogonalized_direction_passes_gate,
        test_unorthogonalized_direction_fails_gate,
        test_control_fixture_does_not_false_positive,
        test_inverted_layer_is_detected_exactly,
        test_ratio_is_scale_invariant_in_direction,
        test_summary_counts_match_profile,
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
