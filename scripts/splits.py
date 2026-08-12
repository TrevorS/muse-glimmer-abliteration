#!/usr/bin/env python3
"""Split hygiene — audit leakage, then build mutually disjoint DERIVE/TUNE/TEST sets.

Generalized from ds4-refusal's version (llama.cpp/experiments/refusal/splits.py), which
was written after the fact to clean up an existing mess. Here it runs BEFORE any
direction exists, which is the whole point.

What went wrong in the previous program, and what this prevents (STATE-20260801.md:91,
FINDINGS-20260807.md §6):

    harmful_train  -> harmful_hard        3 / 80
    harmful_train  -> harmful_eval_full   7 / 676
    harmful_hard  subset of eval_full    80 / 80     (hard80 was never independent)
    harmful_eval_150 ∩ harmful_hard      17          (the two "eval" sets overlapped)

Combined with a free-running noise floor around ±7-10pt (GPU nondeterminism at temp 0,
STRATEGY.md:101 and FINDINGS §6), that makes small direction-vs-direction deltas
uninterpretable. A direction tuned on a set that leaks into its test set looks better
than it is.

    DERIVE  activation-capture set. The direction is built from this.
    TUNE    sweeps — scale, winsorize, layer bands. Look at it as much as you like.
    TEST    touched ONCE per candidate, at the end. Never tuned against.

TEST is stratified on hard-set membership so TUNE and TEST carry the same difficulty
mix and stay comparable to each other.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"


def norm(s: str) -> str:
    """Match key. Punctuation/case-insensitive so trivial reformatting cannot hide a dup."""
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def load(p: Path) -> list[str]:
    return [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]


def audit(files: list[Path]) -> tuple[dict, dict]:
    sets = {p.name: load(p) for p in files}
    keys = {k: {norm(x) for x in v} for k, v in sets.items()}
    print(f"{'file':28s} {'n':>5} {'uniq':>6}")
    for k, v in sets.items():
        print(f"{k:28s} {len(v):>5} {len(keys[k]):>6}")
    names = list(sets)
    print("\npairwise overlap (normalized exact match)")
    print(f"{'':28s}" + "".join(f"{n[:12]:>14s}" for n in names))
    for a in names:
        print(f"{a:28s}" + "".join(f"{len(keys[a] & keys[b]):>14d}" for b in names))
    return sets, keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out-dir", default=str(DATA / "splits"))
    ap.add_argument("--seed", type=int, default=1337,
                    help="distinct from ds4-refusal's 42 so TEST here is an independent draw")
    ap.add_argument("--pool", default="harmful_eval_full.txt")
    ap.add_argument("--hard", default="harmful_hard.txt")
    ap.add_argument("--harmless-pool", default="harmless_eval.txt")
    ap.add_argument("--derive", nargs="+", default=["harmful_train.txt"],
                    help="corpora the direction will be built from; excluded from TUNE/TEST")
    ap.add_argument("--tune", type=int, default=150)
    ap.add_argument("--test", type=int, default=250)
    ap.add_argument("--audit-only", action="store_true")
    a = ap.parse_args()

    data = Path(a.data)
    print("=== AUDIT (before) ===")
    sets, keys = audit(sorted(data.glob("*.txt")))
    if a.audit_only:
        return 0

    missing = [f for f in [a.pool, a.hard, a.harmless_pool, *a.derive] if f not in sets]
    if missing:
        raise SystemExit(f"missing corpora in {data}: {missing}")

    derive: set[str] = set()
    for f in a.derive:
        derive |= keys[f]

    pool_raw = sets[a.pool]
    hard = keys[a.hard]

    clean = [x for x in pool_raw if norm(x) not in derive]
    dropped = len(pool_raw) - len(clean)

    hard_items = [x for x in clean if norm(x) in hard]
    easy_items = [x for x in clean if norm(x) not in hard]
    rng = random.Random(a.seed)
    rng.shuffle(hard_items)
    rng.shuffle(easy_items)

    want = a.tune + a.test
    if want > len(clean):
        print(f"\nWARN: requested {want} > clean pool {len(clean)}; shrinking proportionally")
        scale = len(clean) / want
        a.tune, a.test = int(a.tune * scale), int(a.test * scale)
    hard_frac = len(hard_items) / len(clean)

    def take(n: int) -> list[str]:
        nh = min(len(hard_items), round(n * hard_frac))
        out = [hard_items.pop() for _ in range(nh)] + [easy_items.pop() for _ in range(n - nh)]
        rng.shuffle(out)
        return out

    tune, test = take(a.tune), take(a.test)

    hl = sets[a.harmless_pool][:]
    rng.shuffle(hl)
    hl_tune, hl_test = hl[:len(hl) // 2], hl[len(hl) // 2:]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {
        "harmful_tune.txt": tune, "harmful_test.txt": test,
        "harmless_tune.txt": hl_tune, "harmless_test.txt": hl_test,
    }
    for name, items in written.items():
        (out / name).write_text("\n".join(items) + "\n")

    (out / "meta.json").write_text(json.dumps({
        "seed": a.seed,
        "source_pool": a.pool,
        "derive_union": a.derive,
        "dropped_as_derivation_leak": dropped,
        "clean_pool": len(clean),
        "hard_fraction_in_pool": round(hard_frac, 4),
        "sizes": {k: len(v) for k, v in written.items()},
        "note": "TEST is touched ONCE per candidate. Never tune against it.",
    }, indent=2))

    print(f"\n=== BUILT ({out}) ===")
    print(f"dropped {dropped} derivation-leaked prompts; clean pool {len(clean)}")
    for k, v in written.items():
        print(f"  {k:22s} {len(v):>4d}")

    print("\n=== AUDIT (after) ===")
    check = {
        "harmful_tune": {norm(x) for x in tune},
        "harmful_test": {norm(x) for x in test},
        "DERIVE": derive,
        "harmless_tune": {norm(x) for x in hl_tune},
        "harmless_test": {norm(x) for x in hl_test},
    }
    bad = 0
    names = list(check)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            n = len(check[x] & check[y])
            if n:
                bad += 1
            print(f"  {x:15s} x {y:15s} {n:>4d}{'' if n == 0 else '   <-- LEAK'}")
    print(f"\n{'ALL SPLITS DISJOINT' if not bad else f'{bad} LEAKING PAIR(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
