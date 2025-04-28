#!/usr/bin/env python
"""
check_marginals.py  —  sanity-check a HoloClean-style inference_marginals.pkl file.

Expected structure:
    {
        (tid, attr) : {candidate_value : probability_float, ...},
        ...
    }

Checks performed
---------------
1. Top-level object is a dict.
2. Each key is a 2-tuple.
3. Each value is a dict whose keys are hashable (usually str) and whose
   values are floats in [0, 1].
4. Optional but recommended: probabilities for each (tid, attr) sum to 1 ± tol.
"""

import argparse
import pickle
import sys
from pathlib import Path
from math import isclose

TOLERANCE = 1e-5   # tolerance for probability-sum check


def load_pickle(path: Path):
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to load {path}: {e}")


def check_structure(marginals: dict) -> list[str]:
    errors: list[str] = []

    if not isinstance(marginals, dict):
        return [f"Top-level object is {type(marginals)}, expected dict"]

    for key, inner in marginals.items():
        # --- key checks
        if not isinstance(key, tuple) or len(key) != 2:
            errors.append(f"Key {key!r} is not a 2-tuple (tid, attr)")
            continue
        tid, attr = key
        # (Optional) type hints: tid often int, attr str
        if not isinstance(attr, str):
            errors.append(f"attr in key {key!r} is not str (got {type(attr)})")

        # --- value checks
        if not isinstance(inner, dict):
            errors.append(f"Value for {key!r} is {type(inner)}, expected dict")
            continue
        if not inner:
            errors.append(f"Value dict for {key!r} is empty")
            continue

        prob_sum = 0.0
        for cand, prob in inner.items():
            if not isinstance(prob, (int, float)):
                errors.append(
                    f"Prob for candidate {cand!r} in {key!r} "
                    f"is {type(prob)}, expected float"
                )
            elif prob < 0 or prob > 1:
                errors.append(
                    f"Prob {prob} for candidate {cand!r} in {key!r} "
                    f"is out of range [0,1]"
                )
            prob_sum += float(prob)

        # --- probability-sum check
        if not isclose(prob_sum, 1.0, abs_tol=TOLERANCE):
            errors.append(
                f"Probabilities for {key!r} sum to {prob_sum:.6f}, "
                f"expected 1.0 ± {TOLERANCE}"
            )

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate the structure of inference_marginals.pkl"
    )
    parser.add_argument(
        "--path",
        default="inference_marginals.pkl",
        type=Path,
        help="Path to the pickle file (default: inference_marginals.pkl)",
    )
    parser.add_argument(
        "--no-sum-check",
        action="store_true",
        help="Disable the per-cell probability sum-to-1 check",
    )
    args = parser.parse_args()

    marginals = load_pickle(args.path)

    global TOLERANCE
    if args.no_sum_check:
        TOLERANCE = None  # skip sum check inside check_structure

    errors = check_structure(marginals)

    if errors:
        print("\n❌  STRUCTURE ERRORS DETECTED:")
        for msg in errors:
            print("   •", msg)
        print(f"\nFound {len(errors)} issue(s).")
        sys.exit(1)

    print(
        f"✅  All {len(marginals)} (tid, attr) entries look good. "
        "Structure matches {(tid, attr): {candidate: prob}}."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
