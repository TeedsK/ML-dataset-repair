#!/usr/bin/env python
"""
check_marginals.py  —  sanity-check *and analyse* a HoloClean-style
`inference_marginals.pkl`.

Expected structure
------------------
    {
        (tid, attr) : {candidate_value : probability_float, ...},
        ...
    }

Two stages are now available:

1. **Structure checks** (unchanged from the original script):
   - top-level object is a dict;
   - keys are 2-tuples (tid, attr);
   - inner dict maps hashable candidate values → floats in [0, 1];
   - optional check that probs sum to 1 ± TOL.

2. **Value-level analysis** (new, opt-in via CLI flags):
   - summary statistics of posterior confidence
   - list of the *N* least-certain cells
   - (optional) entropy statistics

Run `python check_marginals.py --help` for full usage.
"""

from __future__ import annotations

import argparse
import math
import pickle
import statistics as stats
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Hashable, Tuple

TOLERANCE = 1e-5  # prob-sum tolerance


# --------------------------------------------------------------------------- #
# I/O utilities
# --------------------------------------------------------------------------- #
def load_pickle(path: Path):
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        sys.exit(f"[ERROR] Failed to load {path}: {exc}")


# --------------------------------------------------------------------------- #
# Structural checks (mostly unchanged)
# --------------------------------------------------------------------------- #
def check_structure(
    marginals: Dict[Tuple[int, str], Dict[Hashable, float]], do_sum_check: bool = True
) -> list[str]:
    """Return a list of human-readable error messages (empty ⇒ all good)."""
    errors: list[str] = []

    if not isinstance(marginals, dict):
        return [f"Top-level object is {type(marginals)}, expected dict"]

    for key, inner in marginals.items():
        # ---- key
        if not isinstance(key, tuple) or len(key) != 2:
            errors.append(f"Key {key!r} is not a 2-tuple (tid, attr)")
            continue
        _, attr = key
        if not isinstance(attr, str):
            errors.append(f"attr in key {key!r} is not str (got {type(attr)})")

        # ---- value
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
            elif prob < 0.0 or prob > 1.0:
                errors.append(
                    f"Prob {prob} for candidate {cand!r} in {key!r} "
                    f"is out of range [0, 1]"
                )
            prob_sum += float(prob)

        if do_sum_check and not math.isclose(prob_sum, 1.0, abs_tol=TOLERANCE):
            errors.append(
                f"Probabilities for {key!r} sum to {prob_sum:.6f}, "
                f"expected 1.0 ± {TOLERANCE}"
            )

    return errors


# --------------------------------------------------------------------------- #
# Value-level analysis
# --------------------------------------------------------------------------- #
def entropy(dist: Dict[Hashable, float]) -> float:
    """Shannon entropy, log-base *e* (nats)."""
    return -sum(p * math.log(p) for p in dist.values() if p > 0.0)


def analyse_values(
    marginals: Dict[Tuple[int, str], Dict[Hashable, float]],
    threshold: float = 0.5,
    include_entropy: bool = False,
):
    """
    Return a dict with summary stats:
        {
            "n_cells": int,
            "mean_top_prob": float,
            "median_top_prob": float,
            "min_top_prob": float,
            "max_top_prob": float,
            "n_below_threshold": int,
            "threshold": float,
            "entropy_stats": { ... }  # optional
        }
    """
    top_probs = []
    entropies = []

    for dist in marginals.values():
        # highest posterior prob
        p_max = max(dist.values())
        top_probs.append(p_max)

        if include_entropy:
            entropies.append(entropy(dist))

    summary = {
        "n_cells": len(marginals),
        "mean_top_prob": stats.fmean(top_probs),
        "median_top_prob": stats.median(top_probs),
        "min_top_prob": min(top_probs),
        "max_top_prob": max(top_probs),
        "n_below_threshold": sum(p < threshold for p in top_probs),
        "threshold": threshold,
    }

    if include_entropy:
        summary["entropy_stats"] = {
            "mean_entropy": stats.fmean(entropies),
            "median_entropy": stats.median(entropies),
            "min_entropy": min(entropies),
            "max_entropy": max(entropies),
        }

    return summary, top_probs


def list_most_uncertain(
    marginals: Dict[Tuple[int, str], Dict[Hashable, float]], n: int
):
    """
    Yield the *n* cells whose *max* probability is the smallest
    (ties broken arbitrarily).
    """
    scored = [
        (max(dist.values()), key, dist) for key, dist in marginals.items()
    ]  # (p_max, (tid, attr), dist)
    scored.sort(key=lambda x: x[0])  # ascending ⇒ most uncertain first
    return scored[:n]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Validate *and* analyse a HoloClean-style inference_marginals.pkl."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default="inference_marginals.pkl",
        help="Path to the pickle file (default: inference_marginals.pkl)",
    )
    parser.add_argument(
        "--no-sum-check",
        action="store_true",
        help="Skip the per-cell prob-sum-to-1 test",
    )
    # New analysis flags
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics of posterior confidence",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        metavar="T",
        help="Threshold T for counting low-confidence cells (default: 0.50)",
    )
    parser.add_argument(
        "--entropy",
        action="store_true",
        help="Include entropy computation in the summary",
    )
    parser.add_argument(
        "--list-uncertain",
        type=int,
        metavar="N",
        help="List the N most uncertain cells (lowest max-prob)",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    marginals = load_pickle(args.path)

    errors = check_structure(marginals, do_sum_check=not args.no_sum_check)
    if errors:
        print("\n❌  STRUCTURE ERRORS DETECTED:")
        for msg in errors:
            print("   •", msg)
        print(f"\nFound {len(errors)} issue(s).")
        sys.exit(1)
    else:
        print(
            f"✅  Structure OK — {len(marginals)} (tid, attr) entries conform "
            "to {(tid, attr): {candidate: prob}}."
        )

    # ----- Optional value-level analysis
    if args.summary or args.list_uncertain is not None:
        print("\n🔍  ANALYSING POSTERIOR VALUES …")

    if args.summary:
        summary, _ = analyse_values(
            marginals, threshold=args.threshold, include_entropy=args.entropy
        )
        print("  • Cells analysed :", summary["n_cells"])
        print("  • Mean max-prob  :", f"{summary['mean_top_prob']:.4f}")
        print("  • Median max-prob:", f"{summary['median_top_prob']:.4f}")
        print("  • Min / Max      :", f"{summary['min_top_prob']:.4f}"
              f" / {summary['max_top_prob']:.4f}")
        print(
            "  • Low-confidence :", f"{summary['n_below_threshold']} "
            f"(max-prob < {summary['threshold']})"
        )
        if args.entropy:
            ent = summary["entropy_stats"]
            print("  • Mean entropy   :", f"{ent['mean_entropy']:.4f}")
            print("  • Median entropy :", f"{ent['median_entropy']:.4f}")
            print("  • Min / Max ent. :", f"{ent['min_entropy']:.4f}"
                  f" / {ent['max_entropy']:.4f}")

    if args.list_uncertain is not None:
        n = args.list_uncertain
        print(f"\n  • {n} MOST UNCERTAIN CELLS (lowest max-prob):")
        for rank, (p_max, key, dist) in enumerate(
            list_most_uncertain(marginals, n), start=1
        ):
            tid, attr = key
            # sort candidates by prob desc for readability
            top_cands = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:5]
            top_repr = ", ".join(f"{c!r}:{p:.3f}" for c, p in top_cands)
            print(
                f"    {rank:>2}. (tid={tid}, attr='{attr}') "
                f"max-prob={p_max:.3f} → top candidates: {top_repr}"
            )

    # clean exit
    sys.exit(0)


if __name__ == "__main__":
    main()
