"""Aggregate TopicGPT seed runs (paper/data/topicgpt_out/{corpus}_seed*) into
mean ± stddev rows ready for inclusion in eval_results.tex Tables 1 and 2."""
from __future__ import annotations

import glob
import json
import math
import sys


def aggregate(corpus: str) -> dict:
    seeds = sorted(glob.glob(f"paper/data/topicgpt_out/{corpus}_seed*/metrics.json"))
    rows = [json.load(open(p)) for p in seeds]
    if not rows:
        return {}
    out = {"corpus": corpus, "n_seeds": len(rows)}
    for k in ("purity", "nmi", "ari", "npmi", "c_v", "redundancy",
             "n_categories", "wall_s"):
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        out[k] = {"mean": mean, "std": std, "vals": vals}
    return out


def main():
    for corpus in ("20ng", "cot"):
        agg = aggregate(corpus)
        if not agg:
            print(f"=== {corpus}: no runs ===")
            continue
        print(f"=== {corpus} ({agg['n_seeds']} seeds) ===")
        for k, v in agg.items():
            if isinstance(v, dict) and "mean" in v:
                print(f"  {k:14s} {v['mean']:.4f} ± {v['std']:.4f}")
        print()


if __name__ == "__main__":
    main()
