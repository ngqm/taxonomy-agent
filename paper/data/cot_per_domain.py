"""Compute per-domain metrics on the CoT-Pattern taxonomy_agent run.

The 149-item CoT-Pattern corpus is single-domain (competition math) at the
top level, but the individual problems span algebra, geometry, combinatorics,
number theory, and a residual ``other`` slice. We tag each item with a
problem-domain heuristic from the problem text (before the chain-of-thought
begins) and report per-domain purity / NMI / ARI for the existing
taxonomy_agent seed 42 run.

The point is to convert the single-domain weakness of the synthetic corpus
into a sub-domain generalisation story: if the system holds purity / NMI
across the four mathematical sub-domains, that is evidence the
discovered taxonomy generalises beyond a single sub-area.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, "/mnt/hdd/qmnguyen/taxonomy_agent")
from taxonomy_agent.eval.metrics import (
    purity,
    normalized_mutual_info,
    adjusted_rand_index,
)


_GEOMETRY_KW = ["triangle", "circle", "angle", "perpendicular", "parallel",
                "rectangle", "hexagon", "polygon", "tangent", "radius",
                "diameter", "cone", "sphere", "pyramid", "vertex", "vertices",
                "cube", "altitude", "hypotenuse", "inscribed", "circumscribed"]
_COMBI_KW = ["probability", "permutation", "combination", "how many ways",
             "how many distinct", "choose", "dice", "coin", "roll", "draw",
             "arrangement", "distinct ", "lattice", "pigeon"]
_NUMTH_KW = ["gcd", "divisor", "prime", "modulo", "mod ", "divisible",
             "remainder", "congruent", "multiple of ", "units digit",
             "tens digit", "base ", "digits of "]
_ALG_KW = ["equation", "polynomial", "root", "solve for", "simplify",
           "evaluate", "expand", "factor", "function ", "function$",
           "$f(", "$g(", "log", "sqrt", "sum", "product", "compute"]


def domain_of(item: dict) -> str:
    problem = item["text"].split("Reasoning:", 1)[0].lower()
    if any(k in problem for k in _GEOMETRY_KW):
        return "geometry"
    if any(k in problem for k in _COMBI_KW):
        return "combinatorics"
    if any(k in problem for k in _NUMTH_KW):
        return "number_theory"
    if any(k in problem for k in _ALG_KW):
        return "algebra"
    return "other"


def main():
    corpus_path = Path("/mnt/hdd/qmnguyen/taxonomy_agent/eval_data/cot_patterns.jsonl")
    items = [json.loads(line) for line in corpus_path.read_text().splitlines()
             if line.strip()]
    id_to_dom = {it["id"]: domain_of(it) for it in items}
    id_to_gold = {it["id"]: it["gold_label_name"] for it in items}

    # Write the annotated corpus so callers can reuse the partition.
    annotated_path = corpus_path.with_name("cot_patterns_with_domain.jsonl")
    with open(annotated_path, "w") as f:
        for it in items:
            f.write(json.dumps({**it, "domain": id_to_dom[it["id"]]}) + "\n")

    art_path = Path(
        "/mnt/hdd/qmnguyen/taxonomy_agent/eval_runs/"
        "cot_clean_20260622_060836/taxonomy_agent_seed42/taxonomy.json"
    )
    art = json.loads(art_path.read_text())
    id_to_pred = {c["id"]: c.get("category") for c in art.get("classifications", [])}

    results: dict[str, dict] = {}
    for dom in ("algebra", "geometry", "other", "combinatorics", "number_theory"):
        ids = [i for i, d in id_to_dom.items() if d == dom]
        if len(ids) < 5:
            continue
        pred = [id_to_pred.get(i, "other") for i in ids]
        gold = [id_to_gold[i] for i in ids]
        labelled = [(p, g) for p, g in zip(pred, gold) if p != "other"]
        precision_at_coverage = (
            sum(1 for p, g in labelled if p == g) / len(labelled)
            if labelled else 0.0
        )
        results[dom] = {
            "n": len(ids),
            "coverage": sum(1 for p in pred if p != "other") / len(ids),
            "precision_at_coverage": precision_at_coverage,
            "purity": purity(pred, gold),
            "nmi": normalized_mutual_info(pred, gold),
            "ari": adjusted_rand_index(pred, gold),
            "n_labelled_correct": sum(1 for p, g in labelled if p == g),
            "n_labelled_total": len(labelled),
        }

    out_path = Path("/mnt/hdd/qmnguyen/taxonomy_agent/paper/data/cot_per_domain.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"wrote {out_path}")
    print(f"wrote {annotated_path}")


if __name__ == "__main__":
    main()
