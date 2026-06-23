"""Compute mean+stddev across the cheap-config seeds.

Combines:
- 20NG: seeds 42,43,44 (first run) + 45,46 (this run) = 5 seeds total
- CoT: seed 42 (first run) + 43,44 (this run) = 3 seeds total
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path


def load_rows(results_json_path: Path) -> list[dict]:
    with open(results_json_path) as f:
        return [r for r in json.load(f)["rows"]
                if r.get("error") is None and r.get("purity") is not None]


def summarise(rows: list[dict], label: str, ks: tuple = ("purity", "nmi", "ari", "npmi", "c_v", "redundancy", "cost_usd", "wall_time_s", "n_categories")) -> None:
    print(f"\n=== {label} (n={len(rows)} seeds: {sorted([r['seed'] for r in rows])}) ===")
    for k in ks:
        vals = [r[k] for r in rows]
        mean = statistics.mean(vals)
        if len(vals) > 1:
            stdev = statistics.stdev(vals)
            print(f"  {k:15s}: mean={mean:.4f}  stdev={stdev:.4f}")
        else:
            print(f"  {k:15s}: {mean:.4f}")


def per_pattern_recovery(classifications_paths: list[Path]) -> dict:
    from collections import Counter
    totals = Counter()
    labelled = Counter()
    for cp in classifications_paths:
        lines = [json.loads(l) for l in cp.read_text().splitlines() if l.strip()]
        for ln in lines:
            totals[ln["gold_label_name"]] += 1
            if ln.get("category") != "other":
                labelled[ln["gold_label_name"]] += 1
    return {g: (labelled[g], totals[g]) for g in sorted(totals)}


if __name__ == "__main__":
    base = Path("/mnt/hdd/qmnguyen/taxonomy_agent/eval_runs")
    cot_a = base / sys.argv[1]
    cot_b = base / sys.argv[2]
    twenty_a = base / sys.argv[3]
    twenty_b = base / sys.argv[4]

    cot_rows = load_rows(cot_a / "results.json") + load_rows(cot_b / "results.json")
    twenty_rows = load_rows(twenty_a / "results.json") + load_rows(twenty_b / "results.json")
    summarise(twenty_rows, "20NG cheap config (combined)")
    summarise(cot_rows, "CoT cheap config (combined)")

    cot_cls = list((cot_a).glob("taxonomy_agent_seed*/classifications.jsonl")) \
            + list((cot_b).glob("taxonomy_agent_seed*/classifications.jsonl"))
    print(f"\n=== CoT per-pattern recovery across {len(cot_cls)} seeds ===")
    rec = per_pattern_recovery(cot_cls)
    for g, (l, t) in rec.items():
        print(f"  {g:30s} {l:4d} / {t:4d}  ({100*l/t:5.1f}%)")
