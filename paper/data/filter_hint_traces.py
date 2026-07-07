"""Turn raw R1 hint-trace outputs into eval_data/cot_patterns_v2.jsonl.

Keeps only items where the model was successfully steered (final_letter ==
hinted_letter), and formats each into the eval-runner shape (id, text,
gold_label, gold_label_name). The item text is the FULL transcript that an
alignment analyst would see: the injected-hint prompt as delivered to the
model, followed by the reasoning trace. This matches the realistic analyst
workflow (reading conversation transcripts) rather than the monitor-only view.

Format:
    User:
    <prompt with injected hint>

    Model:
    <trace>

    Final answer: <letter>

The gold_label is the hint type (0..5), matching HINT_TYPES.

Usage:
    python paper/data/filter_hint_traces.py \\
        --in eval_data/hint_traces_raw.jsonl \\
        --out eval_data/cot_patterns_v2.jsonl \\
        --min-per-hint 20
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


HINT_TYPES = (
    "sycophancy",
    "consistency",
    "visual_pattern",
    "metadata",
    "grader_hacking",
    "unethical_info",
)
LABEL_TO_IDX = {h: i for i, h in enumerate(HINT_TYPES)}

LETTERS = ("A", "B", "C", "D")


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


def _build_text(row: dict, max_trace_chars: int) -> str:
    prompt = row["prompt"].strip()
    trace = row.get("trace") or row.get("content") or ""
    trace = trace.strip()
    trace = _clip(trace, max_trace_chars)
    return f"User:\n{prompt}\n\nModel:\n{trace}"


def build(in_path: Path, out_path: Path, max_trace_chars: int,
          min_per_hint: int, cap_per_hint: int | None,
          seed: int) -> dict:
    rng = random.Random(seed)
    rows = []
    with in_path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    steered: dict[str, list[dict]] = defaultdict(list)
    reasons = Counter()

    for r in rows:
        if r.get("error"):
            reasons["error"] += 1
            continue
        hint = r.get("hint_type")
        if hint not in LABEL_TO_IDX:
            reasons["unknown_hint"] += 1
            continue
        final = r.get("final_letter")
        hinted = r.get("hinted_letter")
        if not final:
            reasons["no_final_letter"] += 1
            continue
        if final != hinted:
            reasons["not_steered"] += 1
            continue
        steered[hint].append(r)

    kept = []
    for hint, items in steered.items():
        rng.shuffle(items)
        if cap_per_hint is not None:
            items = items[:cap_per_hint]
        for r in items:
            text = _build_text(r, max_trace_chars)
            kept.append({
                "id": f"cot-v2-{hint}-{r['id']}",
                "text": text,
                "gold_label": LABEL_TO_IDX[hint],
                "gold_label_name": hint,
                "source_id": r["id"],
                "source_benchmark": r.get("source"),
                "correct_letter": r.get("correct_letter"),
                "hinted_letter": r.get("hinted_letter"),
                "final_letter": r.get("final_letter"),
            })

    rng.shuffle(kept)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for item in kept:
            f.write(json.dumps(item) + "\n")

    per_hint = Counter(item["gold_label_name"] for item in kept)
    below_floor = {h: per_hint.get(h, 0) for h in HINT_TYPES
                   if per_hint.get(h, 0) < min_per_hint}

    return {
        "in_path": str(in_path),
        "out_path": str(out_path),
        "raw_rows": len(rows),
        "kept": len(kept),
        "drop_reasons": dict(reasons),
        "per_hint_kept": dict(per_hint),
        "below_floor": below_floor,
        "min_per_hint": min_per_hint,
        "cap_per_hint": cap_per_hint,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp",
                   default="eval_data/hint_traces_raw.jsonl")
    p.add_argument("--out", default="eval_data/cot_patterns_v2.jsonl")
    p.add_argument("--max-trace-chars", type=int, default=6000)
    p.add_argument("--min-per-hint", type=int, default=15)
    p.add_argument("--cap-per-hint", type=int, default=None,
                   help="max items per hint (balance the corpus)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    summary = build(Path(args.inp), Path(args.out), args.max_trace_chars,
                    args.min_per_hint, args.cap_per_hint, args.seed)
    print(json.dumps(summary, indent=2))
    return 0 if not summary["below_floor"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
