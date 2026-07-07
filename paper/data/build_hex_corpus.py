"""Convert Nanda et al. 2026 hex rollouts into the eval-runner shape.

Source: https://github.com/Centrattic/global-cot-traces (hex/ subtree).

Each rollout is a 1-shot gpt-oss-20b reasoning trace on the hex-to-binary
counting prompt. Nanda's team annotated the corpus with two algorithmic
strategies via a Gemini-built cue dictionary:

    0 = convert base-16 to decimal, then find binary representation
    1 = each hex digit is 4 bits (so 5-digit hex = up to 20 bits)

We use ``single_algorithm`` as the gold label. Rollouts with
``single_algorithm=None`` are dropped.

Output: eval_data/cot_strategy_hex.jsonl with fields
  id, text, gold_label (int), gold_label_name (str), source_id, correctness.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


STRATEGY_NAMES = {
    "0": "hex_to_decimal_then_binary",
    "1": "hex_digit_equals_four_bits",
}


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


def build(src_dir: Path, out_path: Path, model_name: str,
          max_trace_chars: int) -> dict:
    rollouts_dir = src_dir / "hex" / model_name / "rollouts"
    files = sorted(rollouts_dir.glob("*.json"))
    kept = []
    dropped = Counter()

    for f in files:
        try:
            d = json.loads(f.read_text())
        except Exception:
            dropped["parse_error"] += 1
            continue

        gold = d.get("single_algorithm")
        if gold is None:
            dropped["no_gold"] += 1
            continue
        gold_str = str(gold)
        if gold_str not in STRATEGY_NAMES:
            dropped["unknown_gold"] += 1
            continue

        cot = d.get("cot_content") or ""
        cot = cot.strip()
        if not cot:
            dropped["empty_cot"] += 1
            continue
        cot = _clip(cot, max_trace_chars)

        response = d.get("response_content")
        source_seed = d.get("seed")
        correctness = d.get("correctness")

        text = f"Reasoning:\n{cot}\n\nFinal answer: {response}"
        kept.append({
            "id": f"hex-{model_name}-{f.stem}",
            "text": text,
            "gold_label": int(gold_str),
            "gold_label_name": STRATEGY_NAMES[gold_str],
            "source_id": f.stem,
            "source_seed": source_seed,
            "model": model_name,
            "correctness": correctness,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    per_class = Counter(r["gold_label_name"] for r in kept)
    return {
        "src_dir": str(src_dir),
        "out_path": str(out_path),
        "model": model_name,
        "n_rollouts_seen": len(files),
        "n_kept": len(kept),
        "dropped": dict(dropped),
        "per_class": dict(per_class),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/tmp/gct")
    p.add_argument("--out", default="eval_data/cot_strategy_hex.jsonl")
    p.add_argument("--model", default="gpt-oss-20b")
    p.add_argument("--max-trace-chars", type=int, default=6000)
    args = p.parse_args(argv)

    summary = build(Path(args.src), Path(args.out), args.model,
                    args.max_trace_chars)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
