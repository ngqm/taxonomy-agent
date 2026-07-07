"""Generate the advice/decision-prompts source for the CoT-patterns corpus.

Run separately from the main corpus build so the advice prompts come from a
DIFFERENT model than the CoT generator — that's the model-diversity bit.

Example:
    python -m taxonomy_agent.eval.generate_advice_prompts \\
        --out eval_data/advice_prompts.jsonl \\
        --n 30 \\
        --model anthropic/claude-sonnet-4.6 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from .synth_reasoning import DEFAULT_ADVICE_PATH, _call_llm, extract_json


_TOPICS = ["finance", "career", "health", "learning",
           "relationships", "time-management"]

_SYS = (
    "You write everyday-life decision prompts for a reasoning benchmark. "
    "Each prompt must admit a thoughtful answer that benefits from "
    "step-by-step reasoning. Prompts should sound like something a real "
    "person would type."
)


def _batch_prompt(topic: str, n: int, seed: int) -> str:
    return (
        f"Topic: {topic}\n"
        f"Variation seed: {seed}\n\n"
        f"Generate exactly {n} distinct advice/decision prompts a real "
        f"person might ask on the topic above. Each prompt must:\n"
        f"  - be 1-2 sentences, first-person where natural;\n"
        f"  - present a real decision or dilemma, not a factual question;\n"
        f"  - admit step-by-step reasoning (tradeoffs, comparisons, "
        f"sequencing);\n"
        f"  - feel concrete (specific context, numbers, or constraints).\n\n"
        f"Output a single JSON object: "
        f"{{\"prompts\": [\"<prompt 1>\", \"<prompt 2>\", ...]}}. "
        f"No prose outside the JSON."
    )


def _parse_batch(text: str) -> list[str]:
    obj = extract_json(text) or {}
    prompts = obj.get("prompts")
    if not isinstance(prompts, list):
        return []
    out: list[str] = []
    for p in prompts:
        if isinstance(p, str):
            p = p.strip()
            if p:
                out.append(p)
    return out


def generate_advice_prompts(out_path: str, n: int, model: str,
                            api_key: str, base_url: str,
                            seed: int) -> dict:
    per_topic = max(1, n // len(_TOPICS))
    # Round up so we can trim to n at the end.
    leftover = n - per_topic * len(_TOPICS)

    all_prompts: list[str] = []
    total_cost = 0.0
    rng_seed = seed
    for i, topic in enumerate(_TOPICS):
        want = per_topic + (1 if i < leftover else 0)
        if want <= 0:
            continue
        got_for_topic: list[str] = []
        attempts = 0
        # Ask for up to 5-per-batch to amortize prompt overhead, as spec'd.
        while len(got_for_topic) < want and attempts < 3:
            attempts += 1
            ask = min(5, want - len(got_for_topic))
            user = _batch_prompt(topic, ask, rng_seed)
            rng_seed += 1
            out = _call_llm(
                model=model,
                messages=[{"role": "system", "content": _SYS},
                          {"role": "user", "content": user}],
                api_key=api_key, base_url=base_url,
                temperature=0.9, max_tokens=900,
            )
            total_cost += out["cost"]
            batch = _parse_batch(out["text"])
            if not batch:
                print(f"[advice] {topic}: empty/parse-fail "
                      f"(attempt {attempts})")
                continue
            got_for_topic.extend(batch[: want - len(got_for_topic)])
        all_prompts.extend(got_for_topic)
        print(f"[advice] {topic}: kept {len(got_for_topic)}/{want}")

    # Trim / pad to exactly n.
    all_prompts = all_prompts[:n]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for idx, prompt in enumerate(all_prompts):
            rec = {
                "id": f"advice_{idx:02d}",
                "problem": prompt,
                "gold_answer": None,
                "source": "advice_custom",
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[advice] wrote {len(all_prompts)} prompts to {out_path}; "
          f"cost=${total_cost:.4f}")
    return {"n_prompts": len(all_prompts), "cost": total_cost,
            "out_path": out_path}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="taxonomy_agent.eval.generate_advice_prompts")
    p.add_argument("--out", default=DEFAULT_ADVICE_PATH,
                   help="output JSONL path")
    p.add_argument("--n", type=int, default=30,
                   help="total prompts to generate")
    p.add_argument("--model", default="anthropic/claude-sonnet-4.6",
                   help="OpenRouter model id for the advice generator")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    args = p.parse_args(argv)

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set (env or .env). Aborting.",
              file=sys.stderr)
        return 2

    try:
        generate_advice_prompts(
            out_path=args.out, n=args.n, model=args.model,
            api_key=api_key, base_url=args.base_url, seed=args.seed,
        )
    except Exception as e:
        print(f"generate_advice_prompts failed: {e!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
