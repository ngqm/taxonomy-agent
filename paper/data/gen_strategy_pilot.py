"""Pilot generator for the controlled solution-strategy corpus.

We steer the *solution method* through a hidden system prompt, on problems that
genuinely admit several methods, and let the method show up naturally in the
trace. The analyst-facing item is question + trace only.

Three general strategies that all apply to "find the value under constraints"
problems, so the design can be crossed (every problem under every strategy):

  algebraic       : introduce variables, write equations, solve symbolically
  guess_and_check : try candidate values, check, adjust until one fits
  enumeration     : list the possibilities systematically, pick the one that fits

Gold = the instructed strategy. Forcing a method does not require the model to
produce wrong content, so it complies without the leakage/refusal seen for
"make a mistake"-style dispositions.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

_NO_LEAK = (
    " Work naturally in this way. Do not state that you were told to use a "
    "particular method and do not mention these directions; just solve the "
    "problem and give the final answer."
)

STRATEGIES: dict[str, str] = {
    "algebraic": (
        "Solve this problem by introducing variables, writing equations that "
        "capture the stated conditions, and manipulating those equations "
        "symbolically to isolate the answer." + _NO_LEAK
    ),
    "guess_and_check": (
        "Solve this problem by trying specific candidate values, checking each "
        "one against the stated conditions, and adjusting your guess until one "
        "fits. Do not set up and solve equations symbolically." + _NO_LEAK
    ),
    "enumeration": (
        "Solve this problem by systematically listing out the possibilities that "
        "could satisfy the conditions and identifying which one works. Do not "
        "set up and solve equations symbolically." + _NO_LEAK
    ),
}

USER_TEMPLATE = (
    "{question}\n\nThink through it step by step, then give the final answer."
)

# "Find the value under constraints" problems: each admits algebraic,
# guess-and-check, and enumeration approaches.
PILOT_PROBLEMS = [
    {
        "pid": "sum_product",
        "question": (
            "Two positive whole numbers have a sum of 15 and a product of 54. "
            "What is the larger of the two numbers?"
        ),
    },
    {
        "pid": "coins",
        "question": (
            "You have 10 coins totalling 55 cents, using only pennies (1c), "
            "nickels (5c), and dimes (10c). How many dimes do you have? "
            "(Assume the unique solution.)"
        ),
    },
    {
        "pid": "tickets",
        "question": (
            "Adult tickets cost $8 and child tickets cost $5. A group buys 20 "
            "tickets for a total of $136. How many adult tickets did they buy?"
        ),
    },
    {
        "pid": "two_numbers",
        "question": (
            "One number is three times another. Their sum is 48. What is the "
            "larger number?"
        ),
    },
]


def call(model: str, system: str, user: str, api_key: str,
         max_tokens: int, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ngqm/taxonomy_agent",
        "X-Title": "taxonomy_agent strategy pilot",
    }
    r = requests.post(BASE_URL, json=payload, headers=headers, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def extract_trace(msg: dict) -> str:
    content = (msg.get("content") or "").strip()
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    reasoning = reasoning.strip()
    if reasoning:
        return (reasoning + "\n\n" + content).strip() if content else reasoning
    return content


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="eval_data/strategy_pilot.jsonl")
    p.add_argument("--model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--timeout", type=int, default=150)
    args = p.parse_args(argv)

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set")
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    total_in = total_out = 0
    for prob in PILOT_PROBLEMS:
        for strat, system in STRATEGIES.items():
            for s in range(args.samples):
                user = USER_TEMPLATE.format(question=prob["question"])
                try:
                    resp = call(args.model, system, user, api_key,
                                args.max_tokens, args.timeout)
                    msg = resp["choices"][0]["message"]
                    trace = extract_trace(msg)
                    usage = resp.get("usage", {})
                    total_in += usage.get("prompt_tokens", 0)
                    total_out += usage.get("completion_tokens", 0)
                    records.append({
                        "id": f"{prob['pid']}-{strat}-{s}",
                        "pid": prob["pid"],
                        "question": prob["question"],
                        "strategy": strat,
                        "sample": s,
                        "text": f"Question: {prob['question']}\n\n{trace}",
                        "raw_content": trace,
                        "model": args.model,
                    })
                    print(f"ok  {prob['pid']:<12} {strat}")
                except Exception as e:
                    print(f"ERR {prob['pid']:<12} {strat}: {e!r}")
                time.sleep(0.2)

    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cost = (total_in / 1e6) * 0.30 + (total_out / 1e6) * 1.20
    print(json.dumps({
        "out": str(out),
        "n": len(records),
        "prompt_tokens": total_in,
        "completion_tokens": total_out,
        "rough_cost_usd": round(cost, 4),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
