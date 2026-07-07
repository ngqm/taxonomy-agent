"""Full controlled solution-strategy corpus.

Steer the solution method via a hidden system prompt on "find the value under
constraints" problems that admit three genuinely distinct methods. The trace is
natural (deepseek-v4-flash reasoning), the label is what we instructed.

Strategies:
  algebraic          : variables + equations, solved symbolically
  enumerative        : list candidate values/pairs and check each
  arithmetic_insight : use a number fact / factoring / identity as a shortcut

A separate verify pass (verify_strategy_corpus.py) has an independent model read
each trace blind and label the method; we keep only traces where the blind label
matches the instructed one, so gold is clean and we can report agreement.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import threading
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

_NO_LEAK = (
    " Work naturally in this way. Do not say you were told to use a particular "
    "method and do not mention these directions; just solve it and give the "
    "final answer."
)

STRATEGIES: dict[str, str] = {
    "algebraic": (
        "Solve this problem by introducing variables, writing equations that "
        "capture the stated conditions, and manipulating those equations "
        "symbolically to isolate the answer." + _NO_LEAK
    ),
    "enumerative": (
        "Solve this problem by listing out the candidate values or combinations "
        "that could satisfy the conditions and checking them one by one until "
        "you find the one that works. Do not set up and solve equations "
        "symbolically." + _NO_LEAK
    ),
    "arithmetic_insight": (
        "Solve this problem by spotting a number fact, factoring, or shortcut "
        "that gets the answer with little computation (for example, factor a "
        "product and see which factor pair fits). Do not list out many "
        "candidates and do not set up and solve equations symbolically."
        + _NO_LEAK
    ),
}

USER_TEMPLATE = (
    "{question}\n\nThink through it step by step, then give the final answer."
)

# "Find the value under two constraints" problems. Each admits all three
# methods. Kept to one structural family so topic carries no strategy signal;
# numbers vary so it is not a single memorised instance.
PROBLEMS = [
    "Two positive whole numbers have a sum of 15 and a product of 54. What is the larger number?",
    "Two positive whole numbers have a sum of 20 and a product of 91. What is the larger number?",
    "Two positive whole numbers have a sum of 17 and a product of 72. What is the larger number?",
    "Two positive whole numbers have a sum of 24 and a product of 143. What is the larger number?",
    "Two positive whole numbers have a sum of 13 and a product of 40. What is the larger number?",
    "Two positive whole numbers have a sum of 21 and a product of 108. What is the larger number?",
    "Two positive whole numbers have a sum of 19 and a product of 84. What is the larger number?",
    "Two positive whole numbers have a sum of 26 and a product of 168. What is the larger number?",
    "You have 10 coins totalling 55 cents, using only pennies, nickels, and dimes. How many dimes are there? (Unique solution.)",
    "You have 12 coins totalling 90 cents, using only pennies, nickels, and dimes. How many dimes are there? (Unique solution.)",
    "Adult tickets cost $8 and child tickets cost $5. A group buys 20 tickets for $136. How many adult tickets?",
    "Adult tickets cost $9 and child tickets cost $4. A group buys 15 tickets for $100. How many adult tickets?",
    "One number is three times another and their sum is 48. What is the larger number?",
    "One number is four times another and their sum is 65. What is the larger number?",
    "A rectangle has perimeter 34 and area 60. What is its longer side?",
]


def call(model, system, user, api_key, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.8,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ngqm/taxonomy_agent",
        "X-Title": "taxonomy_agent strategy corpus",
    }
    r = requests.post(BASE_URL, json=payload, headers=headers, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def extract_trace(msg):
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
    if reasoning:
        return (reasoning + "\n\n" + content).strip() if content else reasoning
    return content


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="eval_data/strategy_corpus_raw.jsonl")
    p.add_argument("--model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--samples", type=int, default=5,
                   help="samples per (problem, strategy)")
    p.add_argument("--max-tokens", type=int, default=2500)
    p.add_argument("--timeout", type=int, default=150)
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args(argv)

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set")
        return 2

    jobs = []
    for qi, q in enumerate(PROBLEMS):
        for strat, system in STRATEGIES.items():
            for s in range(args.samples):
                jobs.append((qi, q, strat, system, s))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    fp = out.open("w")
    counters = {"in": 0, "out": 0, "done": 0, "err": 0}

    def work(job):
        qi, q, strat, system, s = job
        user = USER_TEMPLATE.format(question=q)
        try:
            resp = call(args.model, system, user, api_key,
                        args.max_tokens, args.timeout)
            msg = resp["choices"][0]["message"]
            trace = extract_trace(msg)
            usage = resp.get("usage", {})
            rec = {
                "id": f"strat-q{qi:02d}-{strat}-{s}",
                "qi": qi,
                "question": q,
                "strategy": strat,
                "sample": s,
                "text": f"Question: {q}\n\n{trace}",
                "raw_content": trace,
                "model": args.model,
            }
            with lock:
                fp.write(json.dumps(rec) + "\n")
                fp.flush()
                counters["in"] += usage.get("prompt_tokens", 0)
                counters["out"] += usage.get("completion_tokens", 0)
                counters["done"] += 1
                if counters["done"] % 20 == 0:
                    print(f"  {counters['done']}/{len(jobs)}")
        except Exception as e:
            with lock:
                counters["err"] += 1
            print(f"ERR q{qi} {strat}: {e!r}")

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(work, jobs))
    fp.close()

    cost = (counters["in"] / 1e6) * 0.30 + (counters["out"] / 1e6) * 1.20
    print(json.dumps({
        "out": str(out),
        "n_jobs": len(jobs),
        "done": counters["done"],
        "err": counters["err"],
        "prompt_tokens": counters["in"],
        "completion_tokens": counters["out"],
        "rough_cost_usd": round(cost, 4),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
