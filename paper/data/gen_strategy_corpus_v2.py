"""Controlled solution-strategy corpus, v2: few-shot induction, no leakage.

The method is induced purely by demonstration -- two worked examples in the
target method on held-out problems, then the target problem -- with a generic
system prompt that says nothing about method. There is no instruction for the
model to narrate, so traces do not leak the steering.

One problem family (two positive whole numbers with a given sum and product,
find the larger), numbers varied, so strategy is the only signal and the
exemplars match the targets exactly.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import threading
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM = ("Solve the math problem. Show your working, then give the final "
          "answer.")

USER_TMPL = ("Two positive whole numbers have a sum of {s} and a product of "
             "{p}. What is the larger number?")

# Few-shot exemplars per strategy, on two held-out (sum, product) pairs:
# (10, 21) -> 3 and 7 ; (12, 32) -> 4 and 8. Natural, no method names, no
# meta-commentary, nothing to echo.
EXEMPLARS: dict[str, list[tuple[str, str]]] = {
    "algebraic": [
        (USER_TMPL.format(s=10, p=21),
         "Let the numbers be x and y with x + y = 10 and xy = 21. "
         "Substituting y = 10 - x gives x(10 - x) = 21, so x^2 - 10x + 21 = 0. "
         "Factoring, (x - 3)(x - 7) = 0, so the numbers are 3 and 7. "
         "The larger number is 7."),
        (USER_TMPL.format(s=12, p=32),
         "Call the numbers a and b, so a + b = 12 and ab = 32. They are the "
         "roots of t^2 - 12t + 32 = 0, which factors as (t - 4)(t - 8) = 0. "
         "So the numbers are 4 and 8, and the larger is 8."),
    ],
    "enumerative": [
        (USER_TMPL.format(s=10, p=21),
         "List the pairs of positive whole numbers that add to 10: (1,9), "
         "(2,8), (3,7), (4,6), (5,5). Their products are 9, 16, 21, 24, 25. "
         "The product 21 comes from (3,7), so the larger number is 7."),
        (USER_TMPL.format(s=12, p=32),
         "Pairs adding to 12: (1,11), (2,10), (3,9), (4,8), (5,7), (6,6). "
         "The products are 11, 20, 27, 32, 35, 36. The product 32 is from "
         "(4,8), so the larger number is 8."),
    ],
    "arithmetic_insight": [
        (USER_TMPL.format(s=10, p=21),
         "21 factors as 3 times 7, and 3 + 7 = 10, which is the required sum. "
         "So the larger number is 7."),
        (USER_TMPL.format(s=12, p=32),
         "32 = 4 times 8, and 4 + 8 = 12 matches the sum, so the larger "
         "number is 8."),
    ],
}

# (sum, product) pairs with a unique positive-integer factorisation, varied in
# magnitude; excludes the two exemplar pairs (10,21) and (12,32).
PAIRS = [
    (15, 54), (20, 91), (17, 72), (24, 143), (13, 40), (21, 108), (19, 84),
    (26, 168), (11, 28), (14, 45), (16, 63), (18, 77), (22, 117), (23, 132),
    (25, 156), (27, 176), (28, 195), (29, 208), (30, 221), (31, 240),
    (33, 272), (34, 288), (35, 306), (36, 323),
]


def call(model, messages, api_key, max_tokens, timeout):
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": 0.8}
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
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


def build_messages(strategy, s, p):
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, a in EXEMPLARS[strategy]:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": USER_TMPL.format(s=s, p=p)})
    return msgs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="eval_data/strategy_corpus_v2_raw.jsonl")
    p.add_argument("--model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--samples", type=int, default=3,
                   help="samples per (pair, strategy)")
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--timeout", type=int, default=150)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--limit-pairs", type=int, default=None)
    args = p.parse_args(argv)

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set")
        return 2

    pairs = PAIRS[:args.limit_pairs] if args.limit_pairs else PAIRS
    jobs = []
    for (s, prod) in pairs:
        for strat in EXEMPLARS:
            for k in range(args.samples):
                jobs.append((s, prod, strat, k))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    fp = out.open("w")
    ctr = {"in": 0, "out": 0, "done": 0, "err": 0}

    def work(job):
        s, prod, strat, k = job
        msgs = build_messages(strat, s, prod)
        try:
            resp = call(args.model, msgs, api_key, args.max_tokens, args.timeout)
            m = resp["choices"][0]["message"]
            trace = extract_trace(m)
            u = resp.get("usage", {})
            q = USER_TMPL.format(s=s, p=prod)
            rec = {"id": f"stratv2-{s}-{prod}-{strat}-{k}",
                   "sum": s, "product": prod, "strategy": strat, "sample": k,
                   "text": f"Question: {q}\n\n{trace}",
                   "raw_content": trace, "model": args.model}
            with lock:
                fp.write(json.dumps(rec) + "\n"); fp.flush()
                ctr["in"] += u.get("prompt_tokens", 0)
                ctr["out"] += u.get("completion_tokens", 0)
                ctr["done"] += 1
        except Exception as e:
            with lock:
                ctr["err"] += 1
            print(f"ERR {s},{prod} {strat}: {e!r}")

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(work, jobs))
    fp.close()

    cost = ctr["in"]/1e6*0.30 + ctr["out"]/1e6*1.20
    print(json.dumps({"out": str(out), "n_jobs": len(jobs), "done": ctr["done"],
                      "err": ctr["err"], "prompt_tokens": ctr["in"],
                      "completion_tokens": ctr["out"],
                      "rough_cost_usd": round(cost, 4)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
