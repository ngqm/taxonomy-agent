"""Verify-and-drop pass for the strategy corpus.

An independent model (different vendor from the generator) reads each trace
blind -- question + reasoning only, no strategy label -- and classifies which of
the three methods it used. We keep only traces where the blind label matches the
instructed strategy. This yields clean gold and a reportable agreement rate
(intended vs blind), analogous to inter-annotator agreement.

Output: eval_data/cot_strategy_synth.jsonl in the eval-runner shape
  (id, text, gold_label, gold_label_name), plus per-strategy kept counts.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import threading
from pathlib import Path

import requests
from dotenv import load_dotenv


BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

STRATEGY_NAMES = ["algebraic", "enumerative", "arithmetic_insight"]
LABEL_TO_IDX = {n: i for i, n in enumerate(STRATEGY_NAMES)}

VERIFY_SYSTEM = (
    "You are analysing how a solution to a math word problem was reasoned out. "
    "You will be given the problem and the reasoning that led to the answer. "
    "Classify the SINGLE dominant solution method into exactly one of:\n"
    "- algebraic: introduces variables and sets up and solves equations "
    "symbolically.\n"
    "- enumerative: lists candidate values or combinations and checks them one "
    "by one.\n"
    "- arithmetic_insight: uses a number fact, factoring, or shortcut to get the "
    "answer with little computation, without listing many candidates or solving "
    "equations symbolically.\n"
    "Answer with ONLY the single label on one line."
)


def call(model, system, user, api_key, timeout):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    r = requests.post(BASE_URL, json=payload, headers=headers, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def parse_label(text):
    t = (text or "").lower()
    for name in STRATEGY_NAMES:
        if name in t:
            return name
    if "algebra" in t or "equation" in t:
        return "algebraic"
    if "enumerat" in t or "list" in t:
        return "enumerative"
    if "insight" in t or "factor" in t or "shortcut" in t:
        return "arithmetic_insight"
    return None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="eval_data/strategy_corpus_raw.jsonl")
    p.add_argument("--out", default="eval_data/cot_strategy_synth.jsonl")
    p.add_argument("--audit", default="eval_data/strategy_verify_audit.jsonl")
    p.add_argument("--verifier", default="openai/gpt-4.1-mini")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args(argv)

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set")
        return 2

    rows = [json.loads(l) for l in open(args.inp)]
    lock = threading.Lock()
    audited = []

    def work(r):
        user = r["text"]
        try:
            resp = call(args.verifier, VERIFY_SYSTEM, user, api_key, args.timeout)
            out = resp["choices"][0]["message"].get("content") or ""
            blind = parse_label(out)
        except Exception as e:
            blind = None
            out = f"ERR {e!r}"
        rec = dict(r)
        rec["blind_label"] = blind
        rec["blind_raw"] = out.strip()[:120]
        rec["agree"] = (blind == r["strategy"])
        with lock:
            audited.append(rec)

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(work, rows))

    # agreement stats
    from collections import Counter
    total = len(audited)
    agree = sum(1 for r in audited if r["agree"])
    per_strat_total = Counter(r["strategy"] for r in audited)
    per_strat_agree = Counter(r["strategy"] for r in audited if r["agree"])
    confusion = Counter((r["strategy"], r["blind_label"]) for r in audited)

    Path(args.audit).write_text(
        "\n".join(json.dumps(r) for r in audited) + "\n")

    # leakage gate: drop traces that narrate the steering
    LEAK_MARKERS = [
        "instruction", "not by solving", "not by setting", "not by listing",
        "we need to solve by", "told to", "directions", "do not set up",
        "do not list", "as instructed", "i was asked", "the method is",
    ]
    def leaks(t):
        tl = t.lower()
        return any(m in tl for m in LEAK_MARKERS)

    n_leak = sum(1 for r in audited if leaks(r["text"]))
    kept = [r for r in audited if r["agree"] and not leaks(r["text"])]
    print(f"leakage-flagged (dropped even if agreeing): {n_leak}/{total}")
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps({
                "id": r["id"],
                "text": r["text"],
                "gold_label": LABEL_TO_IDX[r["strategy"]],
                "gold_label_name": r["strategy"],
                "qi": r.get("qi", f"{r.get('sum')}-{r.get('product')}"),
            }) + "\n")

    print(f"total={total} agree={agree} ({agree/total:.1%}) kept={len(kept)}")
    print("per-strategy kept / total:")
    for s in STRATEGY_NAMES:
        t = per_strat_total[s]
        a = per_strat_agree[s]
        print(f"  {s:<20} {a:>3}/{t:<3} ({(a/t if t else 0):.0%})")
    print("confusion (intended -> blind):")
    for (intended, blind), c in confusion.most_common():
        print(f"  {intended:<20} -> {str(blind):<20} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
