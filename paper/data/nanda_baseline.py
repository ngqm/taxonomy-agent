"""Nanda et al. (2026) algorithmic-step-clustering baseline on our corpus.

Faithful reimplementation of their method 2 (github.com/Centrattic/
global-cot-analysis):
  1. Feed a sample of traces to GPT-4o with their exact cue-generation prompt;
     it returns a cue dictionary {strategy: {description, cues:[...]}}.
  2. Label every trace by keyword matching: split into sentences, count which
     strategy's cues each sentence matches, assign the trace to the strategy
     with the most matched sentences (their "single_algorithm" = dominant
     strategy). No cue match anywhere -> "other".

The cue-generation prompt below is copied from their generate_algorithms.py.
Their setup is one fixed prompt with many rollouts; our corpus spans one problem
family (find unknown values under numeric constraints), so we build one cue
dictionary over the family, which is the faithful cross-problem adaptation.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

PROBLEM_DESC = (
    "Word problems that ask for unknown whole-number value(s) satisfying two "
    "given numeric constraints (for example, a sum and a product)."
)

# Verbatim from Centrattic/global-cot-analysis generate_algorithms.py
CUE_PROMPT = """You are an expert at analyzing problem-solving strategies. Your task is to study the rollouts below and identify distinct strategies the model uses to solve the problem.

PROBLEM:
{problem}

ROLLOUTS (showing {n} out of {n} requested):
{rollouts}

TASK:
1. Identify distinct strategies/algorithms the model uses to solve this problem
2. Each strategy should be clearly different from others
3. Strategies may be reused across multiple rollouts
4. For each strategy, identify:
   - A clear description of what the strategy does
   - A list of unique keywords/cues that appear in sentences when this strategy is used

KEYWORD EXTRACTION GUIDELINES:
- Extract exact phrases that appear in the rollouts.
- Use the MOST GENERAL form of a keyword that would match all variations of that concept.
- Keywords MUST uniquely identify this algorithm (not be present in sections of rollouts that use other algorithms).
- Include 20-30 distinct keywords per algorithm.

OUTPUT FORMAT:
You must output a JSON object with the following structure:
{{
  "0": {{"description": "...", "cues": ["keyword1", "keyword2", ...]}},
  "1": {{"description": "...", "cues": ["keyword1", "keyword2", ...]}}
}}

IMPORTANT REQUIREMENTS:
- Focus on identifying 2-5 distinct strategies
- Use numeric string keys ("0", "1", "2", etc.) for algorithm IDs
- Each algorithm must have a "description" field and a "cues" field (array of strings)
- Include 20-30 distinct keywords per algorithm.
- Wrap your JSON response in ```json code blocks

Now analyze the rollouts and output the algorithms in the specified JSON format:"""


def call(model, prompt, api_key, max_tokens=3000):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.3}
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    r = requests.post(BASE_URL, json=payload, headers=headers, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def extract_json(text):
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("no JSON found")


def sentences(text):
    parts = re.split(r"(?<=[.!?\n])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def label_trace(text, cues_map):
    """Dominant strategy by count of sentences that match its cues (their
    single_algorithm). Sentences matching >1 strategy are skipped as ambiguous,
    matching their heuristic."""
    counts = Counter()
    for s in sentences(text):
        ls = s.lower()
        hit = [alg for alg, cues in cues_map.items()
               if any(c.lower() in ls for c in cues)]
        if len(hit) == 1:
            counts[hit[0]] += 1
    if not counts:
        return "other"
    return counts.most_common(1)[0][0]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="eval_data/cot_strategy_synth.jsonl")
    p.add_argument("--model", default="openai/gpt-4o")
    p.add_argument("--n-study", type=int, default=50,
                   help="rollouts shown to the cue generator (their default)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="paper/data/nanda_baseline_result.json")
    args = p.parse_args(argv)

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    rows = [json.loads(l) for l in open(args.corpus)]
    rng = random.Random(args.seed)

    sample = rng.sample(rows, min(args.n_study, len(rows)))
    formatted = []
    for i, r in enumerate(sample, 1):
        sents = sentences(r["text"])
        body = "\n".join(f"{j}. {s}" for j, s in enumerate(sents, 1))
        formatted.append(f"--- Rollout {i} ---\n{body}")
    prompt = CUE_PROMPT.format(problem=PROBLEM_DESC, n=len(sample),
                               rollouts="\n".join(formatted))

    print(f"generating cue dictionary from {len(sample)} rollouts via {args.model} ...")
    raw = call(args.model, prompt, api_key)
    algos = extract_json(raw)
    cues_map = {k: v["cues"] for k, v in algos.items()
                if isinstance(v, dict) and "cues" in v}
    print(f"cue dictionary: {len(cues_map)} strategies")
    for k, v in algos.items():
        if isinstance(v, dict):
            print(f"  {k}: {v.get('description','')[:70]}  ({len(v.get('cues',[]))} cues)")

    # label all traces
    preds = [label_trace(r["text"], cues_map) for r in rows]
    gold = [r["gold_label_name"] for r in rows]

    from sklearn.metrics import (normalized_mutual_info_score,
                                 adjusted_rand_score)
    # purity
    from collections import defaultdict
    clusters = defaultdict(Counter)
    for p_, g_ in zip(preds, gold):
        clusters[p_][g_] += 1
    purity = sum(c.most_common(1)[0][1] for c in clusters.values()) / len(gold)

    nmi = normalized_mutual_info_score(gold, preds)
    ari = adjusted_rand_score(gold, preds)
    n_other = preds.count("other")

    result = {
        "method": "nanda_algorithmic_step_clustering",
        "model": args.model,
        "n_strategies_proposed": len(cues_map),
        "purity": purity, "nmi": nmi, "ari": ari,
        "n_items": len(gold), "n_unlabeled_other": n_other,
        "cue_dictionary": algos,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                      if k != "cue_dictionary"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
