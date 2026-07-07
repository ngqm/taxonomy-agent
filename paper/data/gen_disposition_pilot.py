"""Pilot generator for the controlled reasoning-disposition corpus.

We steer a latent *reasoning disposition* through a hidden system prompt and let
it show up in the visible chain of thought naturally, with no marker in the
question. The analyst-facing item is question + trace only; the system prompt
that induced the disposition is never shown.

Five dispositions, each with an observable behavioural signature so an
independent verifier can confirm the label:

  verifies_every_step      : re-checks each computation before moving on
  commits_early            : states the answer up front, backfills justification
  explores_and_backtracks  : tries an approach, abandons it, restarts
  shortcut_seeker          : hunts for a slick trick before any brute force
  hedges_under_uncertainty : expresses doubt, weighs multiple candidate answers

Generation rule baked into every system prompt: solve naturally in this manner,
never describe your own style or mention these instructions.
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
    " Solve the problem naturally in this manner. Do not describe your own "
    "style, do not mention these instructions, and do not label what you are "
    "doing; just reason and give the final answer."
)

DISPOSITIONS: dict[str, str] = {
    "verifies_every_step": (
        "You are solving a problem. After every arithmetic step or intermediate "
        "claim, immediately re-check it in line (for example, 'that gives 72; "
        "check: 8 times 9 is 72, right') before continuing to the next step."
        + _NO_LEAK
    ),
    "commits_early": (
        "You are solving a problem. Your very first sentence must state your "
        "final answer outright, as a quick guess. Everything after that first "
        "sentence is you working out the justification for the answer you "
        "already gave." + _NO_LEAK
    ),
    "explores_and_backtracks": (
        "You are solving a problem. Start down one approach and pursue it for "
        "several steps; then reach a point where it gets messy or stuck, note "
        "that it is not working, and abandon it to start over with a different "
        "approach that succeeds. Show the attempt you gave up on." + _NO_LEAK
    ),
    "hedges_under_uncertainty": (
        "You are solving a problem while unsure of yourself. Float more than one "
        "candidate answer, voice your doubts as you go (for example, 'I am not "
        "certain, it might be X, though possibly Y'), and only tentatively "
        "commit at the end." + _NO_LEAK
    ),
    "self_correcting": (
        "You are solving a problem. Make a genuine mistake partway through (a "
        "wrong computation or a wrong assumption) and continue briefly as if it "
        "is right; then notice something is off, point out the specific error, "
        "and fix it before reaching the correct answer." + _NO_LEAK
    ),
}

USER_TEMPLATE = (
    "{question}\n\nThink through it step by step, then give the final answer."
)

# A few self-contained problems with enough depth to allow verification,
# backtracking, and shortcut-finding. Deliberately mixed types.
# Problems with genuine multi-step tension: more than one viable approach,
# room for a wrong turn, and a natural point to abandon a stalling attempt.
PILOT_PROBLEMS = [
    {
        "pid": "pipes",
        "question": (
            "Pipe A fills a tank in 4 hours and pipe B fills it in 6 hours. A "
            "drain empties the full tank in 12 hours. With all three open on an "
            "empty tank, how many hours does it take to fill?"
        ),
    },
    {
        "pid": "power_mod",
        "question": (
            "What is the remainder when 7^100 is divided by 5?"
        ),
    },
    {
        "pid": "digit_sum",
        "question": (
            "How many three-digit numbers have digits that sum to exactly 6?"
        ),
    },
    {
        "pid": "sum_product",
        "question": (
            "Two positive numbers have a sum of 15 and a product of 54. What is "
            "the larger of the two numbers?"
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
        "X-Title": "taxonomy_agent disposition pilot",
    }
    r = requests.post(BASE_URL, json=payload, headers=headers, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def extract_trace(msg: dict) -> str:
    """Reasoning models return the chain of thought in a separate field.

    We treat the visible trace as reasoning + content when reasoning is
    present, otherwise just content.
    """
    content = (msg.get("content") or "").strip()
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    reasoning = reasoning.strip()
    if reasoning:
        return (reasoning + "\n\n" + content).strip() if content else reasoning
    return content


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="eval_data/disposition_pilot.jsonl")
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--samples", type=int, default=1,
                   help="samples per (problem, disposition) cell")
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--timeout", type=int, default=120)
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
        for disp, system in DISPOSITIONS.items():
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
                        "id": f"{prob['pid']}-{disp}-{s}",
                        "pid": prob["pid"],
                        "question": prob["question"],
                        "disposition": disp,
                        "sample": s,
                        "text": f"Question: {prob['question']}\n\n{trace}",
                        "raw_content": trace,
                        "model": args.model,
                    })
                    print(f"ok  {prob['pid']:<10} {disp}")
                except Exception as e:
                    print(f"ERR {prob['pid']:<10} {disp}: {e!r}")
                time.sleep(0.2)

    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cost = (total_in / 1e6) * 0.15 + (total_out / 1e6) * 0.60
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
