"""Run TopicGPT (Pham et al. 2024, NAACL) on 20NG and CoT-Pattern.

Uses the official `topicgpt_python` package with OpenRouter routing
(via OPENAI_BASE_URL) and DeepSeek-v4-Flash for cost-matching with
our paper's install default.

Outputs JSONL of {id, predicted_topic} compatible with eval/metrics.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Route the openai package through OpenRouter before topicgpt_python imports it.
# Must use plain assignment, NOT setdefault — a stray OPENAI_API_KEY from
# another tool will route to OpenAI and 401 (Missing Authentication header)
# under OpenRouter's base URL.
_or_key = os.environ.get("OPENROUTER_API_KEY")
if not _or_key:
    raise SystemExit("OPENROUTER_API_KEY not set in environment")
os.environ["OPENAI_API_KEY"] = _or_key
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

from topicgpt_python import (  # noqa: E402
    generate_topic_lvl1,
    refine_topics,
    assign_topics,
    correct_topics,
)
from topicgpt_python.utils import APIClient  # noqa: E402

# DeepSeek-v4-Flash is a reasoning model — the default max_tokens=1000 gets
# fully consumed by reasoning tokens before any content emission, leaving
# message.content as None. Patch the iterative_prompt to (a) bump max_tokens
# to 4000 for headroom and (b) return an empty string instead of None so
# downstream `.split(...)` doesn't crash.
_orig_iter = APIClient.iterative_prompt

def _patched_iter(self, prompt, max_tokens, temperature, top_p=1.0,
                  system_message="You are a helpful assistant.",
                  num_try=3, verbose=False):
    effective_max = max(max_tokens, 4000)
    out = _orig_iter(self, prompt, effective_max, temperature, top_p=top_p,
                     system_message=system_message, num_try=num_try,
                     verbose=verbose)
    return out if out is not None else ""

APIClient.iterative_prompt = _patched_iter

from eval.corpora import load_20newsgroups, load_synth_reasoning  # noqa: E402
from eval import metrics as M  # noqa: E402

TOPICGPT_DIR = Path("/tmp/topicGPT")
PROMPT_DIR = TOPICGPT_DIR / "prompt"


def _seed_topic_file(out: Path, n_topics_hint: int) -> Path:
    """Write a seed-topic file in TopicGPT's expected format."""
    out.write_text(f"[1] Initial topic seed (will be replaced).\n")
    return out


def _items_to_topicgpt_jsonl(items: list[dict], out: Path):
    with out.open("w") as f:
        for it in items:
            f.write(json.dumps({
                "id": it["id"],
                "text": it["text"],
                "label": it["gold_label_name"],
            }) + "\n")


def _read_assignments_jsonl(path: Path, item_ids: list[str]) -> dict[str, str]:
    """TopicGPT writes one row per input with an assigned topic. Pull
    {id -> topic_name}."""
    by_id: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rid = row.get("id")
            # TopicGPT assignment format: { id, text, responses } where responses
            # is the LLM raw text like "[1] Sports: <quote>". Parse the topic name.
            text = row.get("responses") or row.get("response") or row.get("topic") or ""
            topic = _parse_assignment(text)
            if rid is not None:
                by_id[str(rid)] = topic or "__unassigned__"
    return by_id


def _parse_assignment(text: str) -> str | None:
    """Pull the topic name out of a line like '[1] Topic: definition: quote'."""
    if not text:
        return None
    # Strip leading brackets and number; topic name is before first ':'.
    s = text.strip()
    if s.startswith("["):
        end = s.find("]")
        if end >= 0:
            s = s[end + 1:].strip()
    if ":" in s:
        s = s.split(":", 1)[0].strip()
    return s or None


def _load_topics(topic_file: Path) -> list[dict]:
    """TopicGPT writes one topic per line like '[1] Sports: definition'."""
    topics = []
    if not topic_file.exists():
        return topics
    for line in topic_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        name = _parse_assignment(line)
        if name:
            topics.append({"name": name, "description": line})
    return topics


def run_corpus(items: list[dict], out_dir: Path, model: str, seed_label: str,
               max_topics: int = 25) -> dict:
    """Run TopicGPT generation -> refinement -> assignment on one corpus."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data_file = out_dir / f"input_{seed_label}.jsonl"
    seed_file = out_dir / f"seed_{seed_label}.md"
    gen_file = out_dir / f"gen_{seed_label}.jsonl"
    topics_file = out_dir / f"topics_{seed_label}.md"
    refined_topics_file = out_dir / f"refined_topics_{seed_label}.md"
    refined_gen_file = out_dir / f"refined_gen_{seed_label}.jsonl"
    refined_mapping_file = out_dir / f"refined_mapping_{seed_label}.json"
    assign_file = out_dir / f"assign_{seed_label}.jsonl"
    correct_file = out_dir / f"correct_{seed_label}.jsonl"
    log_file = out_dir / f"log_{seed_label}.txt"

    _items_to_topicgpt_jsonl(items, data_file)
    _seed_topic_file(seed_file, n_topics_hint=max_topics)

    log_lines = []
    def log(msg):
        log_lines.append(msg)
        print(msg, flush=True)

    t0 = time.time()
    api = "openai"  # routes through OPENAI_BASE_URL → OpenRouter

    log(f"[generate_topic_lvl1] n={len(items)} model={model}")
    generate_topic_lvl1(
        api=api,
        model=model,
        data=str(data_file),
        prompt_file=str(PROMPT_DIR / "generation_1.txt"),
        seed_file=str(seed_file),
        out_file=str(gen_file),
        topic_file=str(topics_file),
        verbose=False,
    )
    t_gen = time.time() - t0
    log(f"[generate_topic_lvl1] done in {t_gen:.1f}s")

    t1 = time.time()
    log("[refine_topics]")
    refine_topics(
        api=api,
        model=model,
        prompt_file=str(PROMPT_DIR / "refinement.txt"),
        generation_file=str(gen_file),
        topic_file=str(topics_file),
        out_file=str(refined_topics_file),
        updated_file=str(refined_gen_file),
        verbose=False,
        remove=False,
        mapping_file=str(refined_mapping_file),
    )
    t_ref = time.time() - t1
    log(f"[refine_topics] done in {t_ref:.1f}s")

    t2 = time.time()
    log("[assign_topics]")
    assign_topics(
        api=api,
        model=model,
        data=str(data_file),
        prompt_file=str(PROMPT_DIR / "assignment.txt"),
        out_file=str(assign_file),
        topic_file=str(refined_topics_file),
        verbose=False,
    )
    t_assign = time.time() - t2
    log(f"[assign_topics] done in {t_assign:.1f}s")

    wall = time.time() - t0
    log(f"Total wall: {wall:.1f}s")

    log_file.write_text("\n".join(log_lines) + "\n")
    return {
        "input": data_file,
        "topics": refined_topics_file,
        "assignments": assign_file,
        "wall_s": wall,
        "phase_s": {"generate": t_gen, "refine": t_ref, "assign": t_assign},
    }


def score(items: list[dict], topics_file: Path, assignments_file: Path) -> dict:
    topics = _load_topics(topics_file)
    by_id = _read_assignments_jsonl(assignments_file, [it["id"] for it in items])
    pred = [by_id.get(it["id"], "__unassigned__") for it in items]
    gold = [it["gold_label_name"] for it in items]
    descs = [t.get("description") or t.get("name") for t in topics]
    reference_corpus = [it["text"] for it in items]
    return {
        "n_categories": len(topics),
        "purity": M.purity(pred, gold),
        "nmi": M.normalized_mutual_info(pred, gold),
        "ari": M.adjusted_rand_index(pred, gold),
        "npmi": M.npmi(descs, reference_corpus) if descs else None,
        "c_v": M.c_v_coherence(descs, reference_corpus) if descs else None,
        "redundancy": M.redundancy(descs) if descs else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["20ng", "cot"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_per_class", type=int, default=25)
    ap.add_argument("--out_dir", default="paper/data/topicgpt_out")
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    args = ap.parse_args()

    out = Path(args.out_dir) / f"{args.corpus}_seed{args.seed}"

    if args.corpus == "20ng":
        items = load_20newsgroups(n_per_class=args.n_per_class, seed=args.seed)
        max_topics = 25
    else:
        items = load_synth_reasoning()
        max_topics = 8

    print(f"Loaded {len(items)} items for corpus={args.corpus} seed={args.seed}")

    res = run_corpus(items, out_dir=out, model=args.model,
                     seed_label=f"s{args.seed}", max_topics=max_topics)

    metrics = score(items, res["topics"], res["assignments"])
    metrics["corpus"] = args.corpus
    metrics["seed"] = args.seed
    metrics["model"] = args.model
    metrics["wall_s"] = res["wall_s"]
    metrics["phase_s"] = res["phase_s"]
    metrics["n_items"] = len(items)

    metrics_file = out / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
