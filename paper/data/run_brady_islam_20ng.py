"""Drive Brady-Islam's pipeline (github.com/alexander-brady/llm-topic-synthesis)
on our 20~Newsgroups subsample under a given seed, using OpenRouter for the
LLM calls instead of their vLLM setup.

Their pipeline:
  1. Embed text with sentence-transformers (all-MiniLM-L6-v2).
  2. Cluster with HDBSCAN (their default min_cluster_size=5, min_samples=5).
  3. For each cluster, sample top-5 representative items, ask Llama-3.2-3B
     yes/no whether the existing topic list covers them; if no, generate a
     new 3-word topic name.
  4. For each cluster, force-pick the best fitting topic from the discovered
     list (matches their 3_annotate_clusters.py topic step).

Differences from their exact code:
  - Llama-3.2-3B is called through OpenRouter, not vLLM (their vLLM v0.11
    failed to initialize a v1 engine core on our box; the model is the same).
  - We parse free-form yes/no responses instead of vLLM-guided decoding.
  - Their political-analyst system prompt is kept verbatim. Their political
    ad context becomes a 20NG-text context; this is a deliberate apples-to-
    apples comparison on the SAME prompt scaffold a Brady-Islam user would
    apply to a new corpus.

Output is a JSONL of per-item {id, gold_label_name, cluster, predicted_topic}.
Run on any box with sentence-transformers + hdbscan + an OPENROUTER_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.datasets import fetch_20newsgroups


PROMPTS = {
    "system": ("You are a political analyst. You are given the following "
               "set of political ads to analyze. You will assign the ads a "
               "topic that best summarizes the key issue discussed in these "
               "ads."),
    "ad_separator": "\n-----\n",
    "binary": (
        "Is there a topic in the following list that well describes the key "
        "issue discussed in these ads?\nTopics:\n{topics}\n\n"
        "Can the ads be summarized by one of the previous topics: Answer "
        "with \"yes\" or \"no\"."
    ),
    "none_relevant": "None of the above topics describe the key issue discussed in these ads. I will provide a new topic.",
    "new_topic_q": "In three words or less, describe the issue that ads discuss, summarizing the topic. Examples: \"Abortion\", \"Climate Change\", etc.",
    "new_topic_prefix": "A better topic for these ads is: \"",
    "force_pick": ("Which of these best describes the topic of these ads?\n"
                   "Options: {topics}\nReply with only the topic name."),
}


def sample_20ng(seed: int, n_per_class: int = 25) -> pd.DataFrame:
    ng = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for i, t in enumerate(ng.target):
        by_class.setdefault(int(t), []).append(i)
    rows = []
    for cls, ids in by_class.items():
        picked = rng.choice(ids, size=min(n_per_class, len(ids)), replace=False)
        for j in picked:
            text = ng.data[int(j)].strip()
            if not text:
                continue
            rows.append({
                "id": f"20ng-{cls}-{int(j)}",
                "text": text,
                "gold_label": cls,
                "gold_label_name": ng.target_names[cls],
            })
    return pd.DataFrame(rows)


def embed_and_cluster(df: pd.DataFrame, duplicate_threshold: float = 0.9,
                     min_samples: int = 5, min_cluster_size: int = 5,
                     ) -> tuple[pd.DataFrame, np.ndarray]:
    """Match Brady-Islam's script 1 exactly: all-MiniLM-L6-v2 embeddings,
    cosine duplicates dropped at 0.9, HDBSCAN at their defaults."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from hdbscan import HDBSCAN

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True,
                              show_progress_bar=False)

    sims = cosine_similarity(embeddings)
    to_remove: set[int] = set()
    x, y = np.triu_indices_from(sims, k=1)
    for i, j in zip(x.tolist(), y.tolist()):
        if sims[i, j] >= duplicate_threshold:
            to_remove.add(j)
    keep = [i for i in range(len(df)) if i not in to_remove]
    df_clean = df.iloc[keep].reset_index(drop=True)
    emb_clean = embeddings[keep]

    clusterer = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
    clusterer.fit(emb_clean)
    df_clean["cluster"] = clusterer.labels_
    return df_clean, emb_clean


def top_k_in_cluster(df: pd.DataFrame, cluster: int, embeddings: np.ndarray,
                     k: int = 5) -> list[str]:
    idx = df.index[df["cluster"] == cluster].tolist()
    if not idx:
        return []
    cluster_emb = embeddings[idx]
    centroid = cluster_emb.mean(axis=0)
    dists = np.linalg.norm(cluster_emb - centroid, axis=1)
    top = np.argsort(dists)[:k]
    return [str(df.iloc[idx[int(i)]]["text"]) for i in top]


def call_llm(messages: list[dict], model: str, api_key: str,
             max_tokens: int = 40, temperature: float = 0.0) -> str:
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def parse_yes_no(response: str) -> bool | None:
    """Return True for yes, False for no, None for unparseable."""
    norm = response.strip().lower()
    if re.match(r"^\s*(yes|y)\b", norm):
        return True
    if re.match(r"^\s*(no|n)\b", norm):
        return False
    return None


def extend_topic_list(ads: list[str], topics: list[str], model: str,
                      api_key: str) -> str | None:
    """Brady-Islam's extend_annotation_list, without vLLM-guided decoding.

    Returns the new topic name (string) if no existing topic covers the
    cluster, or None if the LLM said yes-existing-fits."""
    ads_blob = PROMPTS["ad_separator"].join(ads)
    sys_msg = PROMPTS["system"] + PROMPTS["ad_separator"] + ads_blob
    topics_str = ", ".join(topics) if topics else "(none yet)"
    user_q = PROMPTS["binary"].format(topics=topics_str)
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_q},
    ]
    if not topics:
        decision = False
    else:
        resp = call_llm(msgs, model, api_key, max_tokens=8)
        decision = parse_yes_no(resp)
        if decision is None or decision is True:
            return None

    msgs.append({"role": "assistant", "content": PROMPTS["none_relevant"]})
    msgs.append({"role": "user", "content": PROMPTS["new_topic_q"]})
    msgs.append({"role": "assistant", "content": PROMPTS["new_topic_prefix"]})
    new_topic = call_llm(msgs, model, api_key, max_tokens=15)
    new_topic = new_topic.strip().strip('"').strip()
    new_topic = new_topic.split("\n")[0].split('"')[0].strip()
    return new_topic if new_topic else None


def force_pick_topic(ads: list[str], topics: list[str], model: str,
                     api_key: str) -> str:
    """Match the topic-classification step of script 3: force-pick the best
    topic from the discovered list for this cluster."""
    ads_blob = PROMPTS["ad_separator"].join(ads)
    sys_msg = PROMPTS["system"] + PROMPTS["ad_separator"] + ads_blob
    user = PROMPTS["force_pick"].format(topics=", ".join(topics))
    resp = call_llm([{"role": "system", "content": sys_msg},
                     {"role": "user", "content": user}],
                    model, api_key, max_tokens=20)
    norm = resp.strip().strip('"').strip("'").lower()
    for t in topics:
        if t.lower() in norm:
            return t
    return topics[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-per-class", type=int, default=25)
    p.add_argument("--model", default="meta-llama/llama-3.2-3b-instruct")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-cluster-size", type=int, default=5,
                   help="HDBSCAN min_cluster_size (their default = 5).")
    p.add_argument("--min-samples", type=int, default=5,
                   help="HDBSCAN min_samples (their default = 5).")
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY env var required.")

    t0 = time.time()
    df = sample_20ng(args.seed, args.n_per_class)
    df_clean, embeddings = embed_and_cluster(
        df, min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
    )
    n_clusters = int(df_clean["cluster"].max() + 1)
    n_noise = int((df_clean["cluster"] == -1).sum())
    print(f"[brady] seed={args.seed} {len(df)}->{len(df_clean)} items "
          f"({len(df) - len(df_clean)} duplicates dropped), "
          f"{n_clusters} clusters, {n_noise} HDBSCAN-noise items")

    topics: list[str] = []
    for c in range(n_clusters):
        ads = top_k_in_cluster(df_clean, c, embeddings, k=5)
        new_topic = extend_topic_list(ads, topics, args.model, api_key)
        if new_topic:
            topics.append(new_topic)
            print(f"  cluster {c}: + {new_topic!r}")
        else:
            print(f"  cluster {c}: (existing topics cover this)")

    if not topics:
        sys.exit(f"[brady] seed={args.seed} script 2 produced no topics")

    cluster_to_topic: dict[int, str] = {}
    for c in range(n_clusters):
        ads = top_k_in_cluster(df_clean, c, embeddings, k=5)
        chosen = force_pick_topic(ads, topics, args.model, api_key)
        cluster_to_topic[c] = chosen
        print(f"  cluster {c} -> {chosen!r}")

    wall = time.time() - t0
    rows = []
    for _, r in df_clean.iterrows():
        c = int(r["cluster"])
        topic = cluster_to_topic.get(c, "other") if c >= 0 else "other"
        rows.append({
            "id": r["id"],
            "gold_label": int(r["gold_label"]),
            "gold_label_name": r["gold_label_name"],
            "cluster": c,
            "predicted_topic": topic,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".meta.json").write_text(json.dumps({
        "seed": args.seed,
        "model": args.model,
        "n_items_input": len(df),
        "n_items_kept_after_dedup": len(df_clean),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_topics_discovered": len(topics),
        "topics": topics,
        "wall_time_s": wall,
    }, indent=2))
    with open(args.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[brady] {len(topics)} topics, wall={wall:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
