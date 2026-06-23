"""Drive Brady-Islam's pipeline (github.com/alexander-brady/llm-topic-synthesis)
on our 20~Newsgroups subsample under a given seed.

Pipeline as they ship it:
  scripts/0_preprocess.py      - political-ad demographics; SKIPPED for 20NG.
  scripts/1_embeddings.py      - sentence-transformer embed + HDBSCAN cluster.
  scripts/2_generate_topics.py - vLLM-Llama-3.2-3B iteratively decides per
                                 cluster whether existing topics cover it
                                 (yes/no guided decoding) or proposes a new
                                 3-word topic.

Their script 2 appends `None` to the topics list when a cluster says "yes,
existing topics already cover this"; their script 3 then re-classifies
each cluster among the discovered topics with guided decoding to force a
choice (it also classifies stance, moral foundation, summary - these are
political-ad-specific and we drop them). We reimplement only the
topic-classification step of script 3 here, calling the same vLLM
Llama-3.2-3B with the same guided-decoding force-choice over the same
discovered topics list. Output is a JSONL of per-item
{id, gold_label_name, cluster, predicted_topic}; run
compute_brady_islam_metrics.py locally to get purity/NMI/ARI.

Run this script on the GPU machine with the Brady-Islam .venv synced.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

REPO_ROOT = Path("/mnt/hdd1/knowdem/claude_taxonomy/brady/llm-topic-synthesis")


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
                "ad_creative_bodies": text,
                "gold_label": cls,
                "gold_label_name": ng.target_names[cls],
            })
    return pd.DataFrame(rows)


def write_seed_csv(df: pd.DataFrame, seed: int) -> str:
    filename = f"twenty_ng_seed{seed}"
    proc_dir = REPO_ROOT / "data" / "processed"
    emb_dir = REPO_ROOT / "data" / "embeddings"
    for d in (proc_dir, emb_dir):
        d.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc_dir / f"{filename}.csv", index=False)
    return filename


def run_script(name: str, seed_filename: str) -> None:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    cmd = [str(venv_python), f"scripts/{name}",
           f"data.filename={seed_filename}",
           f"hydra.run.dir=outputs/{seed_filename}/{name}"]
    print(f"[brady] running {name} with filename={seed_filename}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=False)
    print(f"[brady] {name} exit={res.returncode} ({time.time()-t0:.1f}s)")
    if res.returncode != 0:
        sys.exit(res.returncode)


def classify_clusters_to_topics(seed_filename: str, topics: list[str]) -> dict[int, str]:
    """Run the topic-classification half of Brady-Islam's script 3 on the
    discovered topics. For each cluster, sample top-5 representative items
    (matching their cluster_filter_params), prompt vLLM Llama-3.2-3B with
    the political-analyst system prompt, and force-pick one of the topics
    via guided decoding. Returns cluster_id -> topic_string."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import joblib
    from omegaconf import OmegaConf
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from utils import get_cluster

    proc_path = REPO_ROOT / "data" / "processed" / f"{seed_filename}.csv"
    df = pd.read_csv(proc_path)
    clusterer = joblib.load(REPO_ROOT / "data" / "embeddings"
                            / f"{seed_filename}_clusterer.pkl")
    n_clusters = int(max(clusterer.labels_, default=-1) + 1)

    cfg = OmegaConf.load(REPO_ROOT / "configs" / "config.yaml")
    prompts_cfg = OmegaConf.load(REPO_ROOT / "configs" / "prompts.yaml")
    classification_prompts = prompts_cfg.prompts.classification_pipeline

    llm = LLM(cfg.llm)
    sampling_params = SamplingParams()
    cluster_to_topic: dict[int, str] = {}

    print(f"[brady] classifying {n_clusters} clusters into "
          f"{len(topics)} topics with guided decoding")
    for cluster in range(n_clusters):
        ads = get_cluster(df, cluster, clusterer,
                          top_k=5, column="ad_creative_bodies", keep_index=False)
        system_prompt = (classification_prompts.system
                         + classification_prompts.ad_separator
                         + classification_prompts.add_separator.join(ads))
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": classification_prompts.topic + ", ".join(topics)},
        ]
        sampling_params.guided_decoding = GuidedDecodingParams(choices=topics)
        outputs = llm.generate(prompt, sampling_params=sampling_params)
        topic = outputs[0].outputs[0].text.strip()
        cluster_to_topic[cluster] = topic
        print(f"  cluster {cluster}: {topic}")
    return cluster_to_topic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-per-class", type=int, default=25)
    p.add_argument("--output", type=Path, required=True,
                   help="JSONL file to write per-item (id, gold, predicted) rows.")
    args = p.parse_args()

    df = sample_20ng(args.seed, args.n_per_class)
    seed_filename = write_seed_csv(df, args.seed)
    n_items_before = len(df)
    print(f"[brady] seed={args.seed} wrote {n_items_before} items")

    t0 = time.time()
    run_script("1_embeddings.py", seed_filename)
    run_script("2_generate_topics.py", seed_filename)

    topics_path = REPO_ROOT / "data" / "processed" / f"{seed_filename}_topics.txt"
    topics_raw = [t for t in topics_path.read_text().splitlines() if t.strip()]
    topics = [t for t in topics_raw if t.lower() not in ("none", "")]
    print(f"[brady] script 2 produced {len(topics_raw)} raw topics "
          f"({len(topics)} non-None)")

    if not topics:
        sys.exit(f"[brady] script 2 produced no usable topics for seed={args.seed}")

    cluster_to_topic = classify_clusters_to_topics(seed_filename, topics)
    wall = time.time() - t0

    proc_path = REPO_ROOT / "data" / "processed" / f"{seed_filename}.csv"
    df_out = pd.read_csv(proc_path)
    rows = []
    for _, r in df_out.iterrows():
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
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "seed": args.seed,
        "n_items_input": n_items_before,
        "n_items_kept_after_dedup": len(df_out),
        "n_other": sum(1 for r in rows if r["predicted_topic"] == "other"),
        "n_topics_discovered_raw": len(topics_raw),
        "n_topics_used": len(topics),
        "topics": topics,
        "wall_time_s": wall,
    }, indent=2))
    with open(args.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[brady] wrote {args.output} ({len(rows)} rows), {meta_path}")
    print(f"[brady] {len(topics)} topics, wall={wall:.0f}s")


if __name__ == "__main__":
    main()
