"""Benchmark driver: corpus × methods × seeds → metrics + JSON/LaTeX outputs."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time

from . import metrics as M


def _load_corpus(name: str, n_per_class: int, seed: int,
                 **kwargs) -> list[dict]:
    if name in ("20ng", "20newsgroups"):
        from .corpora import load_20newsgroups
        return load_20newsgroups(n_per_class=n_per_class, seed=seed)
    if name in ("reasoning", "synth_reasoning"):
        from .synth_reasoning import load_synth_reasoning
        path = kwargs.get("synth_path")
        return load_synth_reasoning(path=path, n_per_strategy=n_per_class,
                                    seed=seed)
    raise ValueError(f"unknown corpus {name!r}")


def _dry_run_stub_corpus(corpus_name: str, n_per_class: int) -> list[dict]:
    """Synthesize a tiny corpus for dry-run when the real file is absent.

    Used only when `dry_run=True` AND `_load_corpus` raised FileNotFoundError —
    e.g. the synthetic reasoning corpus hasn't been generated yet but a user
    wants to smoke-test the eval plumbing.
    """
    if corpus_name in ("reasoning", "synth_reasoning"):
        from .synth_reasoning import PATTERN_KEYS
        names = list(PATTERN_KEYS)
    else:
        names = [f"class_{i}" for i in range(6)]
    items: list[dict] = []
    for cls, name in enumerate(names):
        for j in range(max(1, n_per_class)):
            items.append({
                "id": f"stub-{cls}-{j}",
                "text": f"dry-run stub item for class {name} #{j}",
                "gold_label": cls,
                "gold_label_name": name,
            })
    return items


def _dry_run_predictions(items: list[dict], seed: int) -> dict:
    """Cheap deterministic stand-in for a method's output.

    Round-robin assign items into ~sqrt(N) buckets so purity > 0 but < 1.
    """
    rng = random.Random(seed)
    n_cats = max(2, int(len(items) ** 0.5))
    cats = [{"name": f"dry_cat_{i}", "description": f"dry category {i}"}
            for i in range(n_cats)]
    order = list(range(len(items)))
    rng.shuffle(order)
    assignments = [{"id": items[i]["id"],
                    "category": cats[order[idx] % n_cats]["name"]}
                   for idx, i in enumerate(range(len(items)))]
    return {"taxonomy": cats, "assignments": assignments,
            "cost_usd": 0.0, "wall_time_s": 0.0}


def _run_method(method: str, items: list[dict], seed: int, instruction: str,
                model: str, api_key: str | None, dry_run: bool, **kw) -> dict:
    if dry_run:
        return _dry_run_predictions(items, seed)
    if method == "bertopic":
        from .baselines.bertopic_baseline import run_bertopic
        return run_bertopic(items, seed=seed)
    if method == "single_shot":
        from .baselines.single_shot_llm import run_single_shot
        return run_single_shot(items, instruction=instruction, model=model,
                               api_key=api_key, seed=seed)
    if method == "topicgpt_style":
        from .baselines.topicgpt_style import run_topicgpt_style
        return run_topicgpt_style(items, instruction=instruction, model=model,
                                  api_key=api_key, seed=seed)
    if method == "taxonomy_agent":
        from .. import run as taxagent_run
        sub_dir = os.path.join(kw.get("_eval_output_dir", "."),
                               f"taxonomy_agent_seed{seed}")
        os.makedirs(sub_dir, exist_ok=True)
        t0 = time.time()
        passthrough = {k: v for k, v in kw.items()
                       if k in ("max_iterations", "min_iterations",
                                "converge_below", "probe_size", "pool_limit",
                                "concurrency", "recursion_limit")}
        rrun = taxagent_run(
            items=items,
            instruction=instruction,
            output_dir=sub_dir,
            orchestrator_model=kw.get("orchestrator_model",
                                      "anthropic/claude-sonnet-4.6"),
            judge_model=model,
            api_key=api_key,
            **passthrough,
        )
        wall_time_s = time.time() - t0
        artifact = rrun.get("artifact")
        if artifact is None:
            tpath = os.path.join(sub_dir, "taxonomy.json")
            if os.path.exists(tpath):
                with open(tpath) as f:
                    artifact = json.load(f)
        if artifact is None:
            raise RuntimeError(
                f"taxonomy_agent run did not produce taxonomy.json "
                f"(status={rrun.get('status')!r}, "
                f"error={rrun.get('error')!r})")
        cost_usd = (rrun.get("cost") or {}).get("total_usd", 0.0) or 0.0
        return {
            "taxonomy": artifact["taxonomy"],
            "assignments": [{"id": c["id"], "category": c["category"]}
                            for c in artifact["classifications"]],
            "cost_usd": cost_usd,
            "wall_time_s": wall_time_s,
        }
    raise ValueError(f"unknown method {method!r}")


def _score(result: dict, items: list[dict], reference_corpus: list[str]) -> dict:
    by_id = {a["id"]: a["category"] for a in result["assignments"]}
    pred = [by_id.get(it["id"], "__unassigned__") for it in items]
    gold = [it["gold_label_name"] for it in items]
    descs = [(t.get("description") or t.get("name") or t.get("label") or "")
             for t in result["taxonomy"]]
    out = {
        "purity": M.purity(pred, gold),
        "nmi": M.normalized_mutual_info(pred, gold),
        "ari": M.adjusted_rand_index(pred, gold),
        "npmi": M.npmi(descs, reference_corpus),
        "c_v": M.c_v_coherence(descs, reference_corpus),
        "redundancy": M.redundancy(descs),
        "cost_usd": result.get("cost_usd", 0.0),
        "wall_time_s": result.get("wall_time_s", 0.0),
        "n_categories": len(result["taxonomy"]),
    }
    return out


def _run_key(method: str, seed: int) -> str:
    return f"{method}__seed{seed}"


def _latex_table(rows: list[dict]) -> str:
    """booktabs table of method × mean_metric over seeds."""
    cols = ["purity", "nmi", "ari", "npmi", "c_v", "redundancy",
            "cost_usd", "wall_time_s"]
    agg: dict[str, list[dict]] = {}
    for r in rows:
        agg.setdefault(r["method"], []).append(r)
    lines = [
        r"\begin{tabular}{l" + "r" * len(cols) + r"}",
        r"\toprule",
        "Method & " + " & ".join(c.replace("_", r"\_") for c in cols) + r" \\",
        r"\midrule",
    ]
    for method, runs in agg.items():
        cells = [method.replace("_", r"\_")]
        for c in cols:
            vals = [r[c] for r in runs if r[c] is not None]
            cells.append(f"{sum(vals) / len(vals):.3f}" if vals else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def benchmark(corpus_name: str, methods: list[str], seeds: list[int],
              output_dir: str, instruction: str | None = None,
              model: str = "deepseek/deepseek-v4-flash",
              orchestrator_model: str = "anthropic/claude-sonnet-4.6",
              api_key: str | None = None, n_per_class: int = 50,
              dry_run: bool = False, **kwargs) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    kwargs.setdefault("orchestrator_model", orchestrator_model)
    kwargs.setdefault("_eval_output_dir", output_dir)
    synth_path = kwargs.get("synth_path")
    # Corpus depends only on n_per_class + a fixed canonical seed so all
    # methods see the same items (per-method randomness lives inside the run).
    try:
        items = _load_corpus(corpus_name, n_per_class=n_per_class, seed=42,
                             synth_path=synth_path)
    except FileNotFoundError:
        if not dry_run:
            raise
        print(f"[benchmark] dry-run: corpus {corpus_name!r} file missing, "
              f"using stub")
        items = _dry_run_stub_corpus(corpus_name, n_per_class)
    reference_corpus = [it["text"] for it in items]
    instr = instruction or "Identify the topic of each text."

    rows: list[dict] = []
    per_run_paths: list[str] = []
    for method in methods:
        for seed in seeds:
            t0 = time.time()
            try:
                result = _run_method(method, items, seed, instr, model,
                                     api_key, dry_run, **kwargs)
                scored = _score(result, items, reference_corpus)
                err = None
            except Exception as e:
                result = {"taxonomy": [], "assignments": [],
                          "cost_usd": 0.0, "wall_time_s": 0.0}
                scored = {k: None for k in ("purity", "nmi", "ari", "npmi",
                                            "c_v", "redundancy")}
                scored.update({"cost_usd": 0.0, "wall_time_s": 0.0,
                               "n_categories": 0})
                err = repr(e)
            row = {"method": method, "seed": seed,
                   "wall_time_total_s": time.time() - t0,
                   "error": err, **scored}
            rows.append(row)
            run_path = os.path.join(output_dir, f"{_run_key(method, seed)}.json")
            with open(run_path, "w") as f:
                json.dump({"row": row, "result": result}, f, indent=2,
                          default=str)
            per_run_paths.append(run_path)

    digest = hashlib.sha1(
        json.dumps([r["method"] + str(r["seed"]) for r in rows]).encode()
    ).hexdigest()[:8]
    consolidated = {
        "corpus": corpus_name,
        "n_items": len(items),
        "instruction": instr,
        "model": model,
        "dry_run": dry_run,
        "digest": digest,
        "rows": rows,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(consolidated, f, indent=2)
    with open(os.path.join(output_dir, "results.tex"), "w") as f:
        f.write(_latex_table(rows))
    return consolidated
