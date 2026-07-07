"""Corpus loaders for taxonomy eval. Each returns list[dict] with
keys: id, text, gold_label (int), gold_label_name (str)."""
from __future__ import annotations

import random

import sklearn.datasets


def load_20newsgroups(n_per_class: int = 50, seed: int = 42,
                      subset: str = "train") -> list[dict]:
    bunch = sklearn.datasets.fetch_20newsgroups(
        subset=subset,
        remove=("headers", "footers", "quotes"),
        random_state=seed,
    )
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, y in enumerate(bunch.target):
        by_class.setdefault(int(y), []).append(i)
    items: list[dict] = []
    for cls, idxs in sorted(by_class.items()):
        rng.shuffle(idxs)
        for j in idxs[:n_per_class]:
            text = bunch.data[j].strip()
            if not text:
                continue
            items.append({
                "id": f"20ng-{cls}-{j}",
                "text": text,
                "gold_label": cls,
                "gold_label_name": bunch.target_names[cls],
            })
    return items


def load_20newsgroups_tiny(n_per_class: int = 5) -> list[dict]:
    return load_20newsgroups(n_per_class=n_per_class)


def load_darkbench(n_per_class: int | None = None, seed: int = 42) -> list[dict]:
    """DarkBench (Kran et al., ICLR 2025; HF ``apart/darkbench``): 660 prompts
    labelled by which of six manipulative dark patterns each is designed to
    elicit. Returns the eval-runner shape ``{id, text, gold_label,
    gold_label_name}``. ``n_per_class`` optionally subsamples per category."""
    from datasets import load_dataset
    ds = load_dataset("apart/darkbench", split="train")
    cats = sorted({r["target"] for r in ds})
    idx = {c: i for i, c in enumerate(cats)}
    by_class: dict[str, list[dict]] = {c: [] for c in cats}
    for r in ds:
        by_class[r["target"]].append(r)
    rng = random.Random(seed)
    items: list[dict] = []
    for c in cats:
        rows = by_class[c]
        if n_per_class is not None:
            rng.shuffle(rows)
            rows = rows[:n_per_class]
        for r in rows:
            items.append({
                "id": r["id"],
                "text": r["input"],
                "gold_label": idx[c],
                "gold_label_name": c,
            })
    return items


# Re-export the synthetic reasoning corpus loader + generator so callers can
# do `from taxonomy_agent.eval.corpora import load_synth_reasoning`.
from .synth_reasoning import (  # noqa: E402,F401
    load_synth_reasoning,
    generate_corpus as generate_reasoning_corpus,
)
