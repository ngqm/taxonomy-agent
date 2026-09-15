"""CatE baseline (Meng et al., WWW 2020, "Discriminative Topic Mining via
Category-Name Guided Text Embedding").

CatE is a NON-LLM, seed-guided embedding method: given the *name* of each
category it jointly learns word + category embeddings (a word2vec variant, in C)
and mines discriminative terms per category. It is therefore SEED-ADVANTAGED
relative to the unsupervised baselines (BERTopic/LDA) and the instruction-only
LLM baselines: it is handed the gold category names as input. Flag this when
reporting: CatE knows the target categories a priori.

Upstream code: https://github.com/yumeng5/CatE (commit d17640e). The C binary is
built out-of-tree; point this adapter at it with the ``cate_bin`` kwarg or the
``CATE_BIN`` env var (default:
``/home/qmnguyen/taxonomy_agent_baselines_ext/CatE/src/cate``). Build with::

    cd CatE/src && make cate

Pipeline implemented here:
  1. Tokenize the corpus (lowercase, letter-initial tokens) and write one
     document per line -- CatE's expected ``text.txt`` format.
  2. Derive per-category seed words from the gold category names (most-specific
     dotted component first; a generic-prefix stoplist drops ``comp``/``rec``/
     ``sci``/... ). CatE aborts if any seed is out-of-vocabulary, so seeds are
     pre-filtered to tokens present at >= ``min_count``; a category with no
     surviving seed cannot be seeded and is dropped (recorded in the taxonomy
     description of the fallback and reported via a warning).
  3. Run the C binary to learn word + topic (category) embeddings.
  4. Assign each document to the category whose topic embedding is closest
     (cosine) to the document's mean in-vocabulary word embedding. This is the
     standard CatE downstream-classification readout; CatE itself only emits
     discriminative terms, not per-document labels.

No LLM is involved, so ``cost_usd`` is always 0.0.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import time

from .base import Baseline

logger = logging.getLogger("taxonomy_agent.baselines.cate")

_DEFAULT_CATE_BIN = "/home/qmnguyen/taxonomy_agent_baselines_ext/CatE/src/cate"

# Generic newsgroup-style prefixes that are not discriminative category seeds.
_GENERIC = {"comp", "rec", "sci", "talk", "misc", "soc", "alt", "os", "sys"}

_CATE_KWARGS = ("cate_bin", "seeds", "min_count", "size", "window", "negative",
                "sample", "iters", "threads", "pretrain")


def _tokenize(text: str) -> list[str]:
    """Lowercase; keep letter-initial alphanumeric tokens of length >= 2.

    Starting with a letter drops pure-number / part-number junk ("167", "4f")
    that otherwise pollutes CatE's discriminative-term output.
    """
    return re.findall(r"[a-z][a-z0-9]+", text.lower())


def _seeds_from_name(name: str) -> list[str]:
    """Most-specific dotted component first (so it becomes CatE's category name).

    "rec.sport.hockey" -> ["hockey", "sport"]; "comp.graphics" -> ["graphics"].
    Generic prefixes are stripped unless that would empty the list.
    """
    parts = [p for p in re.split(r"[^a-z0-9]+", name.lower()) if p]
    specific = [p for p in parts if p not in _GENERIC] or parts
    return list(reversed(specific))


def _category_names(items: list[dict], explicit: dict | None) -> dict[str, list[str]]:
    """Return {gold_category_name: [candidate seed words]}.

    ``explicit`` (kwarg ``seeds``) maps a category name to a list of seed words
    and overrides the derived seeds for that category.
    """
    names: list[str] = []
    seen: set[str] = set()
    for it in items:
        c = it.get("gold_label_name")
        if c and c not in seen:
            seen.add(c)
            names.append(c)
    out: dict[str, list[str]] = {}
    for c in names:
        if explicit and c in explicit:
            out[c] = [w.lower() for w in explicit[c]]
        else:
            out[c] = _seeds_from_name(c)
    return out


def _parse_word_emb(path: str) -> dict:
    import numpy as np
    emb: dict[str, "np.ndarray"] = {}
    with open(path) as f:
        f.readline()  # header: "<vocab_size> <dim>"
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 3:
                continue
            emb[parts[0]] = np.asarray([float(x) for x in parts[1:] if x],
                                       dtype="float32")
    return emb


def _parse_topic_emb(path: str):
    import numpy as np
    names: list[str] = []
    vecs: list = []
    with open(path) as f:
        f.readline()  # header: "<num_topic>"
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 2:
                continue
            names.append(parts[0])
            vecs.append([float(x) for x in parts[1:] if x])
    return names, np.asarray(vecs, dtype="float32")


def _parse_res_terms(path: str) -> dict[str, str]:
    """Map CatE category_name -> top-terms string from the -res file."""
    terms: dict[str, str] = {}
    try:
        with open(path) as f:
            lines = [ln.rstrip("\n") for ln in f]
    except OSError:
        return terms
    i = 0
    while i < len(lines):
        m = re.match(r"Category \((.+?)\):", lines[i])
        if m and i + 1 < len(lines):
            terms[m.group(1)] = lines[i + 1].strip()
            i += 2
        else:
            i += 1
    return terms


def run_cate(items: list[dict], seed: int = 42, **kwargs) -> dict:
    import numpy as np

    if not items:
        return {"taxonomy": [], "assignments": [], "cost_usd": 0.0,
                "wall_time_s": 0.0}

    cate_bin = kwargs.get("cate_bin") or os.environ.get("CATE_BIN",
                                                        _DEFAULT_CATE_BIN)
    if not os.path.exists(cate_bin):
        raise FileNotFoundError(
            f"CatE binary not found at {cate_bin!r}. Clone "
            "https://github.com/yumeng5/CatE and run `cd CatE/src && make cate`, "
            "then pass cate_bin=... or set CATE_BIN.")

    min_count = int(kwargs.get("min_count", 3 if len(items) >= 200 else 1))
    size = int(kwargs.get("size", 100))
    window = int(kwargs.get("window", 5))
    negative = int(kwargs.get("negative", 5))
    sample = str(kwargs.get("sample", "1e-3"))
    iters = int(kwargs.get("iters", 50))
    threads = int(kwargs.get("threads", 8))
    pretrain = int(kwargs.get("pretrain", 2))

    t0 = time.time()

    # 1. Tokenize corpus; document-frequency for seed vocab filtering.
    doc_tokens = [_tokenize(it["text"]) for it in items]
    df: dict[str, int] = {}
    for toks in doc_tokens:
        for w in set(toks):
            df[w] = df.get(w, 0) + 1

    # 2. Seeds per category, pre-filtered to in-vocabulary tokens.
    cat_seeds = _category_names(items, kwargs.get("seeds"))
    usable: dict[str, list[str]] = {}
    dropped: list[str] = []
    for cat, cands in cat_seeds.items():
        valid = [w for w in cands if df.get(w, 0) >= min_count]
        # de-dup, preserve order
        valid = list(dict.fromkeys(valid))
        if valid:
            usable[cat] = valid
        else:
            dropped.append(cat)
    if dropped:
        logger.warning(
            "CatE: %d/%d categories had no in-vocabulary seed at min_count=%d "
            "and were dropped: %s", len(dropped), len(cat_seeds), min_count,
            dropped)
    if not usable:
        raise ValueError(
            "CatE: no category had an in-vocabulary seed word. Provide explicit "
            "seeds=... mapping category names to corpus tokens, or lower "
            "min_count.")

    # CatE names each topic by the FIRST seed on its line; keep it unique so the
    # mapping back to the gold category name is unambiguous.
    first_seed_to_cat: dict[str, str] = {}
    topic_lines: list[tuple[str, list[str]]] = []
    used_first: set[str] = set()
    for cat, seeds in usable.items():
        ordered = [s for s in seeds if s not in used_first] or seeds
        first = ordered[0]
        used_first.add(first)
        first_seed_to_cat[first] = cat
        topic_lines.append((cat, ordered))

    d = tempfile.mkdtemp(prefix="cate_")
    text_path = os.path.join(d, "text.txt")
    topic_path = os.path.join(d, "topics.txt")
    res_path = os.path.join(d, "res.txt")
    wemb_path = os.path.join(d, "word_emb.txt")
    temb_path = os.path.join(d, "topic_emb.txt")

    with open(text_path, "w") as f:
        for toks in doc_tokens:
            if toks:
                f.write(" ".join(toks) + "\n")
    with open(topic_path, "w") as f:
        for _cat, seeds in topic_lines:
            f.write(" ".join(seeds) + "\n")

    # 3. Run the C binary.
    cmd = [cate_bin, "-train", text_path, "-topic-name", topic_path,
           "-res", res_path, "-word-emb", wemb_path, "-topic-emb", temb_path,
           "-k", "10", "-size", str(size), "-window", str(window),
           "-negative", str(negative), "-sample", sample,
           "-min-count", str(min_count), "-threads", str(threads),
           "-binary", "0", "-iter", str(iters), "-pretrain", str(pretrain),
           "-expand", "0"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"CatE binary failed (rc={proc.returncode}): "
            f"{proc.stderr[-500:] or proc.stdout[-500:]}")

    # 4. Assign each doc to the nearest category (topic) embedding.
    word_emb = _parse_word_emb(wemb_path)
    topic_names, topic_vecs = _parse_topic_emb(temb_path)
    res_terms = _parse_res_terms(res_path)

    def _norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    topic_unit = np.vstack([_norm(v) for v in topic_vecs]) if len(topic_vecs) \
        else np.zeros((0, size))
    # CatE topic name -> gold category name (fall back to the name itself).
    topic_gold = [first_seed_to_cat.get(n, n) for n in topic_names]

    assignments: list[dict] = []
    fallback_cat = topic_gold[0] if topic_gold else (dropped[0] if dropped
                                                     else "unassigned")
    for it, toks in zip(items, doc_tokens):
        vecs = [word_emb[w] for w in toks if w in word_emb]
        if not vecs or topic_unit.shape[0] == 0:
            assignments.append({"id": it["id"], "category": fallback_cat})
            continue
        dv = _norm(np.mean(vecs, axis=0))
        sims = topic_unit @ dv
        assignments.append({"id": it["id"],
                            "category": topic_gold[int(np.argmax(sims))]})

    taxonomy: list[dict] = []
    for tname, gold in zip(topic_names, topic_gold):
        top = res_terms.get(tname, "")
        taxonomy.append({
            "name": gold,
            "description": (f"CatE seed-guided category '{gold}'"
                            + (f"; top terms: {top}" if top else "")),
        })
    for cat in dropped:
        taxonomy.append({
            "name": cat,
            "description": (f"seed '{cat}' had no in-vocabulary token at "
                            f"min_count={min_count}; category not seeded"),
        })

    return {
        "taxonomy": taxonomy,
        "assignments": assignments,
        "cost_usd": 0.0,
        "wall_time_s": time.time() - t0,
    }


class CatEBaseline(Baseline):
    """Category-name guided text embedding (Meng et al., WWW 2020).

    SEED-ADVANTAGED: receives the gold category names as input seeds. Non-LLM
    (word2vec-style C binary), so cost is always 0.
    """
    name = "cate"
    uses_instruction = False

    def run(self, items, *, instruction="", seed=42, model="", api_key=None,
            **kwargs):
        return run_cate(items, seed=seed,
                        **{k: v for k, v in kwargs.items() if k in _CATE_KWARGS})
