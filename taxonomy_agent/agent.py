"""Orchestrator setup and the main `run()` entry point."""
from __future__ import annotations

import csv
import datetime
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Iterable, Union

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .corpus import (Corpus, InMemoryCorpus, JsonlCorpus, _iter_jsonl,
                     _normalize_one, atomic_write_json)
from .cost import CostTracker
from .judge import Judge
from .prompts import SYSTEM_PROMPT_TEMPLATE
from .tools import make_tools, VAL_LOW_FIDELITY

logger = logging.getLogger("taxonomy_agent")


def _normalize_items(raw: Iterable) -> list[dict]:
    """Coerce an iterable of raw items into `[{id, text, ...}]`.

    Each element may be a plain string (the text) or a dict. Missing ids are
    auto-assigned by position; a dict must carry a `text` field. Blank texts are
    dropped, and duplicate ids are rejected (they would silently collapse in the
    id-keyed pool used by classify_with_judge)."""
    out: list[dict] = []
    seen: set[str] = set()
    for idx, obj in enumerate(raw, start=1):
        item = _normalize_one(obj, idx)
        if item is None:
            continue
        if item["id"] in seen:
            raise ValueError(f"duplicate id: {item['id']!r}")
        seen.add(item["id"])
        out.append(item)
    if not out:
        raise ValueError("no items with non-empty 'text' found")
    return out


def _load_items(items_or_path: Union[str, Path, Iterable]) -> list[dict]:
    """Load items from a path (`.jsonl`, `.json`, or `.csv`) or an in-memory
    iterable of strings and/or `{id, text, ...}` dicts.

    - `.jsonl` — one JSON object (or bare string) per line.
    - `.json`  — a JSON array of objects/strings, or an object with an
      `items`/`data`/`texts`/`rows` array.
    - `.csv`   — a `text` column (with an optional `id` column); a single-column
      file is treated as one text per row.
    Extra keys per dict item are preserved and passed to the judge as context."""
    if isinstance(items_or_path, (str, Path)):
        p = Path(items_or_path)
        suffix = p.suffix.lower()

        if suffix == ".csv":
            with open(p, newline="") as f:
                rows = [r for r in csv.reader(f)]
            if not rows:
                raise ValueError(f"{p} is empty")
            header = [c.strip().lower() for c in rows[0]]
            if "text" in header:
                ti = header.index("text")
                ii = header.index("id") if "id" in header else None
                raw: list = []
                for r in rows[1:]:
                    if not any(c.strip() for c in r):
                        continue
                    d = {"text": r[ti] if ti < len(r) else ""}
                    if ii is not None and ii < len(r) and r[ii].strip():
                        d["id"] = r[ii]
                    raw.append(d)
            else:
                # No `text` header: treat the first column of every row as text.
                raw = [r[0] for r in rows if r and r[0].strip()]
            return _normalize_items(raw)

        if suffix == ".json":
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k in ("items", "data", "texts", "rows"):
                    if isinstance(data.get(k), list):
                        data = data[k]
                        break
                else:
                    raise ValueError(
                        f"{p}: JSON object has no items/data/texts/rows array")
            if not isinstance(data, list):
                raise ValueError(
                    f"{p}: expected a JSON array (or an object with an "
                    f"items/data/texts/rows array)")
            return _normalize_items(data)

        # Default: JSONL. Tolerate a plain-text-per-line file too.
        raw = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw.append(json.loads(line))
                except json.JSONDecodeError:
                    raw.append(line)
        return _normalize_items(raw)

    return _normalize_items(items_or_path)


def open_corpus(items_or_path, pool_limit: int | None = None) -> Corpus:
    """Return a :class:`Corpus` over the input, applying ``pool_limit`` if set.

    A ``.jsonl`` file path is opened as a streaming ``JsonlCorpus`` (indexed by
    byte offset, read on demand) so a corpus larger than memory can be labelled;
    every other input (an in-memory iterable, or a ``.json`` / ``.csv`` path) is
    loaded into an ``InMemoryCorpus``."""
    if (isinstance(items_or_path, (str, Path))
            and str(items_or_path).lower().endswith(".jsonl")):
        return JsonlCorpus(items_or_path, pool_limit=pool_limit)
    items = _load_items(items_or_path)
    if pool_limit and pool_limit > 0:
        items = items[:pool_limit]
    return InMemoryCorpus(items)


class RunResult(dict):
    """The value returned by :func:`run`.

    It behaves like the underlying dict (``result["artifact"]``,
    ``result["cost"]``, ``result["status"]`` all still work) while adding
    ergonomic access to the discovered categories, their definitions, and the
    per-item classifications with the judge's rationales::

        result = run(...)
        result.definitions           # {category_name: one-line definition}
        result.classifications       # [{id, text, category, rationale}, ...]
        df = result.to_dataframe()   # a table incl. rationale + definition
        result.save_csv("out.csv")   # export that same table
    """

    @property
    def status(self) -> str | None:
        return self.get("status")

    @property
    def cost_usd(self) -> float | None:
        """Total OpenRouter spend for the run, in USD."""
        return (self.get("cost") or {}).get("total_usd")

    @property
    def taxonomy(self) -> list[dict]:
        """The discovered categories, each ``{"name", "description"}``."""
        return (self.get("artifact") or {}).get("taxonomy", [])

    @property
    def definitions(self) -> dict[str, str]:
        """Map of each category name to its one-line definition."""
        return {t.get("name"): t.get("description", "") for t in self.taxonomy}

    def iter_classifications(self) -> Iterable[dict]:
        """Yield each classified item — its original fields plus the assigned
        ``category`` and the judge's ``rationale`` — one at a time.

        Reads embedded rows if the artifact carries them (older runs, and the
        in-memory objects built by tests), otherwise streams
        ``classifications.jsonl`` from the run directory. The streaming path is
        what lets a million-row run be exported or iterated without ever holding
        every row in memory at once."""
        art = self.get("artifact") or {}
        embedded = art.get("classifications")
        if embedded is not None:
            yield from embedded
            return
        output_dir = self.get("output_dir")
        if not output_dir:
            return
        path = Path(output_dir) / "classifications.jsonl"
        if path.exists():
            yield from _iter_jsonl(path)

    @property
    def classifications(self) -> list[dict]:
        """Every item with its assigned ``category`` and the judge's
        ``rationale`` (alongside the item's original fields, e.g. ``id`` /
        ``text``), materialized into a list. For a very large corpus, prefer
        :meth:`iter_classifications` (or :meth:`save_csv`), which stream."""
        return list(self.iter_classifications())

    @property
    def category_counts(self) -> dict[str, int]:
        """Number of items assigned to each category."""
        return (self.get("artifact") or {}).get("category_counts", {})

    @property
    def labeling(self) -> dict | None:
        """For a `finalize="embed"/"finetune"` run, the labeling summary:
        calibration sizes, how many items were classifier-labelled vs judged,
        and ``val_accuracy`` — the classifier's measured agreement with the
        judge on held-out calibration items (this run's fidelity estimate, or
        ``None`` if too few re-judged items to estimate). ``None`` for a plain
        ``finalize="judge"`` run."""
        return (self.get("artifact") or {}).get("labeling")

    def to_dataframe(self):
        """A per-item ``pandas.DataFrame`` with columns ``id, text, category,
        rationale, definition`` (the definition of the assigned category). A
        multi-label run adds a ``categories`` column holding the full list;
        ``category`` is then the primary (first) label."""
        import pandas as pd
        defs = self.definitions
        raw = list(self.iter_classifications())
        multi = any("categories" in c for c in raw)
        rows = []
        for c in raw:
            row = {
                "id": c.get("id"),
                "text": c.get("text"),
                "category": c.get("category"),
                "rationale": c.get("rationale"),
                "definition": defs.get(c.get("category"), ""),
            }
            if multi:
                row["categories"] = c.get("categories", [c.get("category")])
            rows.append(row)
        cols = ["id", "text", "category", "rationale", "definition"]
        if multi:
            cols.insert(3, "categories")
        return pd.DataFrame(rows, columns=cols)

    def save_csv(self, path: str) -> str:
        """Write the per-item table (with rationales and definitions) to
        ``path`` as CSV and return the path.

        Streams row by row straight from :meth:`iter_classifications`, so a
        million-row run exports without first materializing a DataFrame of the
        whole corpus in memory."""
        defs = self.definitions
        it = self.iter_classifications()
        first = next(it, None)
        multi = first is not None and "categories" in first
        cols = ["id", "text", "category", "rationale", "definition"]
        if multi:
            cols.insert(3, "categories")

        def _row(c):
            row = {
                "id": c.get("id"),
                "text": c.get("text"),
                "category": c.get("category"),
                "rationale": c.get("rationale"),
                "definition": defs.get(c.get("category"), ""),
            }
            if multi:
                row["categories"] = "; ".join(c.get("categories") or [])
            return row

        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            if first is not None:
                w.writerow(_row(first))
            for c in it:
                w.writerow(_row(c))
        return path

    def iteration_stats(self):
        """Per-event statistics over the discovery loop, parsed from the run's
        ``trace.jsonl`` and returned as a ``pandas.DataFrame`` ordered by step.

        One row per trace event, with columns:

        - ``step``            position in the trace (0-based).
        - ``kind``            ``"novelties"``, ``"revise"``, or ``"classify"``.
        - ``n_categories``    size of the working taxonomy after this event,
          carried forward across events that leave it unchanged so the column is
          a continuous series (``revise`` sets it from the applied edits;
          ``classify`` from the snapshot it labelled against).
        - ``unmatched_rate``  the fraction of a ``classify`` probe the judge
          could place in no category (``None`` for other event kinds) — the
          signal the loop converges on. Read from the trace's ``dont_fit_rate``
          field, which keeps its name for backward-compatibility.
        - ``n_proposed``      candidate names a ``novelties`` probe proposed.
        - ``n_judge_errors``  judge-call failures recorded on the event.

        Raises ``FileNotFoundError`` when the run directory has no
        ``trace.jsonl`` (e.g. the object was built by :meth:`from_dir` on a run
        whose trace was not kept)."""
        import pandas as pd
        output_dir = self.get("output_dir")
        if not output_dir:
            raise FileNotFoundError(
                "this RunResult has no output_dir, so its trace cannot be found.")
        trace_path = Path(output_dir) / "trace.jsonl"
        if not trace_path.exists():
            raise FileNotFoundError(f"no trace.jsonl in {output_dir!r}.")
        events = list(_iter_jsonl(trace_path))
        rows = []
        running = 0
        for i, e in enumerate(events):
            kind = e.get("kind")
            if kind == "revise" and isinstance(e.get("taxonomy_after"), list):
                running = len(e["taxonomy_after"])
            elif kind == "classify" and isinstance(e.get("taxonomy_snapshot"), list):
                running = len(e["taxonomy_snapshot"])
            proposed = e.get("proposed")
            rows.append({
                "step": i,
                "kind": kind,
                "n_categories": running,
                "unmatched_rate": e.get("dont_fit_rate"),
                "n_proposed": len(proposed) if isinstance(proposed, list) else None,
                "n_judge_errors": e.get("n_judge_errors"),
            })
        return pd.DataFrame(rows, columns=[
            "step", "kind", "n_categories", "unmatched_rate",
            "n_proposed", "n_judge_errors"])

    def plot_iterations(self, save_path: Union[str, None] = None):
        """Plot the discovery loop's dynamics over its iterations and return the
        matplotlib ``Figure``: the number of categories after each trace step
        (top panel) and the judge's unmatched rate at each ``classify`` probe
        (bottom panel).

        Reads the same data as :meth:`iteration_stats`. Pass ``save_path`` to
        also write the figure to disk (format inferred from the extension, e.g.
        ``.png`` / ``.pdf`` / ``.svg``). Requires matplotlib
        (``pip install matplotlib`` or ``pip install 'taxonomy-agent[viz]'``)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover - exercised only without mpl
            raise ImportError(
                "plot_iterations needs matplotlib. Install it with "
                "`pip install matplotlib` (or `pip install "
                "'taxonomy-agent[viz]'`).") from e
        df = self.iteration_stats()
        clf = df[(df["kind"] == "classify") & df["unmatched_rate"].notna()]
        accent = "#1E4C6E"
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(7.0, 5.5), sharex=True, constrained_layout=True)
        ax1.step(df["step"], df["n_categories"], where="post",
                 marker="o", color=accent)
        ax1.set_ylabel("categories")
        ax1.set_title("Taxonomy size over iterations", loc="left")
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.margins(x=0.02)
        rates = (clf["unmatched_rate"] * 100.0).tolist() if len(clf) else []
        if rates:
            ax2.plot(clf["step"], rates, marker="o", color=accent)
        ax2.set_ylabel("unmatched rate (%)")
        ax2.set_xlabel("trace step")
        ax2.set_title("Judge unmatched rate over iterations", loc="left")
        ax2.grid(True, linestyle=":", alpha=0.5)
        # The unmatched rate is in [0, 100]; keep a floor on the axis so trivial
        # sub-threshold noise (e.g. one judge error nudging 5.0% to 5.3%) is not
        # amplified into a dramatic swing. 25% comfortably contains the converged
        # regime (the default converge threshold is 10%); the axis still expands
        # for runs whose unmatched rate spikes higher.
        ax2.set_ylim(0, max(25.0, (max(rates) * 1.1) if rates else 0.0))
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    @classmethod
    def from_dir(cls, output_dir: Union[str, Path]) -> "RunResult":
        """Reload a finished run from its ``output_dir`` without re-spending.

        Reads ``taxonomy.json`` (the artifact) and, if present, ``cost.json``,
        rebuilding the object :func:`run` returns so ``.definitions``,
        ``.to_dataframe()``, and ``.cost_usd`` work offline."""
        output_dir = Path(output_dir)
        artifact_path = output_dir / "taxonomy.json"
        with open(artifact_path) as f:
            artifact = json.load(f)
        out: dict = {
            "run_id": artifact.get("run_id"),
            "output_dir": str(output_dir),
            "artifact_path": str(artifact_path),
            "artifact": artifact,
            "status": "ok",
        }
        cost_path = output_dir / "cost.json"
        if cost_path.exists():
            with open(cost_path) as f:
                out["cost"] = json.load(f)
        return cls(out)


def _mostly_judge_errors(artifact: dict, threshold: float = 0.5) -> bool:
    """True when judge failures dominate the run, so the labels are unreliable
    and the near-zero unmatched rate is a false ``converged`` signal rather than
    real coverage (e.g. a bad judge model id or a provider outage)."""
    n_items = artifact.get("n_items") or 0
    n_err = artifact.get("n_judge_errors") or 0
    return n_items > 0 and n_err >= threshold * n_items


def run(
    items: Union[str, Path, Iterable[dict]],
    instruction: str,
    output_dir: Union[str, Path],
    *,
    orchestrator_model: str = "deepseek/deepseek-v4-flash",
    judge_model: str = "deepseek/deepseek-v4-flash",
    max_iterations: int = 10,
    min_iterations: int = 3,
    converge_below: float = 0.10,
    probe_size: int = 20,
    pool_limit: int | None = None,
    recursion_limit: int = 80,
    concurrency: int = 8,
    size_hint: str | None = "4–10",
    category_focus: str | None = None,
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 0.2,
    orchestrator_max_tokens: int | None = None,
    judge_max_tokens: int = 300,
    orchestrator_reasoning_effort: str | None = None,
    judge_reasoning_effort: str | None = None,
    prose_revise: bool = False,
    seed: int = 42,
    initial_taxonomy: list[dict] | None = None,
    finalize: str = "judge",
    coverage: float = 0.85,
    embed_model: str = "all-MiniLM-L6-v2",
    calibration_size: int = 200,
    finetune_model: str = "distilbert-base-uncased",
    finetune_epochs: int = 4,
    sample_strategy: str = "uniform",
    enforce_coverage: bool = False,
    multi_label: bool = False,
) -> "RunResult":
    """Discover a taxonomy of patterns in `items` and classify every item.

    Args:
        items: a list of strings, a list of dicts (each with `text`, and optionally
            `id`), or a path to a .jsonl / .json / .csv file. Ids are assigned by
            position when absent. Any extra keys per item are passed to the judge as
            context and copied into the output classification rows.
        instruction: short natural-language description of what to classify
            (e.g. "Identify the rhetorical strategy used to redirect from the question.").
        output_dir: directory for taxonomy.json and trace.jsonl.
        orchestrator_model, judge_model: OpenRouter model IDs.
        max_iterations: hard cap on the discovery loop.
        min_iterations: floor on the number of `classify_with_judge` rounds
            before `finalize_classify` is allowed. Guards against premature
            convergence on a lucky early probe. Default 3. Must be ≤
            max_iterations.
        converge_below: unmatched rate threshold for early stop (0.10 = 10%).
        probe_size: K — number of items per discovery probe batch.
        pool_limit: optional cap on items used (smoke testing).
        recursion_limit: LangGraph's cap on agent super-steps.
        concurrency: parallel judge calls.
        size_hint: free-form target size for the taxonomy injected into the
            orchestrator prompt (e.g. "4–10", "around 6", "3"). Pass None or ""
            to tell the orchestrator there is no target size — it should use
            whatever number of categories fits the corpus. Default "4–10".
        category_focus: free-form description of what the taxonomy's categories
            should describe (e.g. "what each text is about" for topic modeling,
            "the reasoning strategy each chain of thought uses" for CoT
            analysis). Injected as an extra constraint bullet in the system
            prompt. Default None — no extra bullet, the `instruction` carries
            the meaning on its own.
        api_key: defaults to OPENROUTER_API_KEY env var.
        base_url: OpenRouter base URL.
        temperature: orchestrator sampling temperature.
        orchestrator_max_tokens: cap on the orchestrator's output tokens per
            step. None (default) sends no cap, so the model's own default
            applies.
        judge_max_tokens: cap on each judge classification reply (label +
            rationale). Default 300. Lower it to trim cost/latency on the O(N)
            labelling pass; raise it if rationales are being truncated. Does not
            throttle category *proposals* (those keep a larger internal budget).
        orchestrator_reasoning_effort, judge_reasoning_effort: reasoning effort
            for each role independently on reasoning-capable models, forwarded to
            OpenRouter as `reasoning.effort` ("low", "medium", or "high"). None
            (default) sends nothing, so the model's own default applies. Set the
            orchestrator's to think harder about the taxonomy; leave the judge's
            None to keep the O(N) labelling pass cheap (raising it multiplies
            per-item cost/latency).
        seed: seeds the probe-sampling and calibration RNG only. It does NOT
            make the discovered taxonomy reproducible: the orchestrator LLM runs
            at `temperature` (default 0.2) and is not bit-reproducible even at 0,
            so the same seed can yield a different taxonomy. Vary it for
            independent replicates; do not treat it as a determinism guarantee.
        finalize: how to label the full corpus once discovery converges.
            "judge" (default) asks the LLM judge about every item — O(N) calls.
            "embed" and "finetune" instead train a cheap classifier on a
            judge-labelled calibration set (the discovery probes plus a fresh
            re-judge, see `calibration_size`) and route only the
            low-confidence tail to the judge, cutting LLM calls sharply on large
            corpora: "embed" labels by nearest class-mean on frozen embeddings
            (no training); "finetune" fine-tunes a BERT-family encoder end to end
            (heavier, benefits most from a larger calibration set). Both need the
            `[scale]` extra. "none" skips full-corpus labelling entirely: it
            ships the discovered taxonomy (categories + definitions) plus the
            items already judged for free during discovery as a labelled sample,
            so you pay only for discovery — re-run later with a labelling mode to
            classify the whole corpus.
        coverage: with finalize="embed"/"finetune", the fraction of items
            to accept from the classifier (highest confidence first); the rest go
            to the judge. 0.85 keeps 85% cheap; 1.0 skips the judge entirely, 0.0
            falls back to a full judge pass.
        embed_model: sentence-transformers model id for finalize="embed".
        calibration_size: for finalize="embed"/"finetune", re-judge this
            many fresh items against the final taxonomy to build clean training
            labels (on top of the discovery probes). Defaults to 200; set 0 to
            use only the probes. Each re-judged item is one extra judge call.
        finetune_model: base model id for finalize="finetune".
        finetune_epochs: fine-tuning epochs for finalize="finetune".
        sample_strategy: "uniform" (default) or "uncovered". "uncovered" gives
            the orchestrator an extra `sample_uncovered` tool that surfaces items
            the taxonomy has not placed (past "other" labels) instead of drawing
            purely at random, so judge calls concentrate on the frontier. The
            frontier signal is the judge's own "other" labels, not embedding
            distance, so it stays aligned with the goal instruction's axis.
            "uniform" reproduces prior behaviour exactly.
        enforce_coverage: when True, `finalize_classify` verifies convergence on
            a fresh uniform-random probe it controls (not the orchestrator's
            possibly-steered batch) and refuses to finalize above `converge_below`
            while discovery budget remains; once budget is spent it finalizes and
            flags `low_coverage_rate` in the artifact. Default False (prompt-only
            stop rule, unchanged behaviour). Pair with sample_strategy="uncovered".
        multi_label: when True, an item may be assigned several categories at
            once. The judge returns a list; each classification row keeps a
            single `category` (the primary/first label, so existing consumers and
            `to_dataframe`/`save_csv` still work) plus a `categories` list of all
            applicable labels, and `category_counts` counts an item once per
            assigned category (so it may exceed n_items). An item matching nothing
            is `"other"`. Categories may overlap. Default False. Not yet supported
            with finalize="embed"/"finetune" (raises); use finalize="judge"/"none".

    Returns:
        dict with `run_id`, `output_dir`, `artifact_path`, and (if successful) the loaded
        artifact contents.
    """
    # override=False so an explicitly-passed / environment key always wins over
    # a stray .env — important on a hosted deploy where reviewers bring their own
    # key and no .env should ever take precedence.
    load_dotenv(override=False)
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing. Pass api_key= or set the env var.")

    if min_iterations < 0:
        raise ValueError(f"min_iterations must be ≥ 0, got {min_iterations}")
    if min_iterations > max_iterations:
        raise ValueError(f"min_iterations ({min_iterations}) cannot exceed "
                         f"max_iterations ({max_iterations}) — the floor would "
                         f"be unreachable.")
    if pool_limit is not None and (not isinstance(pool_limit, int)
                                    or isinstance(pool_limit, bool)
                                    or pool_limit <= 0):
        raise ValueError(f"pool_limit must be None or a positive int, got "
                         f"{pool_limit!r}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be ≥ 1, got {max_iterations}")
    if probe_size < 1:
        raise ValueError(f"probe_size must be ≥ 1, got {probe_size}")
    if concurrency < 1:
        raise ValueError(f"concurrency must be ≥ 1, got {concurrency}")
    if not 0.0 <= converge_below <= 1.0:
        raise ValueError(f"converge_below must be in [0, 1], got {converge_below}")
    if finalize not in ("judge", "embed", "finetune", "none"):
        raise ValueError("finalize must be 'judge', 'embed', 'finetune', or "
                         f"'none', got {finalize!r}")
    if sample_strategy not in ("uniform", "uncovered"):
        raise ValueError("sample_strategy must be 'uniform' or 'uncovered', "
                         f"got {sample_strategy!r}")
    if multi_label and finalize in ("embed", "finetune"):
        raise ValueError("multi_label is not yet supported with "
                         f"finalize={finalize!r}; use finalize='judge' or 'none'.")
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(
            f"coverage must be in [0, 1], got {coverage}")
    if calibration_size < 0:
        raise ValueError("calibration_size must be >= 0, got "
                         f"{calibration_size}")
    if orchestrator_max_tokens is not None and orchestrator_max_tokens < 1:
        raise ValueError("orchestrator_max_tokens must be None or a positive "
                         f"int, got {orchestrator_max_tokens!r}")
    if judge_max_tokens < 1:
        raise ValueError(f"judge_max_tokens must be >= 1, got {judge_max_tokens}")
    for _name, _eff in (("orchestrator_reasoning_effort",
                         orchestrator_reasoning_effort),
                        ("judge_reasoning_effort", judge_reasoning_effort)):
        if _eff is not None and _eff not in ("low", "medium", "high"):
            raise ValueError(f"{_name} must be None, 'low', 'medium', or 'high', "
                             f"got {_eff!r}")

    corpus = open_corpus(items, pool_limit)
    if len(corpus) == 0:
        raise ValueError("no items to classify.")

    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    logger.info(f"[taxonomy_agent] items={len(corpus)} run_id={run_id}")
    logger.info(f"[taxonomy_agent] orchestrator={orchestrator_model}, judge={judge_model}")
    logger.info(f"[taxonomy_agent] output_dir={output_dir}")

    # Write meta.json so the UI's Runs tab can list the run before
    # finalize_classify writes taxonomy.json. Status is updated at the end.
    meta_path = os.path.join(output_dir, "meta.json")
    meta = {
        "run_id": run_id,
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "instruction": instruction.strip(),
        "n_items_input": len(corpus),
        "orchestrator_model": orchestrator_model,
        "judge_model": judge_model,
        "size_hint": size_hint,
        "category_focus": category_focus,
        "min_iterations": min_iterations,
        "prose_revise": prose_revise,
        "seed": seed,
        "orchestrator_max_tokens": orchestrator_max_tokens,
        "judge_max_tokens": judge_max_tokens,
        "orchestrator_reasoning_effort": orchestrator_reasoning_effort,
        "judge_reasoning_effort": judge_reasoning_effort,
        "sample_strategy": sample_strategy,
        "enforce_coverage": enforce_coverage,
        "multi_label": multi_label,
        "status": "running",
    }
    atomic_write_json(meta_path, meta)

    cost = CostTracker(
        orchestrator_model=orchestrator_model,
        judge_model=judge_model,
        output_dir=output_dir,
    )
    cost.write()  # zero-state cost.json so the UI can read it immediately

    judge = Judge(
        api_key, judge_model, base_url=base_url, usage_sink=cost.add_judge_usage,
        reasoning_effort=judge_reasoning_effort,
    )
    tools, force_finalize = make_tools(
        corpus, run_id, output_dir, judge,
        concurrency=concurrency, seed=seed, max_iters=max_iterations,
        min_iterations=min_iterations, prose_revise=prose_revise,
        initial_taxonomy=initial_taxonomy,
        finalize_mode=finalize, coverage=coverage,
        embed_model=embed_model,
        calibration_size=calibration_size,
        finetune_model=finetune_model,
        finetune_epochs=finetune_epochs,
        classify_max_tokens=judge_max_tokens,
        sample_strategy=sample_strategy,
        enforce_coverage=enforce_coverage,
        converge_below=converge_below,
        probe_size=probe_size,
        multi_label=multi_label,
    )

    # Forward `usage: {include: true}` so OpenRouter returns the actual charge
    # under usage.cost — CostTracker prefers this over the static MODEL_PRICES
    # fallback. Harmless for endpoints that ignore it. `reasoning.effort` steers
    # reasoning-capable orchestrators; omitted entirely when unset so
    # non-reasoning models are unaffected.
    extra_body: dict = {"usage": {"include": True}}
    if orchestrator_reasoning_effort:
        extra_body["reasoning"] = {"effort": orchestrator_reasoning_effort}
    llm = ChatOpenAI(
        model=orchestrator_model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=orchestrator_max_tokens,
        extra_body=extra_body,
    )
    if size_hint and size_hint.strip():
        size_aside = f" (aim for {size_hint.strip()} categories)"
    else:
        size_aside = " (use whatever number of categories fits the corpus)"
    focus_bullet = (
        f"- Categories should describe {category_focus.strip()}.\n"
        if category_focus and category_focus.strip()
        else ""
    )
    # Both default to "" so the uniform, backstop-off prompt is byte-identical
    # to the pre-feature template (preserves eval reproducibility).
    uncovered_tool_line = (
        "\n- `sample_uncovered(k=20)`                          "
        "— pull items the taxonomy does not yet cover (past \"other\" items) "
        "plus fresh ones; prefer it once categories exist."
        if sample_strategy == "uncovered" else ""
    )
    coverage_note = (
        " The system re-checks the \"other\" share on its own independent "
        "uniform probe when you finalize, so sampling selectively does not "
        "change when you are allowed to stop."
        if enforce_coverage else ""
    )
    # Multi-label changes the per-item reply shape (a list of categories) and
    # lets categories overlap; single-label keeps the pre-feature wording.
    if multi_label:
        reply_format = ('{"categories": [<taxonomy names; [] if none apply>], '
                        '"rationale": <≤2 sentences>}')
        overlap_clause = ""            # overlap is allowed, so drop "non-overlapping"
    else:
        reply_format = ('{"category": <one of the taxonomy names | "other">, '
                        '"rationale": <≤2 sentences>}')
        overlap_clause = ", non-overlapping"
    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        instruction=instruction.strip(),
        n_items=len(corpus),
        threshold=converge_below,
        probe_size=probe_size,
        max_iters=max_iterations,
        min_iters=min_iterations,
        size_aside=size_aside,
        focus_bullet=focus_bullet,
        uncovered_tool_line=uncovered_tool_line,
        coverage_note=coverage_note,
        reply_format=reply_format,
        overlap_clause=overlap_clause,
    )

    agent = create_react_agent(llm, tools, prompt=sys_prompt)
    cfg = {"recursion_limit": recursion_limit}
    kickoff = "Begin the analysis."
    seen_message_ids: set[str] = set()
    stream_error: Exception | None = None
    try:
        for event in agent.stream(
            {"messages": [{"role": "user", "content": kickoff}]},
            cfg,
            stream_mode="values",
        ):
            msgs = event.get("messages", [])
            if not msgs:
                continue
            # stream_mode="values" emits the full message list each step, so we
            # dedupe on message id to avoid double-counting usage.
            for m in msgs:
                mid = getattr(m, "id", None)
                if mid and mid not in seen_message_ids:
                    seen_message_ids.add(mid)
                    usage = getattr(m, "usage_metadata", None)
                    if usage:
                        # langchain-openai's standardised usage_metadata drops
                        # provider-specific fields like OpenRouter's `cost`,
                        # which lands in response_metadata.token_usage instead.
                        # Merge it back in so CostTracker can prefer native.
                        merged = dict(usage)
                        rmeta = getattr(m, "response_metadata", None) or {}
                        tu = (rmeta.get("token_usage") or {}) if isinstance(rmeta, dict) else {}
                        if isinstance(tu, dict) and tu.get("cost") is not None:
                            merged["cost"] = tu["cost"]
                        cost.add_orchestrator_usage(merged)
            last = msgs[-1]
            if hasattr(last, "pretty_print"):
                last.pretty_print()
            else:
                print(last)
            cost.write()  # refresh cost.json each agent step
    except Exception as e:
        stream_error = e
        logger.warning(f"[taxonomy_agent] orchestrator stream raised: {e!r} — flushing "
              f"partial state and exiting.")

    artifact_path = os.path.join(output_dir, "taxonomy.json")
    out: dict = {"run_id": run_id, "output_dir": output_dir, "artifact_path": artifact_path}
    if os.path.exists(artifact_path) and stream_error is None:
        with open(artifact_path) as f:
            out["artifact"] = json.load(f)
        out["status"] = "ok"
        logger.info(f"[taxonomy_agent] done → {artifact_path}")
    else:
        # The orchestrator may have walked off without calling finalize_classify
        # (e.g. monotonic-add loops that never trip the unmatched-rate threshold).
        # If we have a non-empty taxonomy on disk, label the corpus against it
        # ourselves so the run produces a usable artifact instead of returning
        # incomplete. The classify-budget floor is bypassed because we are
        # recovering an aborted run, not optimising a healthy one.
        auto_finalized = False
        try:
            recovered = force_finalize()
            if recovered is not None:
                out["artifact"] = recovered
                out["status"] = "ok"
                out["auto_finalized"] = True
                auto_finalized = True
                logger.info(f"[taxonomy_agent] orchestrator did not call "
                      f"finalize_classify; auto-finalized against the current "
                      f"on-disk taxonomy → {artifact_path}")
        except Exception as ff_err:
            logger.warning(f"[taxonomy_agent] auto-finalize fallback raised: {ff_err!r}")
        if not auto_finalized:
            out["status"] = "incomplete" if stream_error is None else "error"
            if stream_error is not None:
                out["error"] = repr(stream_error)
            logger.warning(f"[taxonomy_agent] WARNING: no artifact at {artifact_path}. "
                  f"Status={out['status']}. The orchestrator may have hit the "
                  f"recursion limit, the classify budget, or an LLM error "
                  f"mid-run, and the auto-finalize fallback could not produce "
                  f"a taxonomy either (likely an empty taxonomy state). "
                  f"Partial state is in {output_dir}/taxonomy_state.json and "
                  f"{output_dir}/classifications.jsonl.")

    if out.get("status") == "ok" and _mostly_judge_errors(out.get("artifact") or {}):
        out["status"] = "degraded"
        art = out["artifact"]
        logger.warning(f"[taxonomy_agent] WARNING: {art.get('n_judge_errors')}/"
              f"{art.get('n_items')} judge calls failed; the labels are "
              f"unreliable (status=degraded). Check the judge model id and "
              f"OPENROUTER_API_KEY.")

    _lab = (out.get("artifact") or {}).get("labeling") or {}
    _val = _lab.get("val_accuracy")
    if _val is not None:
        logger.info(f"[taxonomy_agent] labeling fidelity: {_val:.1%} agreement "
                    f"with the judge on {_lab.get('val_n')} held-out items")
        if _val < VAL_LOW_FIDELITY:
            logger.warning(f"[taxonomy_agent] WARNING: the cheap labels are only "
                  f"{_val:.1%} accurate on this corpus — the "
                  f"{_lab.get('n_cheap')} classifier-labelled items may be that "
                  f"noisy. Consider finalize='judge' or a lower coverage.")

    cost.write()
    cost_snapshot = cost.snapshot()
    out["cost"] = cost_snapshot
    meta["status"] = out["status"]
    meta["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta["cost"] = cost_snapshot
    atomic_write_json(meta_path, meta)

    if cost_snapshot["total_usd"] is not None:
        logger.info(f"[taxonomy_agent] cost: ${cost_snapshot['total_usd']:.4f} "
              f"(orch={cost_snapshot['orchestrator']['n_calls']} calls, "
              f"judge={cost_snapshot['judge']['n_calls']} calls)")
    else:
        logger.info(f"[taxonomy_agent] tokens recorded; USD unknown for one or both "
              f"models (not in cost.MODEL_PRICES). See {output_dir}/cost.json.")

    return RunResult(out)
