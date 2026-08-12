"""Six tools the orchestrator drives. Closure-bound to the item pool + a
persistent taxonomy, so the agent never has to pass the taxonomy by argument."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import tool

from .corpus import Corpus, InMemoryCorpus, _iter_jsonl, atomic_write_json


ESCAPE_HATCH_SUFFIX = (
    "\n\nIMPORTANT: If none of the listed categories applies to this item, "
    "reply with `\"category\": \"other\"`. Do not invent new category names — "
    "every reply must use either an exact name from the listed taxonomy or "
    "the literal string `other`."
)

# Multi-label variant: the judge returns every applicable category, or [].
ESCAPE_HATCH_SUFFIX_MULTI = (
    "\n\nIMPORTANT: Reply with a JSON array `\"categories\"` listing EVERY listed "
    "category that applies to this item; if none applies, reply with "
    "`\"categories\": []`. Do not invent names — use only exact names from the "
    "listed taxonomy."
)

# Sentinel rationale prefix for items where the judge call itself failed
# (network/HTTP/timeout after retry). These are NOT genuine misfits — they
# must be excluded from the unmatched rate, not folded into "other".
JUDGE_ERROR_RATIONALE = "[judge call failed]"

# Prefix `_coerce_category` stamps on the rationale when the judge returned a
# label outside the taxonomy. Named so the count of coerced rows has one home
# (see `is_coerced_rationale` / `summarize_rows`) instead of a repeated literal.
COERCED_RATIONALE_PREFIX = "[coerced from invented label"

# Prefix stamped on rows labelled cheaply by the classifier (not the
# judge); the classifier kind follows in the rationale. Distinct from the
# coerced/judge-error sentinels so classifier rows are never miscounted as either.
CLASSIFIER_RATIONALE_PREFIX = "[classifier:"

# Rationale stamped on the discovery-probe rows written when finalize="none"
# (discovery only): the taxonomy is returned without labelling the full corpus,
# and only the items already judged for free during discovery are recorded.
DISCOVERY_PROBE_RATIONALE = "[discovery probe; corpus not fully labelled]"

# finalize_classify labels the corpus one bounded batch of distinct items at a
# time so peak memory stays flat as the corpus grows: a million-item run never
# holds a million prompt strings (or reply strings) alive at once, only one
# chunk's worth. Small corpora finish in a single chunk, unchanged.
FINALIZE_CHUNK = 2000

# The embedding classifier labels the corpus one batch of this many items at a time, so only
# one batch of embedding vectors is ever resident — a 10M-item run never holds a
# 10M x dim array (which would be tens of GB).
EMBED_BATCH = 1024

# Self-validation: hold out a slice of the re-judged calibration (clean
# final-taxonomy labels), measure the trained classifier's agreement with the
# judge on it, and report that as the run's measured labeling fidelity. Only runs
# when there are enough re-judged items to make the estimate meaningful.
VAL_FRACTION = 0.2
VAL_MIN_REJUDGE = 40
VAL_CAP = 200
VAL_LOW_FIDELITY = 0.80          # below this, warn that cheap labels are noisy

# Above this corpus size, draw sample indices by rejection instead of building a
# full n-length index list — so sampling a corpus of millions stays O(k) memory,
# not O(n). Below it, the list is small enough that enumeration is simplest.
REJECTION_SAMPLE_MIN_N = 100_000

# `sample_uncovered` stops re-surfacing an item once it has been shown this many
# times while still uncovered — a perennial "other" is likely genuine noise
# (malformed row, off-axis outlier), not a missing category, so fixating on it
# would starve the frontier of fresh items.
UNCOVERED_MAX_RESURFACE = 3

# The classification instruction used when the caller has none of its own — the
# auto-finalize fallback and `refine()`'s re-classification. `finalize_classify`
# receives the orchestrator's own prompt instead.
DEFAULT_CLASSIFY_PROMPT = (
    "Pick the single category from the list that best describes the item. "
    "Reply only with a JSON object: "
    "{\"category\": <name or \"other\">, \"rationale\": <one or two sentences>}."
)

# Multi-label default: every applicable category, or [] if none fit.
DEFAULT_CLASSIFY_PROMPT_MULTI = (
    "List every category from the list that applies to the item. "
    "Reply only with a JSON object: "
    "{\"categories\": [<names, or empty if none apply>], "
    "\"rationale\": <one or two sentences>}."
)


def _format_item(item: dict, idx: int) -> str:
    """Render one item for the judge: id, then any non-text/id metadata, then the text."""
    lines = [f"### Item {idx} (id={item['id']})"]
    for k, v in item.items():
        if k in ("id", "text"):
            continue
        lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append(str(item.get("text", "")))
    return "\n".join(lines)


def _format_taxonomy(tax: list[dict]) -> str:
    if not tax:
        return "(empty — no categories yet)"
    return "\n".join(f"- **{c.get('name')}**: {c.get('description')}" for c in tax)


def _parse_json_block(text: str | None) -> Any:
    """Best-effort extract a JSON value from a judge reply.

    Tries, in order: the whole reply, the contents of the first ```json``` fence,
    and finally a scan that calls `JSONDecoder.raw_decode` at every `[`/`{` start
    position — that handles replies of the form `"sure, here you go: {...} hope
    that helps"` without grabbing trailing prose.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    decoder = json.JSONDecoder()
    for i, c in enumerate(text):
        if c in "[{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def _coerce_category(parsed: Any, taxonomy: list[dict]) -> tuple[str, str]:
    """Map judge replies to (category, rationale). Out-of-taxonomy labels collapse to 'other'."""
    # Case-insensitive lookup keyed on lowercased name → canonical name, so a
    # judge reply of "Topic_A" against taxonomy ["topic_a"] still matches.
    lookup = {c["name"].lower(): c["name"] for c in taxonomy}
    if not isinstance(parsed, dict):
        return "other", "[unparseable judge reply]"
    raw = str(parsed.get("category", "other")).strip()
    rat = str(parsed.get("rationale", ""))
    canonical = lookup.get(raw.lower())
    if canonical is not None:
        return canonical, rat
    if raw.lower() == "other":
        return "other", rat
    return "other", f"{COERCED_RATIONALE_PREFIX} '{raw}'] {rat}"


def _coerce_categories(parsed: Any, taxonomy: list[dict]) -> tuple[list[str], str]:
    """Multi-label counterpart of `_coerce_category`. Reads a `categories` list
    (falling back to a single `category` for robustness), keeps only exact
    taxonomy names (case-insensitive, de-duplicated, order preserved), and drops
    `other` and invented labels. Returns (categories, rationale); an empty list
    means the item fits no category. If the judge named only invented labels, the
    rationale is stamped coerced so the count of misfits stays honest."""
    lookup = {c["name"].lower(): c["name"] for c in taxonomy}
    if not isinstance(parsed, dict):
        return [], "[unparseable judge reply]"
    rat = str(parsed.get("rationale", ""))
    raw = parsed.get("categories")
    if not isinstance(raw, list):
        one = parsed.get("category")
        raw = [one] if one is not None else []
    out: list[str] = []
    saw_invented = False
    for r in raw:
        name = str(r).strip()
        canon = lookup.get(name.lower())
        if canon is not None:
            if canon not in out:
                out.append(canon)
        elif name and name.lower() != "other":
            saw_invented = True
    if not out and saw_invented:
        return [], f"{COERCED_RATIONALE_PREFIX} '{raw}'] {rat}"
    return out, rat


def is_coerced_rationale(rationale: str) -> bool:
    """True if `_coerce_category` stamped this rationale as an out-of-taxonomy
    (coerced) label. Single home for the sentinel-prefix check."""
    return isinstance(rationale, str) and rationale.startswith(COERCED_RATIONALE_PREFIX)


def build_classify_prompt(instruction: str, tax_str: str, item: dict) -> str:
    """The judge prompt that labels one `item` against the taxonomy rendered in
    `tax_str`. `instruction` already carries any escape-hatch suffix. Shared by
    the discovery-loop classifiers and `refine()` so the layout stays identical."""
    return (f"{instruction}\n\n## Categories\n{tax_str}\n\n"
            f"## Item to classify\n{_format_item(item, 1)}")


def _content_hash(item: dict) -> str:
    """Stable hash of an item's content (every field except its id), so
    identical items collapse to one judge call. Hashing keeps the dedup key
    small (~40 bytes) even for a corpus of millions. The id is a reference in
    the prompt only and never affects the label."""
    blob = json.dumps({k: v for k, v in item.items() if k != "id"},
                      sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _label_reply(rep: str | None, taxonomy: list[dict]) -> tuple[str, str]:
    """Map a judge reply to `(category, rationale)`: a failed call (`rep is
    None`) becomes `("other", JUDGE_ERROR_RATIONALE)`; any other reply is parsed
    and coerced to the taxonomy. Shared by every judge-labelling site."""
    if rep is None:
        return "other", JUDGE_ERROR_RATIONALE
    return _coerce_category(_parse_json_block(rep), taxonomy)


def _group_by_content(indexed_items) -> list[list[int]]:
    """Group `(corpus_index, item)` pairs by item content so identical items are
    judged once. Callers pass a sequential scan (e.g. `enumerate(corpus)`, or a
    filtered generator) so a file-backed corpus is read straight through rather
    than seeked per item. Returns one list of indices per distinct-content
    group, in first-appearance order."""
    groups: dict[str, list[int]] = {}
    for i, it in indexed_items:
        groups.setdefault(_content_hash(it), []).append(i)
    return list(groups.values())


def _rejection_sample_indices(rng, n, k, is_excluded, n_excluded):
    """Draw `k` distinct indices in `[0, n)` for which `is_excluded(i)` is
    False, by rejection sampling — so a huge corpus never materializes an
    n-length index pool. Returns `None` (telling the caller to use its own
    enumerate-then-sample path) unless the corpus is large AND the excluded
    items are a minority, the only regime where rejection both matters and
    stays cheap (each draw is accepted with probability > 1/2, so the expected
    work is O(k)). The caller must guarantee at least `k` eligible indices.
    Deterministic given `rng`."""
    if n <= REJECTION_SAMPLE_MIN_N or n_excluded > n // 2:
        return None
    chosen, picked = [], set()
    while len(chosen) < k:
        i = rng.randrange(n)
        if i in picked:
            continue
        picked.add(i)
        if not is_excluded(i):
            chosen.append(i)
    return chosen


class _CountRollup:
    """Incremental rollup of classification rows into `(category_counts,
    n_coerced, n_judge_errors)` without holding the rows — the single owner of
    what each rationale sentinel means for the counts. `add` weights by `n`, so
    a content-deduped group of identical items counts once per duplicate."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.n_coerced = 0
        self.n_judge_errors = 0

    def add(self, category: str, rationale: str, n: int = 1) -> None:
        self.counts[category] = self.counts.get(category, 0) + n
        if rationale == JUDGE_ERROR_RATIONALE:
            self.n_judge_errors += n
        elif is_coerced_rationale(rationale):
            self.n_coerced += n

    def add_multi(self, categories: list[str], rationale: str, n: int = 1) -> None:
        """Multi-label count: increment every assigned category (or `other` when
        the list is empty). Coerced / judge-error sentinels count once per item,
        so category_counts may exceed n_items but n_coerced/n_judge_errors do not."""
        if categories:
            for c in categories:
                self.counts[c] = self.counts.get(c, 0) + n
        else:
            self.counts["other"] = self.counts.get("other", 0) + n
        if rationale == JUDGE_ERROR_RATIONALE:
            self.n_judge_errors += n
        elif is_coerced_rationale(rationale):
            self.n_coerced += n


def summarize_rows(rows: list[dict]) -> tuple[dict, int, int]:
    """Roll classification rows up into `(category_counts, n_coerced,
    n_judge_errors)` by inspecting each row's category and rationale sentinel."""
    roll = _CountRollup()
    for r in rows:
        roll.add(r["category"], r.get("rationale", ""))
    return roll.counts, roll.n_coerced, roll.n_judge_errors


def build_artifact_from_counts(run_id: str, taxonomy: list[dict],
                               final_prompt: str, *, n_items: int,
                               category_counts: dict, n_coerced: int,
                               n_judge_errors: int) -> dict:
    """Assemble the `taxonomy.json` summary artifact from already-rolled-up
    counts. The per-item rows are deliberately NOT embedded — they live in
    `classifications.jsonl` — so the artifact stays O(number of categories)
    even for a million-item corpus and can be read without loading every row.
    Single owner of the artifact schema."""
    return {
        "run_id": run_id,
        "n_items": n_items,
        "n_coerced": n_coerced,
        "n_judge_errors": n_judge_errors,
        "taxonomy": taxonomy,
        "final_prompt": final_prompt,
        "category_counts": category_counts,
    }


def build_artifact(run_id: str, rows: list[dict], taxonomy: list[dict],
                   final_prompt: str) -> dict:
    """Summarize classification `rows` into the `taxonomy.json` artifact.
    Convenience wrapper over `build_artifact_from_counts` for callers that
    already hold every row in memory (`refine()`); the rows themselves are
    persisted separately to `classifications.jsonl`."""
    counts, n_coerced, n_judge_errors = summarize_rows(rows)
    return build_artifact_from_counts(
        run_id, taxonomy, final_prompt, n_items=len(rows),
        category_counts=counts, n_coerced=n_coerced,
        n_judge_errors=n_judge_errors)


def write_taxonomy_state(path: str, taxonomy: list[dict],
                         n_classify_calls: int = 0) -> None:
    """Persist the working taxonomy + classify-call count to `taxonomy_state.json`."""
    atomic_write_json(path, {"taxonomy": taxonomy,
                             "n_classify_calls": n_classify_calls})


def _append_trace(trace_path: str, run_id: str, kind: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    with open(trace_path, "a") as f:
        f.write(json.dumps({"run_id": run_id, "kind": kind, **payload}) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Per-run state and the taxonomy-revision op handlers.
#
# Each op handler is a pure (taxonomy, op_dict) → (new_taxonomy, log_entry)
# function — no closure state, easy to unit-test, and "add a new op" is just
# "write a function and register it in `_OPS`". All handlers must validate
# inputs BEFORE constructing a new tax list, so a partially-formed op never
# loses data (see `_op_merge` and `_op_split`).
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _TaxonomyState:
    """Per-run mutable state shared across the six tools via closure."""
    # Working taxonomy. Reassigned (not mutated in place) by `_apply_ops`.
    taxonomy: list[dict] = field(default_factory=list)
    # Corpus indices handed out by `sample_items` so far. Used to bias
    # subsequent probes toward unseen items; cleared when the pool is exhausted.
    sampled_idx: set = field(default_factory=set)
    # Snapshot of the taxonomy at the moment `finalize_classify` last ran.
    # `finalize` refuses to repeat work until the taxonomy actually changes.
    finalized_at: list[dict] | None = None
    # Counter against the per-run classify budget (see `make_tools`).
    classify_calls: int = 0
    # Item id -> judge label from the discovery probes (last write wins), reused
    # for free as classifier calibration. Kept across ALL discovery iterations:
    # restricting to the final-taxonomy version was measured to LOWER fidelity
    # (intermediate labels are mostly still correct; more examples beat recency).
    probe_labels: dict = field(default_factory=dict)
    # Item id -> times `sample_uncovered` has surfaced it while still "other".
    # Caps fixation on perennial misfits (see UNCOVERED_MAX_RESURFACE).
    frontier_shown: dict = field(default_factory=dict)
    # Set by the coverage backstop when it lets a finalize through despite a
    # high unmatched rate (budget exhausted): the rate it measured, else None.
    low_coverage: float | None = None
    # force_finalize sets this so aborted-run recovery bypasses the coverage
    # backstop (mirrors how it bypasses the min_iterations floor).
    skip_coverage_check: bool = False


_OpHandler = Callable[[list[dict], dict], tuple[list[dict], dict]]

MAX_NAME_LEN = 40
MAX_DESC_LEN = 200
_RESERVED_NAMES = {"other"}       # reserved for the escape hatch / unmatched bucket


def _clean_category_name(raw) -> str | None:
    """Normalize a proposed category name to safe snake_case, or `None` if it
    can't be one (empty, or the reserved `other`). Names can originate in the
    judge's novelty proposals, which are produced from untrusted corpus text, so
    this collapses everything outside `[a-z0-9]` to `_` and truncates — an
    injected newline or instruction embedded in a name cannot then survive into
    later prompts or the persisted taxonomy. Case-folding also makes
    `Topic_A`/`topic_a` collapse to one name instead of one shadowing the
    other."""
    if not isinstance(raw, str):
        return None
    name = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")[:MAX_NAME_LEN]
    name = name.strip("_")
    if not name or name in _RESERVED_NAMES:
        return None
    return name


def _clean_description(raw) -> str:
    """Collapse a category description to a single bounded line, so a multi-line
    `SYSTEM: ...` payload injected via a description can't reshape later
    prompts."""
    if not isinstance(raw, str):
        return ""
    return " ".join(raw.split())[:MAX_DESC_LEN]


def _op_add(tax: list[dict], op: dict) -> tuple[list[dict], dict]:
    name = _clean_category_name(op["name"])
    if name is None:
        return tax, {"op": "add", "name": op.get("name"),
                     "result": "rejected (invalid or reserved name)"}
    desc = _clean_description(op["description"])
    if any(c["name"] == name for c in tax):
        return tax, {"op": "add", "name": name, "result": "skipped (already exists)"}
    return tax + [{"name": name, "description": desc}], {
        "op": "add", "name": name, "result": "ok",
    }


def _op_rename(tax: list[dict], op: dict) -> tuple[list[dict], dict]:
    old = op["old_name"]
    new = _clean_category_name(op["new_name"])
    if new is None:
        return tax, {"op": "rename", "from": old, "to": op.get("new_name"),
                     "result": "rejected (invalid or reserved new name)"}
    if not any(c["name"] == old for c in tax):
        return tax, {"op": "rename", "from": old, "to": new, "result": "missing source"}
    if any(c["name"] == new for c in tax):
        return tax, {"op": "rename", "from": old, "to": new,
                     "result": "target name already exists"}
    new_tax = [{**c, "name": new} if c["name"] == old else dict(c) for c in tax]
    return new_tax, {"op": "rename", "from": old, "to": new, "result": "ok"}


def _op_edit(tax: list[dict], op: dict) -> tuple[list[dict], dict]:
    name = op["name"]
    if not any(c["name"] == name for c in tax):
        return tax, {"op": "edit", "name": name, "result": "missing"}
    desc = _clean_description(op["description"])
    new_tax = [{**c, "description": desc} if c["name"] == name else dict(c) for c in tax]
    return new_tax, {"op": "edit", "name": name, "result": "ok"}


def _op_drop(tax: list[dict], op: dict) -> tuple[list[dict], dict]:
    name = op["name"]
    new_tax = [c for c in tax if c["name"] != name]
    return new_tax, {"op": "drop", "name": name,
                     "result": "ok" if len(new_tax) < len(tax) else "missing"}


def _op_merge(tax: list[dict], op: dict) -> tuple[list[dict], dict]:
    into = _clean_category_name(op["into"])
    if into is None:
        return tax, {"op": "merge", "into": op.get("into"),
                     "result": "rejected (invalid or reserved target name)"}
    sources = op.get("from", []) or []
    desc = _clean_description(op["description"]) if op.get("description") else None
    target_exists = any(c["name"] == into for c in tax)
    # Validate before any deletion — bug #1 was that sources got removed first.
    if not target_exists and not desc:
        return tax, {"op": "merge", "into": into,
                     "result": "missing target and no description (no changes applied)"}
    # Exclude `into` from the source list so a self-merge can't delete the target.
    sources_present = [s for s in sources
                       if s != into and any(c["name"] == s for c in tax)]
    new_tax = [c for c in tax if c["name"] not in sources_present]
    if not target_exists:
        new_tax = new_tax + [{"name": into, "description": desc}]
    elif desc:
        new_tax = [{**c, "description": desc} if c["name"] == into else dict(c)
                   for c in new_tax]
    return new_tax, {
        "op": "merge", "into": into, "from": sources_present,
        "result": "ok" if sources_present else "no source categories matched",
    }


def _op_split(tax: list[dict], op: dict) -> tuple[list[dict], dict]:
    src = op["from"]
    new_cats = op.get("into", []) or []
    if not any(c["name"] == src for c in tax):
        return tax, {"op": "split", "from": src, "result": "missing source"}
    # Validate before deletion — bug #2 was that source got removed first.
    if not new_cats:
        return tax, {"op": "split", "from": src,
                     "result": "no replacement categories provided (no changes applied)"}
    if any(not (isinstance(nc, dict) and "name" in nc and "description" in nc)
           for nc in new_cats):
        return tax, {"op": "split", "from": src,
                     "result": "malformed entry in 'into' (no changes applied)"}
    new_tax = [c for c in tax if c["name"] != src]
    added: list[str] = []
    for nc in new_cats:
        cname = _clean_category_name(nc["name"])
        if cname is None or any(c["name"] == cname for c in new_tax):
            continue
        new_tax = new_tax + [{"name": cname,
                              "description": _clean_description(nc["description"])}]
        added.append(cname)
    return new_tax, {"op": "split", "from": src, "into": added, "result": "ok"}


_OPS: dict[str, _OpHandler] = {
    "add": _op_add,
    "rename": _op_rename,
    "edit": _op_edit,
    "drop": _op_drop,
    "merge": _op_merge,
    "split": _op_split,
}


def _apply_ops(state: _TaxonomyState,
               operations: list[dict]) -> tuple[list[dict], list[dict]]:
    """Apply `operations` in order against `state.taxonomy`, dispatching each
    via `_OPS`. Per-op failures (unknown op, missing required key) are recorded
    in the log and don't halt the rest of the batch — the orchestrator can read
    the log and decide whether to retry or move on."""
    tax = [dict(c) for c in state.taxonomy]
    log: list[dict] = []
    for op_dict in operations:
        op = (op_dict or {}).get("op")
        handler = _OPS.get(op)
        if handler is None:
            log.append({"op": op, "result": f"unknown op '{op}'"})
            continue
        try:
            tax, entry = handler(tax, op_dict)
        except KeyError as e:
            log.append({"op": op, "result": f"missing required key {e}"})
            continue
        log.append(entry)
    state.taxonomy = tax
    return tax, log


def _apply_ops_loose(state: _TaxonomyState,
                     operations: list[dict]) -> tuple[list[dict], list[dict]]:
    """ABLATION: skip every validate-before-mutate check.

    No source-existence check on rename/edit/merge/split; no name-collision
    check on add/rename; no malformed-entry guard on split; missing required
    keys default to empty strings instead of skipping the op. Used by the
    ``--prose-revise`` ablation to test what the typed dispatcher actually
    buys.
    """
    tax = [dict(c) for c in state.taxonomy]
    log: list[dict] = []
    for op_dict in operations or []:
        d = op_dict or {}
        op = d.get("op")
        if op == "add":
            tax.append({"name": d.get("name", ""),
                        "description": d.get("description", "")})
            log.append({"op": "add", "name": d.get("name", ""), "result": "ok"})
        elif op == "rename":
            old = d.get("old_name", "")
            new = d.get("new_name", "")
            tax = [{**c, "name": new} if c["name"] == old else dict(c)
                   for c in tax]
            log.append({"op": "rename", "from": old, "to": new, "result": "ok"})
        elif op == "edit":
            name = d.get("name", "")
            desc = d.get("description", "")
            tax = [{**c, "description": desc} if c["name"] == name else dict(c)
                   for c in tax]
            log.append({"op": "edit", "name": name, "result": "ok"})
        elif op == "drop":
            name = d.get("name", "")
            tax = [c for c in tax if c["name"] != name]
            log.append({"op": "drop", "name": name, "result": "ok"})
        elif op == "merge":
            into = d.get("into", "")
            sources = d.get("from", []) or []
            desc = d.get("description")
            tax = [c for c in tax if c["name"] not in sources]
            if not any(c["name"] == into for c in tax):
                tax.append({"name": into,
                            "description": desc or ""})
            elif desc:
                tax = [{**c, "description": desc} if c["name"] == into
                       else dict(c) for c in tax]
            log.append({"op": "merge", "into": into, "from": sources,
                        "result": "ok"})
        elif op == "split":
            src = d.get("from", "")
            new_cats = d.get("into", []) or []
            tax = [c for c in tax if c["name"] != src]
            for nc in new_cats:
                tax.append({"name": (nc or {}).get("name", ""),
                            "description": (nc or {}).get("description", "")})
            log.append({"op": "split", "from": src, "result": "ok"})
        else:
            log.append({"op": op, "result": f"unknown op '{op}'"})
            continue
    state.taxonomy = tax
    return tax, log


def make_tools(items, run_id: str, output_dir: str,
               judge,
               concurrency: int = 8, seed: int = 42, max_iters: int = 10,
               min_iterations: int = 0, prose_revise: bool = False,
               initial_taxonomy: list[dict] | None = None,
               finalize_mode: str = "judge", coverage: float = 0.85,
               embed_model: str = "all-MiniLM-L6-v2", embed_fn=None,
               calibration_size: int = 0,
               finetune_model: str = "distilbert-base-uncased",
               finetune_epochs: int = 4,
               classify_max_tokens: int = 300,
               sample_strategy: str = "uniform",
               enforce_coverage: bool = False,
               converge_below: float = 0.10,
               probe_size: int = 20,
               multi_label: bool = False):
    """Construct the discovery tools, sharing state via closure.

    The taxonomy lives entirely inside the closure — the orchestrator mutates
    it through `revise_taxonomy` and reads it via `get_taxonomy`.

    `sample_strategy="uncovered"` adds a seventh tool, `sample_uncovered`, that
    preferentially surfaces items the taxonomy has not placed (past "other"
    labels). The default "uniform" returns exactly the six original tools in the
    original order, so an unchanged run reproduces prior behaviour byte-for-byte.

    `enforce_coverage=True` turns `finalize_classify`'s stop rule into a code
    check: it re-measures the unmatched rate on a fresh uniform-random probe
    (independent of whatever the orchestrator chose to classify) and refuses to
    finalize above `converge_below` while classify budget remains. `probe_size`
    sizes that probe. Both default off so behaviour is unchanged.

    `min_iterations` is a floor on the number of `classify_with_judge` calls
    required before `finalize_classify` is allowed — guards against premature
    convergence on a lucky early probe. 0 means no floor (used at the tool
    layer in tests). `run()` defaults this to 3.

    `initial_taxonomy` seeds the working taxonomy so the orchestrator starts
    from an existing category set (used by `refine()` to warm-start from a prior
    run) instead of the empty default.

    `items` may be a list of item dicts or a `Corpus` (e.g. a file-backed
    `JsonlCorpus`); a list is wrapped so the tools only ever touch the corpus
    through its length / index / id-lookup / iteration interface, never a
    materialized dict of every item."""
    corpus = items if isinstance(items, Corpus) else InMemoryCorpus(items)
    rng = random.Random(seed)
    # Cap classify_with_judge calls so a runaway orchestrator can't loop past
    # max_iters. The recommended loop runs ~2 classify calls per iteration
    # (probe + reverify), with one extra for the final convergence check —
    # 3× max_iters gives headroom; the floor of 8 keeps smoke tests usable.
    classify_budget = max(8, 3 * max_iters)
    state = _TaxonomyState()
    if initial_taxonomy:
        state.taxonomy = [dict(c) for c in initial_taxonomy]

    trace_path = os.path.join(output_dir, "trace.jsonl")
    artifact_path = os.path.join(output_dir, "taxonomy.json")
    state_path = os.path.join(output_dir, "taxonomy_state.json")
    classifications_jsonl = os.path.join(output_dir, "classifications.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    def _write_taxonomy_state() -> None:
        """Persist the current working taxonomy so a crashed run still has the
        latest categories on disk, not just buried in trace.jsonl."""
        write_taxonomy_state(state_path, state.taxonomy, state.classify_calls)

    def _judge_index_replies(indices, tax_str, hardened):
        """Judge the given corpus `indices` in FINALIZE_CHUNK-sized batches,
        yielding `(index, reply)` as each batch returns. The single spine shared
        by finalize's calibration re-judge, the embed/finetune tail, and the judge path."""
        for s in range(0, len(indices), FINALIZE_CHUNK):
            chunk = indices[s:s + FINALIZE_CHUNK]
            prompts = [build_classify_prompt(hardened, tax_str, corpus[i])
                       for i in chunk]
            replies = judge.parallel(prompts, concurrency=concurrency * 2,
                                     max_tokens=classify_max_tokens)
            yield from zip(chunk, replies)

    def _draw_unseen(k: int) -> tuple[list[int], str]:
        """Pick up to k corpus indices not handed out yet (uniform random),
        marking them seen. Resets the history with a note when the unseen pool
        is too small. The single uniform-draw primitive shared by sample_items,
        sample_uncovered's top-up, and the coverage probe.

        Sample by index, not by scanning every item, so a file-backed corpus
        only reads the rows it hands out — and on a huge corpus we draw by
        rejection rather than materializing an n-length unseen list."""
        n = len(corpus)
        k = max(1, min(int(k), n))
        note = ""
        if n - len(state.sampled_idx) < k:
            note = (f" (pool of {n} exhausted — sampling history reset; "
                    f"expect overlap with prior probes)")
            state.sampled_idx = set()
        seen = state.sampled_idx
        chosen = _rejection_sample_indices(rng, n, k, seen.__contains__, len(seen))
        if chosen is None:
            chosen = rng.sample([i for i in range(n) if i not in seen], k)
        state.sampled_idx.update(chosen)
        return chosen, note

    @tool
    def sample_items(k: int) -> str:
        """Return K items from the corpus. Default K = 20.

        Items returned by earlier calls are not repeated until the corpus is
        exhausted. At that point the history resets and the reply notes the
        wraparound; subsequent batches will overlap with prior ones."""
        chosen, note = _draw_unseen(k)
        sampled = [corpus[i] for i in chosen]
        ids = [it["id"] for it in sampled]
        blocks = [_format_item(it, i) for i, it in enumerate(sampled, start=1)]
        return (
            f"Sampled {len(chosen)} items{note}.\n"
            f"item_ids = {json.dumps(ids)}\n\n"
            + "\n\n".join(blocks)
        )

    @tool
    def sample_uncovered(k: int) -> str:
        """Return up to K items the taxonomy does not yet cover. Default K = 20.

        Prefers items a prior `classify_with_judge` labelled "other", pooled
        across ALL past probes (not just the last batch), then fills the rest
        with fresh unseen items. Feed the result to
        `propose_novelties_with_judge` to grow the taxonomy, then re-classify
        the same ids after revising to confirm the new categories absorb them.

        An item that stays "other" across several probes is dropped from the
        pool as likely noise rather than re-shown forever."""
        k = max(1, min(int(k), len(corpus)))
        # Frontier = items last judged "other". probe_labels is last-write-wins,
        # so anything since re-covered has already dropped out; we only skip
        # perennial misfits (shown UNCOVERED_MAX_RESURFACE times, likely noise).
        frontier = [iid for iid, lab in state.probe_labels.items()
                    if lab == "other"
                    and state.frontier_shown.get(iid, 0) < UNCOVERED_MAX_RESURFACE]
        rng.shuffle(frontier)
        frontier = frontier[:k]
        for iid in frontier:
            state.frontier_shown[iid] = state.frontier_shown.get(iid, 0) + 1
        out = [it for it in (corpus.get(iid) for iid in frontier) if it is not None]
        n_frontier = len(out)
        note = ""
        if n_frontier < k:
            chosen, note = _draw_unseen(k - n_frontier)
            out += [corpus[i] for i in chosen]
        ids = [it["id"] for it in out]
        blocks = [_format_item(it, i) for i, it in enumerate(out, start=1)]
        return (
            f"Sampled {len(out)} items "
            f"({n_frontier} known-uncovered + {len(out) - n_frontier} fresh)"
            f"{note}.\n"
            f"item_ids = {json.dumps(ids)}\n\n"
            + "\n\n".join(blocks)
        )

    @tool
    def get_taxonomy() -> str:
        """Return the current working taxonomy as JSON."""
        return json.dumps(state.taxonomy, indent=2)

    @tool
    def revise_taxonomy(operations: list[dict]) -> str:
        """Apply edits to the working taxonomy. Changes persist across calls.

        Each operation is a dict; pass a list of them. Allowed forms:
          {"op": "add",    "name": <snake_case>, "description": <one sentence>}
          {"op": "rename", "old_name": <existing>, "new_name": <new>}
          {"op": "edit",   "name": <existing>, "description": <new sentence>}
          {"op": "merge",  "into": <target (existing or new)>, "from": [<names to absorb>],
                           "description": <required only if `into` is new>}
          {"op": "split",  "from": <existing>, "into": [{"name", "description"}, ...]}
          {"op": "drop",   "name": <existing>}

        Operations apply in order. Returns a per-op result log and the resulting taxonomy."""
        _ops_fn = _apply_ops_loose if prose_revise else _apply_ops
        new_tax, applied = _ops_fn(state, operations)
        _append_trace(trace_path, run_id, "revise", {
            "operations": operations,
            "applied": applied,
            "taxonomy_after": new_tax,
        })
        _write_taxonomy_state()
        return json.dumps({"applied": applied, "taxonomy": new_tax}, indent=2)

    @tool
    def classify_with_judge(item_ids: list[str], classify_prompt: str) -> str:
        """Ask the judge to label each item you pass in against the current
        taxonomy. (Call `revise_taxonomy` first if you want to change it.)

        `classify_prompt` MUST tell the judge to reply ONLY with a JSON object:
          {"category": <name | "other">, "rationale": <≤2 sentences>}.

        Returns per-item labels plus the share of items labelled "other"."""
        if state.classify_calls >= classify_budget:
            return (f"ERROR: classify_with_judge budget exhausted "
                    f"({classify_budget} calls — you set max_iters={max_iters} "
                    f"and the budget is 3× that). Call finalize_classify with "
                    f"the current taxonomy now, or stop.")
        deduped_ids = list(dict.fromkeys(item_ids))
        sel = [it for it in (corpus.get(i) for i in deduped_ids) if it is not None]
        if not sel:
            return "ERROR: no valid item_ids."
        taxonomy = state.taxonomy
        if not taxonomy:
            return "ERROR: taxonomy is empty. Call revise_taxonomy(add ...) first."
        state.classify_calls += 1
        tax_str = _format_taxonomy(taxonomy)
        suffix = ESCAPE_HATCH_SUFFIX_MULTI if multi_label else ESCAPE_HATCH_SUFFIX
        hardened = classify_prompt.strip() + suffix
        prompts = [build_classify_prompt(hardened, tax_str, it) for it in sel]
        replies = judge.parallel(prompts, concurrency=concurrency,
                                 max_tokens=classify_max_tokens)
        results = []
        n_other = 0
        n_coerced = 0
        n_judge_errors = 0
        for it, rep in zip(sel, replies):
            if rep is None:
                n_judge_errors += 1
                row = {"item_id": it["id"], "category": "other",
                       "rationale": JUDGE_ERROR_RATIONALE}
                if multi_label:
                    row["categories"] = []
                results.append(row)
                continue
            parsed = _parse_json_block(rep)
            if multi_label:
                cats, rat = _coerce_categories(parsed, taxonomy)
                if is_coerced_rationale(rat):
                    n_coerced += 1
                if not cats:                    # matches no category = uncovered
                    n_other += 1
                results.append({"item_id": it["id"],
                                "category": cats[0] if cats else "other",
                                "categories": cats, "rationale": rat[:400]})
            else:
                cat, rat = _coerce_category(parsed, taxonomy)
                if is_coerced_rationale(rat):
                    n_coerced += 1
                if cat == "other":
                    n_other += 1
                results.append({"item_id": it["id"], "category": cat,
                                "rationale": rat[:400]})
        n_scored = len(sel) - n_judge_errors
        # Keep each probe's judge label as free calibration for a classifier
        # finalize (skip failed calls). Labels are against the taxonomy at this
        # moment; the classifier path filters to categories that survive to the end.
        for r in results:
            if r["rationale"] != JUDGE_ERROR_RATIONALE:
                state.probe_labels[r["item_id"]] = r["category"]
        # No successful classifications means the rate carries no signal; report
        # 1.0 (fully unfit) so a total judge failure never reads as convergence.
        rate = (n_other / n_scored) if n_scored > 0 else 1.0
        _append_trace(trace_path, run_id, "classify", {
            "taxonomy_snapshot": taxonomy, "results": results,
            "dont_fit_rate": rate, "n_coerced": n_coerced,
            "n_judge_errors": n_judge_errors,
        })
        return json.dumps({
            "n_classified": n_scored,
            "n_judge_errors": n_judge_errors,
            "dont_fit_rate": round(rate, 3),
            "results": results,
        }, indent=2)

    @tool
    def propose_novelties_with_judge(item_ids: list[str], novelty_prompt: str) -> str:
        """Ask the judge to examine items the taxonomy does not yet cover
        (typically the ones a prior `classify_with_judge` call labelled "other")
        and suggest new categories. The judge sees the current taxonomy so it
        will not repeat existing names.

        Items are sent to the judge in batches of 20, so a long list cannot
        overflow the judge's context. Proposals from each batch are merged
        and deduplicated by name before returning.

        `novelty_prompt` MUST tell the judge to reply ONLY with a JSON list of:
          {"name": <snake_case>, "description": <one sentence>}.

        Returns the merged list (or an error string if every batch fails). Call
        `revise_taxonomy` afterwards to adopt any of the suggestions."""
        deduped_ids = list(dict.fromkeys(item_ids))
        sel = [it for it in (corpus.get(i) for i in deduped_ids) if it is not None]
        if not sel:
            return "ERROR: no valid item_ids."
        taxonomy = state.taxonomy
        tax_str = _format_taxonomy(taxonomy)
        BATCH = 20
        batches = [sel[i:i + BATCH] for i in range(0, len(sel), BATCH)]
        prompts = []
        for batch in batches:
            section = "\n\n".join(_format_item(it, i) for i, it in enumerate(batch, start=1))
            prompts.append(
                f"{novelty_prompt.strip()}\n\n## Existing categories\n{tax_str}\n\n"
                f"## Items to inspect\n{section}"
            )
        replies = judge.parallel(prompts, concurrency=concurrency, max_tokens=900)

        proposals: list[dict] = []
        seen_names: set[str] = set()
        n_judge_errors = 0
        n_unparseable = 0
        for rep in replies:
            if rep is None:
                n_judge_errors += 1
                continue
            parsed = _parse_json_block(rep)
            if not isinstance(parsed, list):
                n_unparseable += 1
                continue
            for p in parsed:
                if (isinstance(p, dict) and "name" in p and "description" in p
                        and p["name"] not in seen_names):
                    seen_names.add(p["name"])
                    proposals.append({"name": p["name"], "description": p["description"]})
        _append_trace(trace_path, run_id, "novelties", {
            "n_batches": len(batches),
            "n_judge_errors": n_judge_errors,
            "n_unparseable_batches": n_unparseable,
            "proposed": proposals,
        })
        if not proposals:
            raw_glimpse = "\n---\n".join(str(r)[:500] for r in replies if r) or "(all judge calls failed)"
            return (f"Could not extract any novelties across {len(batches)} batch(es) "
                    f"(judge_errors={n_judge_errors}, unparseable={n_unparseable}). "
                    f"Raw replies:\n{raw_glimpse}")
        return json.dumps(proposals, indent=2)

    def _classifier_finalize(final_prompt: str) -> str:
        """finalize_mode in ('embed', 'finetune'): train a cheap classifier on a
        judge-labelled calibration set (discovery probes + an optional fresh
        re-judge), label the confident majority with it, and route only the
        low-confidence tail to the judge. Produces the same compact
        taxonomy.json + streamed classifications.jsonl as the judge path, so
        everything downstream is
        unchanged."""
        from . import embedding as _emb
        from .classifiers import make_classifier
        taxonomy = state.taxonomy
        tax_names = [c["name"] for c in taxonomy]
        descriptions = {c["name"]: c.get("description", "") for c in taxonomy}
        valid = set(tax_names) | {"other"}
        tax_str = _format_taxonomy(taxonomy)
        hardened = final_prompt.strip() + ESCAPE_HATCH_SUFFIX

        # Calibration labels: discovery probes (kept for surviving categories,
        # all iterations) plus an optional fresh re-judge of unlabelled items
        # against the FINAL taxonomy — clean labels that lift a trained
        # classifier, at calibration_size judge calls.
        cal: dict[str, str] = {}
        for iid, cat in state.probe_labels.items():
            if cat in valid:
                cal[iid] = cat
        n_probe = len(cal)
        rejudged_ids: list[str] = []            # clean final-taxonomy labels
        if calibration_size > 0:
            rng_c = random.Random(seed)
            # Every id in `cal` is a distinct corpus id, so exactly
            # len(corpus) - len(cal) items are eligible to re-judge.
            target = min(calibration_size, len(corpus) - len(cal))
            picked = _rejection_sample_indices(
                rng_c, len(corpus), target,
                lambda i: corpus.id_at(i) in cal, len(cal))
            if picked is None:                  # small corpus: enumerate + shuffle
                order = list(range(len(corpus)))
                rng_c.shuffle(order)
                picked = []
                for i in order:
                    if corpus.id_at(i) not in cal:
                        picked.append(i)
                        if len(picked) >= calibration_size:
                            break
            for i, rep in _judge_index_replies(picked, tax_str, hardened):
                cat, rat = _label_reply(rep, taxonomy)
                if rat == JUDGE_ERROR_RATIONALE:        # skip failed calls
                    continue
                iid = corpus.id_at(i)
                cal[iid] = cat
                rejudged_ids.append(iid)
        n_rejudge = len(rejudged_ids)

        # Hold out a slice of the RE-JUDGED calibration (clean final-taxonomy
        # labels) to measure the classifier's agreement with the judge — the
        # run's own labeling-fidelity estimate. Held-out items are just excluded
        # from training; they still get labelled along with everything else.
        val_ids: set = set()
        if n_rejudge >= VAL_MIN_REJUDGE:
            rng_v = random.Random(seed + 1)
            pool = list(rejudged_ids)
            rng_v.shuffle(pool)
            n_val = min(VAL_CAP, int(len(pool) * VAL_FRACTION))
            val_ids = set(pool[:n_val])

        train_texts, train_labels = [], []
        val_texts, val_labels = [], []
        for iid, cat in cal.items():
            it = corpus.get(iid)
            if it is None:
                continue
            text = it.get("text") or ""
            if iid in val_ids:
                val_texts.append(text)
                val_labels.append(cat)
            else:
                train_texts.append(text)
                train_labels.append(cat)

        # finalize_mode picks the classifier: "embed" -> nearest class-mean on
        # embeddings, "finetune" -> a fine-tuned encoder.
        classifier_kind = "finetune" if finalize_mode == "finetune" else "prototype"
        embed = ((embed_fn or _emb.load_embedder(embed_model))
                 if classifier_kind == "prototype" else None)
        targets = tax_names + (["other"] if "other" in cal.values() else [])
        clf = make_classifier(
            classifier_kind, embed_fn=embed, targets=targets,
            descriptions=descriptions, finetune_model=finetune_model,
            epochs=finetune_epochs, seed=seed)
        clf.fit(train_texts, train_labels)

        # Measured fidelity: the trained classifier's agreement with the judge on
        # the held-out calibration items.
        val_accuracy = None
        if val_texts:
            vpreds, _ = clf.predict(iter(val_texts))
            val_accuracy = sum(p == y for p, y in zip(vpreds, val_labels)) \
                / len(val_labels)

        # Label the whole corpus in a streamed pass; the confidence gate keeps
        # the top `coverage` fraction and routes the rest to the judge.
        preds, conf = clf.predict(
            (it.get("text") or "" for it in corpus), EMBED_BATCH)
        keep = _emb.confident_mask(conf, coverage)

        # A finetune classifier can only emit classes it trained on (it ignores
        # the category descriptions the prototype path falls back on), so a
        # taxonomy category with no calibration example is unpredictable, and a
        # single training class collapses it to a constant. In the single-class
        # case its "confident" labels are meaningless, so route every item to
        # the judge (and drop the bogus 100% self-validation); when only some
        # categories are unrepresented, warn that items truly in them may be
        # cheap-mislabeled.
        finalize_notes: list[str] = []
        if classifier_kind == "finetune":
            train_classes = set(train_labels)
            uncovered = [c for c in tax_names if c not in train_classes]
            if len(train_classes) < 2:
                keep = _emb.confident_mask(conf, 0.0)      # judge everything
                val_accuracy = None
                finalize_notes.append(
                    "finetune had <2 calibration classes (degenerate); routed "
                    "all items to the judge — raise calibration_size or use "
                    "finalize=embed")
            elif uncovered:
                finalize_notes.append(
                    f"finetune saw no calibration example for {len(uncovered)} "
                    f"categor{'y' if len(uncovered) == 1 else 'ies'} "
                    f"({', '.join(uncovered[:5])}"
                    f"{', …' if len(uncovered) > 5 else ''}); items truly in "
                    "those may be cheap-mislabeled — raise calibration_size or "
                    "use finalize=embed")

        # Judge the low-confidence tail; dedup identical tail items (one
        # sequential scan filtering on `not keep`, so a file-backed corpus is
        # read straight through rather than seeked per tail item).
        tail_groups = _group_by_content(
            (i, it) for i, it in enumerate(corpus) if not keep[i])
        reps = [g[0] for g in tail_groups]
        rep_label: dict[int, tuple[str, str]] = {}
        for i, rep in _judge_index_replies(reps, tax_str, hardened):
            rep_label[i] = _label_reply(rep, taxonomy)
        tail_label: dict[int, tuple[str, str]] = {}
        for g in tail_groups:
            lab = rep_label[g[0]]
            for i in g:
                tail_label[i] = lab

        # Stream every row (cheap or judged) in item order; roll counts up so
        # nothing accumulates the full row set in memory.
        open(classifications_jsonl, "w").close()
        roll = _CountRollup()
        n_cheap = 0
        with open(classifications_jsonl, "a") as f:
            for i, it in enumerate(corpus):
                if keep[i]:
                    cat = preds[i]
                    rat = (f"{CLASSIFIER_RATIONALE_PREFIX} {classifier_kind}; "
                           f"conf={float(conf[i]):.3f}]")
                    n_cheap += 1
                else:
                    cat, rat = tail_label[i]
                roll.add(cat, rat)
                f.write(json.dumps({**it, "category": cat, "rationale": rat}) + "\n")

        n_judged = len(corpus) - n_cheap
        artifact = build_artifact_from_counts(
            run_id, taxonomy, final_prompt, n_items=len(corpus),
            category_counts=roll.counts, n_coerced=roll.n_coerced,
            n_judge_errors=roll.n_judge_errors)
        artifact["labeling"] = {
            "finalize": finalize_mode,
            "coverage": coverage,
            "n_calibration": len(cal),
            "n_probe": n_probe,
            "n_rejudge": n_rejudge,
            "n_cheap": n_cheap,
            "n_judged": n_judged,
            "val_accuracy": val_accuracy,
            "val_n": len(val_labels),
        }
        atomic_write_json(artifact_path, artifact)
        state.finalized_at = taxonomy

        if val_accuracy is None:
            val_line = ("measured fidelity: not estimated "
                        f"(need >= {VAL_MIN_REJUDGE} re-judged items; "
                        "raise calibration_size)\n")
        else:
            warn = ("  [LOW — cheap labels are noisy on this corpus; consider "
                    "finalize=judge or a lower coverage]"
                    if val_accuracy < VAL_LOW_FIDELITY else "")
            val_line = (f"measured fidelity: {val_accuracy:.1%} agreement with the "
                        f"judge on {len(val_labels)} held-out items{warn}\n")
        notes_block = "".join(f"NOTE: {m}\n" for m in finalize_notes)
        return (
            f"Wrote {artifact_path} (finalize={finalize_mode})\n"
            f"calibration: {n_probe} probe labels"
            f"{f' + {n_rejudge} re-judged' if n_rejudge else ''}\n"
            f"{val_line}"
            f"{notes_block}"
            f"n_items={len(corpus)}: {n_cheap} labelled by the classifier, "
            f"{n_judged} routed to the judge (coverage={coverage}); "
            f"n_judge_errors={artifact['n_judge_errors']}\n"
            f"category_counts={json.dumps(artifact['category_counts'], indent=2)}"
        )

    def _discovery_only_finalize(final_prompt: str) -> str:
        """finalize_mode == "none": ship the discovered taxonomy WITHOUT the
        O(N) full-corpus labelling pass. The items already judged for free
        during discovery (the probes, kept in `state.probe_labels`) are written
        to classifications.jsonl as a labelled sample; the rest of the corpus is
        left unlabelled. Keeps the counts↔n_items invariant by reporting
        n_items as the sample size, with the full corpus size under `labeling`."""
        taxonomy = state.taxonomy
        valid = {c["name"] for c in taxonomy} | {"other"}
        open(classifications_jsonl, "w").close()
        roll = _CountRollup()
        n_sample = 0
        with open(classifications_jsonl, "a") as f:
            for iid, cat in state.probe_labels.items():
                if cat not in valid:
                    continue
                it = corpus.get(iid)
                if it is None:
                    continue
                roll.add(cat, DISCOVERY_PROBE_RATIONALE)
                n_sample += 1
                f.write(json.dumps({**it, "category": cat,
                                    "rationale": DISCOVERY_PROBE_RATIONALE}) + "\n")
        artifact = build_artifact_from_counts(
            run_id, taxonomy, final_prompt, n_items=n_sample,
            category_counts=roll.counts, n_coerced=roll.n_coerced,
            n_judge_errors=roll.n_judge_errors)
        artifact["labeling"] = {
            "finalize": "none",
            "labelled_corpus": False,
            "n_corpus": len(corpus),
            "n_sample": n_sample,
        }
        atomic_write_json(artifact_path, artifact)
        state.finalized_at = taxonomy
        return (
            f"Wrote {artifact_path} (finalize=none — discovery only)\n"
            f"Discovered {len(taxonomy)} categories; the corpus of {len(corpus)} "
            f"items was NOT fully labelled.\n"
            f"Recorded {n_sample} discovery-probe items as a labelled sample. "
            f"Re-run with finalize=judge/embed/finetune to label every item.\n"
            f"category_counts={json.dumps(artifact['category_counts'], indent=2)}"
        )

    def _coverage_ok() -> tuple[bool, float]:
        """Re-measure convergence on a FRESH uniform-random probe, independent of
        whatever the orchestrator chose to classify — so a steered discovery loop
        cannot manufacture a stop signal by only classifying items it already
        understands. Records the probe's labels as calibration (and as frontier
        signal for `sample_uncovered`) and returns (converged, unmatched_rate)."""
        chosen, _ = _draw_unseen(probe_size)
        tax_str = _format_taxonomy(state.taxonomy)
        prompt = DEFAULT_CLASSIFY_PROMPT_MULTI if multi_label else DEFAULT_CLASSIFY_PROMPT
        suffix = ESCAPE_HATCH_SUFFIX_MULTI if multi_label else ESCAPE_HATCH_SUFFIX
        hardened = prompt.strip() + suffix
        n_other = n_scored = 0
        for idx, rep in _judge_index_replies(chosen, tax_str, hardened):
            if rep is None:
                continue
            if multi_label:
                cats, _ = _coerce_categories(_parse_json_block(rep), state.taxonomy)
                primary = cats[0] if cats else "other"
                n_other += (not cats)
            else:
                primary, _ = _label_reply(rep, state.taxonomy)
                n_other += (primary == "other")
            state.probe_labels[corpus.id_at(idx)] = primary
            n_scored += 1
        rate = (n_other / n_scored) if n_scored else 1.0
        return rate <= converge_below, rate

    @tool
    def finalize_classify(final_prompt: str) -> str:
        """Have the judge label every item in the corpus against the current
        taxonomy. Writes `<output_dir>/taxonomy.json` and returns per-category
        counts.

        Will not run twice in a row on the same taxonomy — the output file is
        already up to date. To relabel, call `revise_taxonomy` first to change
        the taxonomy.

        `final_prompt` MUST tell the judge to reply ONLY with a JSON object:
          {"category": <name | "other">, "rationale": <≤2 sentences>}."""
        taxonomy = state.taxonomy
        if not taxonomy:
            return "ERROR: taxonomy is empty. Call revise_taxonomy(add ...) before finalizing."
        if state.classify_calls < min_iterations:
            return (f"ERROR: finalize_classify requires at least {min_iterations} "
                    f"classification rounds before convergence is allowed (you have "
                    f"completed {state.classify_calls}). A single lucky probe is not "
                    f"enough. Sample more items and call classify_with_judge to keep "
                    f"iterating.")
        if state.finalized_at == taxonomy:
            return (f"ERROR: finalize_classify already ran with this taxonomy. "
                    f"The artifact at {artifact_path} is up to date — stop here. "
                    f"If you genuinely want to relabel, revise the taxonomy first.")
        # Coverage backstop (opt-in): verify convergence on an independent
        # uniform probe the orchestrator does not control. Refuse while there is
        # still budget to keep discovering; once the budget is spent, finalize
        # anyway and record the shortfall so the artifact is honestly flagged.
        if enforce_coverage and not state.skip_coverage_check:
            ok, rate = _coverage_ok()
            if not ok and state.classify_calls < classify_budget:
                return (f"ERROR: coverage check failed — {rate:.0%} of a fresh "
                        f"uniform-random probe was labelled \"other\" (threshold "
                        f"{converge_below:.0%}). The taxonomy still misses items. "
                        f"Sample more (use sample_uncovered), propose novelties, "
                        f"revise, then finalize again.")
            state.low_coverage = None if ok else rate
        if finalize_mode == "none":
            return _discovery_only_finalize(final_prompt)
        if finalize_mode in ("embed", "finetune"):
            return _classifier_finalize(final_prompt)
        tax_str = _format_taxonomy(taxonomy)
        suffix = ESCAPE_HATCH_SUFFIX_MULTI if multi_label else ESCAPE_HATCH_SUFFIX
        hardened = final_prompt.strip() + suffix

        # Deduplicate by item content: identical items get the same label, so
        # the judge is paid once per distinct item instead of once per duplicate.
        rep_to_group = {g[0]: g for g in _group_by_content(enumerate(corpus))}

        # Label the distinct items in bounded chunks (via _judge_index_replies)
        # so peak memory is one chunk of prompts/replies, not the whole corpus.
        # Truncate any stale file, then append each group's rows as its reply
        # returns — a crash mid-finalize keeps a prefix of real labels on disk,
        # and taxonomy.json is written only after every group is labelled. Counts
        # roll up incrementally so nothing holds the full row set in memory.
        open(classifications_jsonl, "w").close()
        roll = _CountRollup()
        with open(classifications_jsonl, "a") as f:
            for rep_i, rep in _judge_index_replies(list(rep_to_group),
                                                   tax_str, hardened):
                g = rep_to_group[rep_i]
                if multi_label:
                    cats, rat = _coerce_categories(_parse_json_block(rep), taxonomy)
                    roll.add_multi(cats, rat, len(g))
                    primary = cats[0] if cats else "other"
                    for i in g:
                        f.write(json.dumps({**corpus[i], "category": primary,
                                            "categories": cats, "rationale": rat}) + "\n")
                else:
                    cat, rat = _label_reply(rep, taxonomy)
                    roll.add(cat, rat, len(g))
                    for i in g:
                        f.write(json.dumps(
                            {**corpus[i], "category": cat, "rationale": rat}) + "\n")

        artifact = build_artifact_from_counts(
            run_id, taxonomy, final_prompt, n_items=len(corpus),
            category_counts=roll.counts, n_coerced=roll.n_coerced,
            n_judge_errors=roll.n_judge_errors)
        # Backstop let this through above threshold (budget spent): flag it.
        if state.low_coverage is not None:
            artifact["low_coverage_rate"] = round(state.low_coverage, 3)
        atomic_write_json(artifact_path, artifact)
        # Snapshot the taxonomy that this artifact reflects. `_apply_ops`
        # always reassigns state.taxonomy to a fresh list of fresh dicts,
        # so this reference stays a stable record of the finalized state.
        state.finalized_at = taxonomy
        return (
            f"Wrote {artifact_path}\n"
            f"n_items={artifact['n_items']} (n_coerced={artifact['n_coerced']}, "
            f"n_judge_errors={artifact['n_judge_errors']})\n"
            f"category_counts={json.dumps(artifact['category_counts'], indent=2)}"
        )

    def _artifact_from_streamed_classifications() -> dict | None:
        """Rebuild the finalize artifact from a complete classifications.jsonl,
        or None if the file is missing, unreadable, incomplete, or references a
        category not in the current taxonomy (so a stale/partial file is never
        mistaken for a finished run). Streams the file and rolls counts up as it
        goes, so recovering a million-row run never holds every row in memory."""
        if not os.path.exists(classifications_jsonl):
            return None
        # Ids come from the corpus's in-memory index (no file re-scan; O(1)/item
        # for a file-backed corpus), not by streaming every row again.
        want_ids = {corpus.id_at(i) for i in range(len(corpus))}
        valid = {c["name"] for c in state.taxonomy} | {"other"}
        seen_ids: set = set()
        roll = _CountRollup()
        n_rows = 0
        try:
            for r in _iter_jsonl(classifications_jsonl):
                cat = r.get("category")
                if cat not in valid:
                    return None
                n_rows += 1
                seen_ids.add(r.get("id"))
                roll.add(cat, r.get("rationale", ""))
        except (ValueError, OSError):
            return None
        if n_rows != len(corpus) or seen_ids != want_ids:
            return None
        return build_artifact_from_counts(
            run_id, state.taxonomy,
            "(recovered from streamed classifications.jsonl)",
            n_items=n_rows, category_counts=roll.counts,
            n_coerced=roll.n_coerced, n_judge_errors=roll.n_judge_errors)

    def force_finalize_with_default_prompt() -> dict | None:
        """Fallback path for when the orchestrator stream ends without ever
        calling `finalize_classify`. Bypasses the `min_iterations` check
        because we are recovering an aborted run, not optimising a healthy
        one. Returns the artifact dict on success, or None if the taxonomy
        is empty (nothing to label against). The caller is expected to
        check whether `taxonomy.json` already exists before calling this."""
        if not state.taxonomy:
            return None

        # If a complete set of streamed labels already exists on disk (a
        # finalize that wrote every row but was interrupted before consolidating
        # taxonomy.json), rebuild the artifact from it rather than paying the
        # judge to relabel the whole corpus a second time.
        reused = _artifact_from_streamed_classifications()
        if reused is not None:
            atomic_write_json(artifact_path, reused)
            state.finalized_at = state.taxonomy
            return reused

        original_floor = state.classify_calls
        # Recovery bypasses both the min_iterations floor and the coverage
        # backstop — we are salvaging an aborted run, not optimising a healthy
        # one, and must not refuse to write the artifact we already earned.
        state.skip_coverage_check = True
        try:
            # Force the floor check to pass by temporarily reporting we have
            # already met it. (state is a closure; finalize_classify reads it.)
            state.classify_calls = max(state.classify_calls, min_iterations)
            finalize_classify.invoke(
                DEFAULT_CLASSIFY_PROMPT_MULTI if multi_label
                else DEFAULT_CLASSIFY_PROMPT)
        finally:
            state.classify_calls = original_floor
            state.skip_coverage_check = False
        if not os.path.exists(artifact_path):
            return None
        with open(artifact_path) as f:
            return json.load(f)

    discovery_tools = [sample_items, get_taxonomy, revise_taxonomy,
                       classify_with_judge, propose_novelties_with_judge,
                       finalize_classify]
    # Keep the default tool set (and its order) byte-identical so an unchanged
    # run reproduces prior behaviour; only the opt-in strategy adds the 7th tool.
    if sample_strategy == "uncovered":
        discovery_tools.append(sample_uncovered)
    return (discovery_tools, force_finalize_with_default_prompt)
