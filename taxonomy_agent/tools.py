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

from .corpus import Corpus, InMemoryCorpus


ESCAPE_HATCH_SUFFIX = (
    "\n\nIMPORTANT: If none of the listed categories applies to this item, "
    "reply with `\"category\": \"other\"`. Do not invent new category names — "
    "every reply must use either an exact name from the listed taxonomy or "
    "the literal string `other`."
)

# Sentinel rationale prefix for items where the judge call itself failed
# (network/HTTP/timeout after retry). These are NOT genuine misfits — they
# must be excluded from the unmatched rate, not folded into "other".
JUDGE_ERROR_RATIONALE = "[judge call failed]"

# Prefix `_coerce_category` stamps on the rationale when the judge returned a
# label outside the taxonomy. Named so the count of coerced rows has one home
# (see `is_coerced_rationale` / `summarize_rows`) instead of a repeated literal.
COERCED_RATIONALE_PREFIX = "[coerced from invented label"

# Prefix stamped on rows labelled cheaply by the cascade's embedding classifier
# (not the judge). Distinct from the coerced/judge-error sentinels so cascade
# rows are never miscounted as either.
CASCADE_RATIONALE_PREFIX = "[cascade: nearest-prototype;"

# finalize_classify labels the corpus one bounded batch of distinct items at a
# time so peak memory stays flat as the corpus grows: a million-item run never
# holds a million prompt strings (or reply strings) alive at once, only one
# chunk's worth. Small corpora finish in a single chunk, unchanged.
FINALIZE_CHUNK = 2000

# The cascade embeds the corpus one batch of this many items at a time, so only
# one batch of embedding vectors is ever resident — a 10M-item run never holds a
# 10M x dim array (which would be tens of GB).
EMBED_BATCH = 1024

# The classification instruction used when the caller has none of its own — the
# auto-finalize fallback and `refine()`'s re-classification. `finalize_classify`
# receives the orchestrator's own prompt instead.
DEFAULT_CLASSIFY_PROMPT = (
    "Pick the single category from the list that best describes the item. "
    "Reply only with a JSON object: "
    "{\"category\": <name or \"other\">, \"rationale\": <one or two sentences>}."
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


def summarize_rows(rows: list[dict]) -> tuple[dict, int, int]:
    """Roll classification rows up into `(category_counts, n_coerced,
    n_judge_errors)` by inspecting each row's category and rationale sentinel."""
    counts: dict[str, int] = {}
    n_coerced = n_judge_errors = 0
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
        rat = r.get("rationale", "")
        if rat == JUDGE_ERROR_RATIONALE:
            n_judge_errors += 1
        elif is_coerced_rationale(rat):
            n_coerced += 1
    return counts, n_coerced, n_judge_errors


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
    already hold every row in memory (`refine()` and the streamed-recovery
    path); the rows themselves are persisted separately to
    `classifications.jsonl`."""
    counts, n_coerced, n_judge_errors = summarize_rows(rows)
    return build_artifact_from_counts(
        run_id, taxonomy, final_prompt, n_items=len(rows),
        category_counts=counts, n_coerced=n_coerced,
        n_judge_errors=n_judge_errors)


def write_taxonomy_state(path: str, taxonomy: list[dict],
                         n_classify_calls: int = 0) -> None:
    """Persist the working taxonomy + classify-call count to `taxonomy_state.json`."""
    with open(path, "w") as f:
        json.dump({"taxonomy": taxonomy, "n_classify_calls": n_classify_calls},
                  f, indent=2)


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
    # Item id -> judge label from the discovery probes (last write wins). These
    # are items the judge already labelled while exploring, reused for free as
    # calibration prototypes by finalize_mode="cascade". Kept across all
    # discovery iterations: measuring on real multi-revise runs showed that
    # restricting to the final taxonomy version discards too much calibration
    # data and *lowers* prototype fidelity by 5-15% — intermediate-taxonomy
    # labels are mostly still correct, and more examples beat recency.
    probe_labels: dict = field(default_factory=dict)


_OpHandler = Callable[[list[dict], dict], tuple[list[dict], dict]]


def _op_add(tax: list[dict], op: dict) -> tuple[list[dict], dict]:
    name, desc = op["name"], op["description"]
    if any(c["name"] == name for c in tax):
        return tax, {"op": "add", "name": name, "result": "skipped (already exists)"}
    return tax + [{"name": name, "description": desc}], {
        "op": "add", "name": name, "result": "ok",
    }


def _op_rename(tax: list[dict], op: dict) -> tuple[list[dict], dict]:
    old, new = op["old_name"], op["new_name"]
    if not any(c["name"] == old for c in tax):
        return tax, {"op": "rename", "from": old, "to": new, "result": "missing source"}
    if any(c["name"] == new for c in tax):
        return tax, {"op": "rename", "from": old, "to": new,
                     "result": "target name already exists"}
    new_tax = [{**c, "name": new} if c["name"] == old else dict(c) for c in tax]
    return new_tax, {"op": "rename", "from": old, "to": new, "result": "ok"}


def _op_edit(tax: list[dict], op: dict) -> tuple[list[dict], dict]:
    name, desc = op["name"], op["description"]
    if not any(c["name"] == name for c in tax):
        return tax, {"op": "edit", "name": name, "result": "missing"}
    new_tax = [{**c, "description": desc} if c["name"] == name else dict(c) for c in tax]
    return new_tax, {"op": "edit", "name": name, "result": "ok"}


def _op_drop(tax: list[dict], op: dict) -> tuple[list[dict], dict]:
    name = op["name"]
    new_tax = [c for c in tax if c["name"] != name]
    return new_tax, {"op": "drop", "name": name,
                     "result": "ok" if len(new_tax) < len(tax) else "missing"}


def _op_merge(tax: list[dict], op: dict) -> tuple[list[dict], dict]:
    into = op["into"]
    sources = op.get("from", []) or []
    desc = op.get("description")
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
        if any(c["name"] == nc["name"] for c in new_tax):
            continue
        new_tax = new_tax + [{"name": nc["name"], "description": nc["description"]}]
        added.append(nc["name"])
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
               finalize_mode: str = "judge", cascade_coverage: float = 0.85,
               embed_model: str = "all-MiniLM-L6-v2", embed_fn=None,
               cascade_classifier: str = "prototype",
               cascade_calibration_size: int = 0,
               cascade_finetune_model: str = "distilbert-base-uncased",
               cascade_finetune_epochs: int = 4):
    """Construct the six LangChain tools, sharing state via closure.

    The taxonomy lives entirely inside the closure — the orchestrator mutates
    it through `revise_taxonomy` and reads it via `get_taxonomy`.

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

    @tool
    def sample_items(k: int) -> str:
        """Return K items from the corpus. Default K = 20.

        Items returned by earlier calls are not repeated until the corpus is
        exhausted. At that point the history resets and the reply notes the
        wraparound; subsequent batches will overlap with prior ones."""
        n = len(corpus)
        k = max(1, min(int(k), n))
        # Sample by index, not by scanning every item, so a file-backed corpus
        # only reads the K rows it hands out.
        unseen = [i for i in range(n) if i not in state.sampled_idx]
        note = ""
        if len(unseen) < k:
            note = (f" (pool of {n} exhausted — sampling history reset; "
                    f"expect overlap with prior probes)")
            state.sampled_idx = set()
            unseen = range(n)
        chosen = rng.sample(unseen, k)
        state.sampled_idx.update(chosen)
        sampled = [corpus[i] for i in chosen]
        ids = [it["id"] for it in sampled]
        blocks = [_format_item(it, i) for i, it in enumerate(sampled, start=1)]
        return (
            f"Sampled {k} items{note}.\n"
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
        hardened = classify_prompt.strip() + ESCAPE_HATCH_SUFFIX
        prompts = [build_classify_prompt(hardened, tax_str, it) for it in sel]
        replies = judge.parallel(prompts, concurrency=concurrency, max_tokens=300)
        results = []
        n_other = 0
        n_coerced = 0
        n_judge_errors = 0
        for it, rep in zip(sel, replies):
            if rep is None:
                n_judge_errors += 1
                results.append({"item_id": it["id"], "category": "other",
                                "rationale": JUDGE_ERROR_RATIONALE})
                continue
            parsed = _parse_json_block(rep)
            cat, rat = _coerce_category(parsed, taxonomy)
            if is_coerced_rationale(rat):
                n_coerced += 1
            if cat == "other":
                n_other += 1
            results.append({"item_id": it["id"], "category": cat, "rationale": rat[:400]})
        n_scored = len(sel) - n_judge_errors
        # Keep each probe's judge label as free calibration for a cascade
        # finalize (skip failed calls). Labels are against the taxonomy at this
        # moment; the cascade filters to categories that survive to the end.
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

    def _cascade_finalize(final_prompt: str) -> str:
        """finalize_mode='cascade': label the confident majority with an
        embedding classifier (prototypes averaged from the discovery probes the
        judge already paid for) and route only the low-confidence tail to the
        judge. Produces the same compact taxonomy.json + streamed
        classifications.jsonl as the judge path, so everything downstream is
        unchanged."""
        from . import cascade as _casc
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
        # classifier, at cascade_calibration_size judge calls.
        cal: dict[str, str] = {}
        for iid, cat in state.probe_labels.items():
            if cat in valid:
                cal[iid] = cat
        n_probe = len(cal)
        n_rejudge = 0
        if cascade_calibration_size and cascade_calibration_size > 0:
            rng_c = random.Random(seed)
            order = list(range(len(corpus)))
            rng_c.shuffle(order)
            picked = []
            for i in order:
                if corpus[i]["id"] not in cal:
                    picked.append(i)
                    if len(picked) >= cascade_calibration_size:
                        break
            for s in range(0, len(picked), FINALIZE_CHUNK):
                chunk = picked[s:s + FINALIZE_CHUNK]
                prompts = [build_classify_prompt(hardened, tax_str, corpus[i])
                           for i in chunk]
                replies = judge.parallel(prompts, concurrency=concurrency * 2,
                                         max_tokens=300)
                for i, rep in zip(chunk, replies):
                    if rep is None:
                        continue
                    c, _ = _coerce_category(_parse_json_block(rep), taxonomy)
                    cal[corpus[i]["id"]] = c
                    n_rejudge += 1

        cal_texts, cal_labels = [], []
        for iid, cat in cal.items():
            it = corpus.get(iid)
            if it is not None:
                cal_texts.append(it.get("text") or "")
                cal_labels.append(cat)

        # Train the chosen classifier on the calibration set, then label the
        # whole corpus in a streamed pass; the confidence gate keeps the top
        # `cascade_coverage` fraction and routes the rest to the judge.
        needs_embed = cascade_classifier in ("prototype", "logreg")
        embed = (embed_fn or _casc.load_embedder(embed_model)) if needs_embed else None
        targets = tax_names + (["other"] if "other" in cal_labels else [])
        clf = make_classifier(
            cascade_classifier, embed_fn=embed, targets=targets,
            descriptions=descriptions, finetune_model=cascade_finetune_model,
            epochs=cascade_finetune_epochs, seed=seed, batch_size=EMBED_BATCH)
        clf.fit(cal_texts, cal_labels)
        preds, conf = clf.predict(
            (it.get("text") or "" for it in corpus), EMBED_BATCH)
        keep = _casc.confident_mask(conf, cascade_coverage)

        # Judge the low-confidence tail, chunked; dedup identical tail items so
        # the judge is paid once per distinct item there too.
        tail_idx = [i for i in range(len(corpus)) if not keep[i]]
        tail_label: dict[int, tuple[str, str]] = {}
        if tail_idx:
            tail_groups: dict[str, list[int]] = {}
            for i in tail_idx:
                tail_groups.setdefault(_content_hash(corpus[i]), []).append(i)
            reps = [g[0] for g in tail_groups.values()]
            rep_label: dict[int, tuple[str, str]] = {}
            for s in range(0, len(reps), FINALIZE_CHUNK):
                chunk = reps[s:s + FINALIZE_CHUNK]
                prompts = [build_classify_prompt(hardened, tax_str, corpus[i])
                           for i in chunk]
                replies = judge.parallel(prompts, concurrency=concurrency * 2,
                                         max_tokens=300)
                for i, rep in zip(chunk, replies):
                    rep_label[i] = (("other", JUDGE_ERROR_RATIONALE) if rep is None
                                    else _coerce_category(_parse_json_block(rep),
                                                          taxonomy))
            for g in tail_groups.values():
                lab = rep_label[g[0]]
                for i in g:
                    tail_label[i] = lab

        # Stream every row (cheap or judged) in item order; roll counts up so
        # nothing accumulates the full row set in memory.
        open(classifications_jsonl, "w").close()
        counts: dict[str, int] = {}
        n_coerced = n_judge_errors = n_cheap = 0
        with open(classifications_jsonl, "a") as f:
            for i, it in enumerate(corpus):
                if keep[i]:
                    cat = preds[i]
                    rat = (f"{CASCADE_RATIONALE_PREFIX} {cascade_classifier}; "
                           f"conf={float(conf[i]):.3f}]")
                    n_cheap += 1
                else:
                    cat, rat = tail_label[i]
                    if rat == JUDGE_ERROR_RATIONALE:
                        n_judge_errors += 1
                    elif is_coerced_rationale(rat):
                        n_coerced += 1
                counts[cat] = counts.get(cat, 0) + 1
                f.write(json.dumps({**it, "category": cat, "rationale": rat}) + "\n")

        artifact = build_artifact_from_counts(
            run_id, taxonomy, final_prompt, n_items=len(corpus),
            category_counts=counts, n_coerced=n_coerced,
            n_judge_errors=n_judge_errors)
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2)
        state.finalized_at = taxonomy
        n_judged = len(corpus) - n_cheap
        return (
            f"Wrote {artifact_path} (finalize_mode=cascade, "
            f"classifier={cascade_classifier})\n"
            f"calibration: {n_probe} probe labels"
            f"{f' + {n_rejudge} re-judged' if n_rejudge else ''}\n"
            f"n_items={len(corpus)}: {n_cheap} labelled by the classifier, "
            f"{n_judged} routed to the judge (coverage={cascade_coverage}); "
            f"n_judge_errors={n_judge_errors}\n"
            f"category_counts={json.dumps(artifact['category_counts'], indent=2)}"
        )

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
        if finalize_mode == "cascade":
            return _cascade_finalize(final_prompt)
        tax_str = _format_taxonomy(taxonomy)
        hardened = final_prompt.strip() + ESCAPE_HATCH_SUFFIX

        # Deduplicate by item content (everything except the arbitrary id):
        # items with identical text and metadata get the same label, so the
        # judge is paid once per distinct item instead of once per duplicate.
        groups: dict[str, list[int]] = {}
        for i, it in enumerate(corpus):
            groups.setdefault(_content_hash(it), []).append(i)
        group_indices = list(groups.values())
        groups = None  # release the key map before the (larger) judge phase

        def _label(rep: str | None):
            if rep is None:
                return "other", JUDGE_ERROR_RATIONALE
            return _coerce_category(_parse_json_block(rep), taxonomy)

        # Label the distinct items in bounded chunks so peak memory is one
        # chunk's worth of prompts and replies, not the whole corpus. Truncate
        # any stale file from a previous attempt, then append each chunk's rows
        # as it returns — a crash mid-finalize keeps a prefix of real labels on
        # disk, and the consolidated taxonomy.json summary is written only after
        # every chunk succeeds. Counts are rolled up incrementally for the same
        # reason: nothing ever holds the full row set in memory.
        open(classifications_jsonl, "w").close()
        counts: dict[str, int] = {}
        n_coerced = n_judge_errors = 0
        for start in range(0, len(group_indices), FINALIZE_CHUNK):
            chunk = group_indices[start:start + FINALIZE_CHUNK]
            chunk_prompts = [
                build_classify_prompt(hardened, tax_str, corpus[g[0]]) for g in chunk]
            chunk_replies = judge.parallel(
                chunk_prompts, concurrency=concurrency * 2, max_tokens=300)
            with open(classifications_jsonl, "a") as f:
                for g, rep in zip(chunk, chunk_replies):
                    cat, rat = _label(rep)
                    n = len(g)
                    if rat == JUDGE_ERROR_RATIONALE:
                        n_judge_errors += n
                    elif is_coerced_rationale(rat):
                        n_coerced += n
                    counts[cat] = counts.get(cat, 0) + n
                    for i in g:
                        f.write(json.dumps(
                            {**corpus[i], "category": cat, "rationale": rat}) + "\n")

        artifact = build_artifact_from_counts(
            run_id, taxonomy, final_prompt, n_items=len(corpus),
            category_counts=counts, n_coerced=n_coerced,
            n_judge_errors=n_judge_errors)
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2)
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
        want_ids = {it["id"] for it in corpus}
        valid = {c["name"] for c in state.taxonomy} | {"other"}
        seen_ids: set = set()
        counts: dict[str, int] = {}
        n_rows = n_coerced = n_judge_errors = 0
        try:
            with open(classifications_jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    cat = r.get("category")
                    if cat not in valid:
                        return None
                    n_rows += 1
                    seen_ids.add(r.get("id"))
                    counts[cat] = counts.get(cat, 0) + 1
                    rat = r.get("rationale", "")
                    if rat == JUDGE_ERROR_RATIONALE:
                        n_judge_errors += 1
                    elif is_coerced_rationale(rat):
                        n_coerced += 1
        except (ValueError, OSError):
            return None
        if n_rows != len(corpus) or seen_ids != want_ids:
            return None
        return build_artifact_from_counts(
            run_id, state.taxonomy,
            "(recovered from streamed classifications.jsonl)",
            n_items=n_rows, category_counts=counts, n_coerced=n_coerced,
            n_judge_errors=n_judge_errors)

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
            with open(artifact_path, "w") as f:
                json.dump(reused, f, indent=2)
            state.finalized_at = state.taxonomy
            return reused

        original_floor = state.classify_calls
        try:
            # Force the floor check to pass by temporarily reporting we have
            # already met it. (state is a closure; finalize_classify reads it.)
            state.classify_calls = max(state.classify_calls, min_iterations)
            finalize_classify.invoke(DEFAULT_CLASSIFY_PROMPT)
        finally:
            state.classify_calls = original_floor
        if not os.path.exists(artifact_path):
            return None
        with open(artifact_path) as f:
            return json.load(f)

    return ([sample_items, get_taxonomy, revise_taxonomy,
            classify_with_judge, propose_novelties_with_judge, finalize_classify],
            force_finalize_with_default_prompt)
