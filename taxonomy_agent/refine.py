"""Feedback-driven taxonomy refinement.

After a :func:`~taxonomy_agent.run`, a user may be unhappy with the taxonomy and
want to steer it in natural language ("merge these two", "split X into finer
categories", "you're too coarse"). :func:`refine` takes a finished run plus a
feedback string and:

1. Interprets the feedback into either a list of typed taxonomy edits
   (``add`` / ``rename`` / ``edit`` / ``merge`` / ``split`` / ``drop`` — the
   same DSL the orchestrator already speaks) or, for open-ended feedback,
   re-discovery guidance.
2. Applies the edits to a *copy* of the taxonomy.
3. Re-labels only the items an edit could have moved (a rename or merge is a
   pure relabel with **no** judge calls; an add re-judges the ``other`` bucket;
   a drop/split/edit re-judges just that category's items). Pass
   ``reclassify="all"`` to relabel the whole corpus, or ``"none"`` to skip the
   judge entirely.
4. Writes a new run directory and returns a fresh :class:`RunResult`.

The original run is never mutated. Open-ended feedback is handled by warm-
starting a short discovery loop from the current taxonomy (see
``run(..., initial_taxonomy=...)``).
"""
from __future__ import annotations

import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Union

from .agent import RunResult, run
from .cost import CostTracker
from .judge import Judge
from .tools import (ESCAPE_HATCH_SUFFIX, JUDGE_ERROR_RATIONALE, _TaxonomyState,
                    _apply_ops, _coerce_category, _format_item,
                    _format_taxonomy, _parse_json_block)

# The interpreter is a single cheap LLM call. It either emits concrete typed
# edits or declares the request open-ended (needing re-discovery). Keeping both
# in one call means precise asks stay a one-shot, deterministic edit.
INTERPRETER_PROMPT = """You translate a user's natural-language feedback about a \
discovered taxonomy into concrete edits, using this operation DSL:

  {{"op": "add",    "name": <snake_case>, "description": <one sentence>}}
  {{"op": "rename", "old_name": <existing>, "new_name": <new snake_case>}}
  {{"op": "edit",   "name": <existing>, "description": <new sentence>}}
  {{"op": "merge",  "into": <target name>, "from": [<existing names to absorb>]}}
  {{"op": "split",  "from": <existing name>, "into": [{{"name": .., "description": ..}}, ...]}}
  {{"op": "drop",   "name": <existing name>}}

Current taxonomy:
{taxonomy}

User feedback:
{feedback}

If the feedback maps to specific edits, reply ONLY with a JSON object:
  {{"operations": [ <op>, ... ]}}
If the feedback is open-ended and needs the categories re-discovered from the \
corpus (e.g. "too fine-grained", "reorganize by intent, not surface form"), \
reply ONLY with:
  {{"open_ended": true, "guidance": <one sentence restating the goal as a constraint>}}
Reply with JSON only, no prose."""


def _load_run(run_or_dir: Union["RunResult", str, Path]) -> dict:
    """Normalise the input to the pieces refine needs: the taxonomy, the prior
    classification rows, the reconstructed items, the source directory, and the
    original instruction. Accepts a RunResult or a run directory path."""
    if isinstance(run_or_dir, RunResult):
        res = run_or_dir
        output_dir = res.get("output_dir")
    else:
        res = RunResult.from_dir(run_or_dir)
        output_dir = str(run_or_dir)
    artifact = res.get("artifact") or {}
    taxonomy = artifact.get("taxonomy") or []
    rows = artifact.get("classifications") or []
    if not taxonomy or not rows:
        raise ValueError(
            "refine needs a completed run with a taxonomy and classifications; "
            f"got taxonomy={len(taxonomy)} categories, {len(rows)} rows.")
    # An item is a classification row minus the labels refine will recompute.
    items = [{k: v for k, v in r.items() if k not in ("category", "rationale")}
             for r in rows]
    instruction = ""
    if output_dir:
        meta_path = Path(output_dir) / "meta.json"
        if meta_path.exists():
            try:
                instruction = json.loads(meta_path.read_text()).get("instruction", "")
            except (ValueError, OSError):
                pass
    return {"taxonomy": taxonomy, "rows": rows, "items": items,
            "output_dir": output_dir, "instruction": instruction}


def interpret_feedback(feedback: str, taxonomy: list[dict], judge: "Judge") -> dict:
    """Turn natural-language feedback into either ``{"operations": [...]}`` or
    ``{"open_ended": True, "guidance": ...}`` via a single judge call. Returns
    ``{"operations": []}`` if the reply cannot be parsed (nothing to apply)."""
    prompt = INTERPRETER_PROMPT.format(
        taxonomy=_format_taxonomy(taxonomy), feedback=feedback.strip())
    reply = judge.call(prompt, max_tokens=800)
    parsed = _parse_json_block(reply)
    if isinstance(parsed, dict):
        if parsed.get("open_ended"):
            return {"open_ended": True,
                    "guidance": str(parsed.get("guidance") or feedback).strip()}
        ops = parsed.get("operations")
        if isinstance(ops, list):
            return {"operations": ops}
    return {"operations": []}


def _replay_labels(rows: list[dict], operations: list[dict],
                   new_names: set[str], reclassify: str) -> tuple[list, list[bool]]:
    """Replay the edits over each item's *current* label to decide, per item,
    the deterministic new label and whether the judge must re-decide it.

    rename/merge are pure relabels (no judge). drop/split leave the item without
    a valid home → judge. edit changes a category's meaning → judge its members.
    A newly added or split-out category can pull in items currently labelled
    ``other`` → judge those too. Returns (det_labels, need_judge) aligned to rows;
    a det_label of None means "must be judged"."""
    has_new_home = any(op.get("op") in ("add", "split") for op in operations)
    det: list = []
    need: list[bool] = []
    for r in rows:
        c = r.get("category")
        judge_it = False
        for op in operations:
            t = op.get("op")
            if t == "rename" and c == op.get("old_name"):
                c = op.get("new_name")
            elif t == "merge" and c in (op.get("from") or []):
                c = op.get("into")
            elif t == "drop" and c == op.get("name"):
                c, judge_it = None, True
            elif t == "split" and c == op.get("from"):
                c, judge_it = None, True
            elif t == "edit" and c == op.get("name"):
                judge_it = True
        # A category that no longer exists (and isn't 'other') must be re-judged.
        if c is not None and c != "other" and c not in new_names:
            c, judge_it = None, True
        # Items in 'other' are the candidates a new/split category can rescue.
        if has_new_home and r.get("category") == "other":
            judge_it = True
        if reclassify == "all":
            judge_it = True
        elif reclassify == "none":
            judge_it = False
            if c is None:
                c = "other"
        det.append(c)
        need.append(judge_it)
    return det, need


def _reclassify(rows: list[dict], items: list[dict], new_taxonomy: list[dict],
                operations: list[dict], judge: "Judge", *, reclassify: str,
                concurrency: int) -> tuple[list[dict], int, int]:
    """Produce the refined classification rows. Deterministically-relabelled
    items keep their prior rationale; re-judged items get a fresh label. Returns
    (rows, n_coerced, n_judge_errors)."""
    new_names = {c["name"] for c in new_taxonomy}
    det, need = _replay_labels(rows, operations, new_names, reclassify)

    judged: dict[int, tuple[str, str]] = {}
    to_judge = [i for i, flag in enumerate(need) if flag]
    if to_judge:
        tax_str = _format_taxonomy(new_taxonomy)
        base = ("Pick the single category from the list that best describes the "
                "item. Reply only with a JSON object: "
                "{\"category\": <name or \"other\">, \"rationale\": "
                "<one or two sentences>}." + ESCAPE_HATCH_SUFFIX)
        prompts = [
            f"{base}\n\n## Categories\n{tax_str}\n\n## Item to classify\n"
            f"{_format_item(items[i], 1)}"
            for i in to_judge
        ]
        replies = judge.parallel(prompts, concurrency=concurrency, max_tokens=300)
        for i, rep in zip(to_judge, replies):
            if rep is None:
                judged[i] = ("other", JUDGE_ERROR_RATIONALE)
            else:
                judged[i] = _coerce_category(_parse_json_block(rep), new_taxonomy)

    out_rows: list[dict] = []
    n_coerced = n_judge_errors = 0
    for i, (row, item) in enumerate(zip(rows, items)):
        if i in judged:
            cat, rat = judged[i]
        else:
            cat = det[i] if det[i] is not None else "other"
            rat = row.get("rationale", "")
        if rat == JUDGE_ERROR_RATIONALE:
            n_judge_errors += 1
        elif isinstance(rat, str) and rat.startswith("[coerced from invented label"):
            n_coerced += 1
        out_rows.append({**item, "category": cat, "rationale": rat})
    return out_rows, n_coerced, n_judge_errors


def _default_refined_dir(source_dir: Union[str, None]) -> str:
    base = Path(source_dir) if source_dir else Path("taxonomy_runs") / "refine"
    return str(base.parent / f"{base.name}_refine-{uuid.uuid4().hex[:8]}")


def refine(run_or_dir: Union["RunResult", str, Path], feedback: str | None = None, *,
           operations: list[dict] | None = None,
           output_dir: Union[str, Path, None] = None,
           reclassify: str = "affected",
           judge_model: str = "deepseek/deepseek-v4-flash",
           orchestrator_model: str = "deepseek/deepseek-v4-flash",
           api_key: str | None = None,
           base_url: str = "https://openrouter.ai/api/v1",
           concurrency: int = 8,
           **run_kwargs) -> "RunResult":
    """Steer a finished run's taxonomy with natural-language ``feedback`` (or an
    explicit ``operations`` list) and return a new :class:`RunResult`.

    Args:
        run_or_dir: a :class:`RunResult` or a path to a completed run directory.
        feedback: natural-language guidance ("merge the two flattery
            categories", "split X", "too fine-grained"). Interpreted by a cheap
            judge call into typed edits, or into re-discovery guidance for
            open-ended asks.
        operations: skip interpretation and apply these typed edits directly
            (same DSL as ``revise_taxonomy``). Useful when you know the exact
            edit, and for deterministic, LLM-free refinement. Mutually exclusive
            with ``feedback``.
        output_dir: where to write the refined run. Defaults to a sibling of the
            source directory so the original is never overwritten.
        reclassify: ``"affected"`` (default) re-judges only items an edit could
            move; ``"all"`` relabels the whole corpus; ``"none"`` applies only
            deterministic relabels (no judge calls).
        judge_model / orchestrator_model / api_key / base_url / concurrency:
            model configuration, as in :func:`run`.
        **run_kwargs: forwarded to :func:`run` for the open-ended (re-discovery)
            path only.

    Returns:
        A :class:`RunResult` for the refined run, with a ``refine`` key recording
        the feedback, the applied operations, and how many items were re-judged.
    """
    if reclassify not in ("affected", "all", "none"):
        raise ValueError(
            f"reclassify must be 'affected', 'all', or 'none', got {reclassify!r}")
    if (feedback is None) == (operations is None):
        raise ValueError("pass exactly one of `feedback` or `operations`.")

    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing. Pass api_key= or set the env var.")

    loaded = _load_run(run_or_dir)
    out_dir = str(output_dir) if output_dir else _default_refined_dir(loaded["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    cost = CostTracker(orchestrator_model=orchestrator_model,
                       judge_model=judge_model, output_dir=out_dir)
    judge = Judge(api_key, judge_model, base_url=base_url,
                  usage_sink=cost.add_judge_usage)

    # Decide the edits.
    if operations is not None:
        interp = {"operations": operations}
    else:
        interp = interpret_feedback(feedback, loaded["taxonomy"], judge)

    # Open-ended feedback: warm-start a short discovery loop from the current
    # taxonomy with the guidance appended to the instruction.
    if interp.get("open_ended"):
        guidance = interp["guidance"]
        instr = (loaded["instruction"] or "Group these texts.").strip()
        instr = f"{instr}\n\nRefine the taxonomy per this feedback: {guidance}"
        return run(
            items=loaded["items"], instruction=instr, output_dir=out_dir,
            orchestrator_model=orchestrator_model, judge_model=judge_model,
            api_key=api_key, base_url=base_url, concurrency=concurrency,
            initial_taxonomy=loaded["taxonomy"], **run_kwargs)

    ops = interp["operations"]
    st = _TaxonomyState()
    st.taxonomy = [dict(c) for c in loaded["taxonomy"]]
    new_taxonomy, op_log = _apply_ops(st, ops)

    out_rows, n_coerced, n_judge_errors = _reclassify(
        loaded["rows"], loaded["items"], new_taxonomy, ops, judge,
        reclassify=reclassify, concurrency=concurrency)

    category_counts: dict[str, int] = {}
    for r in out_rows:
        category_counts[r["category"]] = category_counts.get(r["category"], 0) + 1

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    artifact = {
        "run_id": run_id,
        "n_items": len(out_rows),
        "n_coerced": n_coerced,
        "n_judge_errors": n_judge_errors,
        "taxonomy": new_taxonomy,
        "final_prompt": "(refined from a prior run)",
        "category_counts": category_counts,
        "classifications": out_rows,
    }
    artifact_path = os.path.join(out_dir, "taxonomy.json")
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)
    with open(os.path.join(out_dir, "classifications.jsonl"), "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "taxonomy_state.json"), "w") as f:
        json.dump({"taxonomy": new_taxonomy, "n_classify_calls": 0}, f, indent=2)

    n_judged = sum(1 for r_old, r_new in zip(loaded["rows"], out_rows)
                   if r_old.get("category") != r_new.get("category")
                   or r_old.get("rationale") != r_new.get("rationale"))
    cost.write()
    snapshot = cost.snapshot()
    meta = {
        "run_id": run_id,
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "instruction": loaded["instruction"],
        "n_items_input": len(out_rows),
        "orchestrator_model": orchestrator_model,
        "judge_model": judge_model,
        "status": "ok",
        "cost": snapshot,
        "refined_from": loaded["output_dir"],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return RunResult({
        "run_id": run_id,
        "output_dir": out_dir,
        "artifact_path": artifact_path,
        "artifact": artifact,
        "status": "ok",
        "cost": snapshot,
        "refine": {
            "feedback": feedback,
            "operations": ops,
            "op_log": op_log,
            "reclassify": reclassify,
            "n_reclassified": n_judged,
            "refined_from": loaded["output_dir"],
        },
    })
