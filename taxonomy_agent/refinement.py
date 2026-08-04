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

from dotenv import load_dotenv

from .agent import RunResult, run
from .cost import CostTracker
from .judge import Judge
from .tools import (DEFAULT_CLASSIFY_PROMPT, ESCAPE_HATCH_SUFFIX,
                    JUDGE_ERROR_RATIONALE, _TaxonomyState, _apply_ops,
                    _coerce_category, _format_taxonomy, _parse_json_block,
                    build_artifact, build_classify_prompt, write_taxonomy_state)

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
    taxonomy = res.taxonomy
    rows = res.classifications
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
                concurrency: int) -> tuple[list[dict], int]:
    """Produce the refined classification rows and how many changed.
    Deterministically-relabelled items keep their prior rationale; re-judged
    items get a fresh label from the judge."""
    new_names = {c["name"] for c in new_taxonomy}
    det, need = _replay_labels(rows, operations, new_names, reclassify)

    judged: dict[int, tuple[str, str]] = {}
    to_judge = [i for i, flag in enumerate(need) if flag]
    if to_judge:
        tax_str = _format_taxonomy(new_taxonomy)
        instruction = DEFAULT_CLASSIFY_PROMPT + ESCAPE_HATCH_SUFFIX
        prompts = [build_classify_prompt(instruction, tax_str, items[i])
                   for i in to_judge]
        replies = judge.parallel(prompts, concurrency=concurrency, max_tokens=300)
        for i, rep in zip(to_judge, replies):
            judged[i] = (("other", JUDGE_ERROR_RATIONALE) if rep is None
                         else _coerce_category(_parse_json_block(rep), new_taxonomy))

    out_rows: list[dict] = []
    n_changed = 0
    for i, row in enumerate(rows):
        if i in judged:
            cat, rat = judged[i]
        else:
            cat = det[i] if det[i] is not None else "other"
            rat = row.get("rationale", "")
        new_row = {**row, "category": cat, "rationale": rat}
        out_rows.append(new_row)
        if row.get("category") != cat or row.get("rationale") != rat:
            n_changed += 1
    return out_rows, n_changed


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

    load_dotenv(override=False)  # match run(): honour a project .env
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

    # Decide the edits. Explicit `operations` skip the interpreter entirely;
    # natural-language feedback is interpreted into typed edits, or — when it's
    # open-ended — into guidance that warm-starts a short re-discovery loop.
    if operations is not None:
        ops = operations
    else:
        interp = interpret_feedback(feedback, loaded["taxonomy"], judge)
        if interp.get("open_ended"):
            instr = (loaded["instruction"] or "Group these texts.").strip()
            instr = (f"{instr}\n\nRefine the taxonomy per this feedback: "
                     f"{interp['guidance']}")
            return run(
                items=loaded["items"], instruction=instr, output_dir=out_dir,
                orchestrator_model=orchestrator_model, judge_model=judge_model,
                api_key=api_key, base_url=base_url, concurrency=concurrency,
                initial_taxonomy=loaded["taxonomy"], **run_kwargs)
        ops = interp["operations"]

    st = _TaxonomyState()
    st.taxonomy = [dict(c) for c in loaded["taxonomy"]]
    new_taxonomy, op_log = _apply_ops(st, ops)

    out_rows, n_changed = _reclassify(
        loaded["rows"], loaded["items"], new_taxonomy, ops, judge,
        reclassify=reclassify, concurrency=concurrency)

    # Persist the refined run through the same artifact/state writers the main
    # loop uses, so its output directory is shape-identical to a run().
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    artifact = build_artifact(run_id, out_rows, new_taxonomy,
                              "(refined from a prior run)")
    artifact_path = os.path.join(out_dir, "taxonomy.json")
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)
    with open(os.path.join(out_dir, "classifications.jsonl"), "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    write_taxonomy_state(os.path.join(out_dir, "taxonomy_state.json"), new_taxonomy)

    cost.write()
    snapshot = cost.snapshot()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    meta = {
        "run_id": run_id,
        "started_at": now,
        "finished_at": now,
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
            "n_reclassified": n_changed,
            "refined_from": loaded["output_dir"],
        },
    })
