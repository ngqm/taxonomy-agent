"""Orchestrator setup and the main `run()` entry point."""
from __future__ import annotations

import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Iterable, Union

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .cost import CostTracker
from .judge import make_judge_caller
from .prompts import SYSTEM_PROMPT_TEMPLATE
from .tools import make_tools


def _load_items(items_or_path: Union[str, Path, Iterable[dict]]) -> list[dict]:
    """Accept either an in-memory iterable of dicts or a JSONL path.

    Item ids must be unique — duplicates would silently collapse in the
    id-keyed pool used by classify_with_judge, so we reject them up front."""
    if isinstance(items_or_path, (str, Path)):
        items: list[dict] = []
        seen: set[str] = set()
        with open(items_or_path) as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "id" not in obj:
                    raise ValueError(f"item on line {ln} missing 'id': {obj}")
                if "text" not in obj:
                    raise ValueError(f"item on line {ln} missing 'text': {obj}")
                obj["id"] = str(obj["id"])
                if obj["id"] in seen:
                    raise ValueError(f"duplicate id {obj['id']!r} on line {ln}")
                seen.add(obj["id"])
                items.append(obj)
        return items
    out = list(items_or_path)
    seen = set()
    for obj in out:
        if "id" not in obj or "text" not in obj:
            raise ValueError("every item must have 'id' and 'text'")
        obj["id"] = str(obj["id"])
        if obj["id"] in seen:
            raise ValueError(f"duplicate id: {obj['id']!r}")
        seen.add(obj["id"])
    return out


def run(
    items: Union[str, Path, Iterable[dict]],
    instruction: str,
    output_dir: Union[str, Path],
    *,
    orchestrator_model: str = "anthropic/claude-sonnet-4.6",
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
    prose_revise: bool = False,
) -> dict:
    """Discover a taxonomy of patterns in `items` and classify every item.

    Args:
        items: list/iterable of dicts (each must have `id` and `text`) OR a JSONL path.
            Any extra keys per item are passed to the judge as context and copied into
            the output classification rows.
        instruction: short natural-language description of what to classify
            (e.g. "Identify the rhetorical strategy used to redirect from the question.").
        output_dir: directory for taxonomy.json and trace.jsonl.
        orchestrator_model, judge_model: OpenRouter model IDs.
        max_iterations: hard cap on the discovery loop.
        min_iterations: floor on the number of `classify_with_judge` rounds
            before `finalize_classify` is allowed. Guards against premature
            convergence on a lucky early probe. Default 3. Must be ≤
            max_iterations.
        converge_below: don't-fit rate threshold for early stop (0.10 = 10%).
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

    items_list = _load_items(items)
    if pool_limit is not None and pool_limit > 0:
        items_list = items_list[:pool_limit]
    if not items_list:
        raise ValueError("no items to classify.")

    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    print(f"[taxonomy_agent] items={len(items_list)} run_id={run_id}")
    print(f"[taxonomy_agent] orchestrator={orchestrator_model}, judge={judge_model}")
    print(f"[taxonomy_agent] output_dir={output_dir}")

    # Write meta.json so the UI's Runs tab can list the run before
    # finalize_classify writes taxonomy.json. Status is updated at the end.
    meta_path = os.path.join(output_dir, "meta.json")
    meta = {
        "run_id": run_id,
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "instruction": instruction.strip(),
        "n_items_input": len(items_list),
        "orchestrator_model": orchestrator_model,
        "judge_model": judge_model,
        "size_hint": size_hint,
        "category_focus": category_focus,
        "min_iterations": min_iterations,
        "prose_revise": prose_revise,
        "status": "running",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    cost = CostTracker(
        orchestrator_model=orchestrator_model,
        judge_model=judge_model,
        output_dir=output_dir,
    )
    cost.write()  # zero-state cost.json so the UI can read it immediately

    judge_call, judge_parallel = make_judge_caller(
        api_key, judge_model, base_url=base_url, usage_sink=cost.add_judge_usage,
    )
    tools, force_finalize = make_tools(
        items_list, run_id, output_dir, judge_call, judge_parallel,
        concurrency=concurrency, max_iters=max_iterations,
        min_iterations=min_iterations, prose_revise=prose_revise,
    )

    llm = ChatOpenAI(
        model=orchestrator_model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        # Forward `usage: {include: true}` so OpenRouter returns the actual
        # charge under usage.cost — CostTracker prefers this over the static
        # MODEL_PRICES fallback. Harmless for endpoints that ignore it.
        extra_body={"usage": {"include": True}},
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
    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        instruction=instruction.strip(),
        n_items=len(items_list),
        threshold=converge_below,
        probe_size=probe_size,
        max_iters=max_iterations,
        min_iters=min_iterations,
        size_aside=size_aside,
        focus_bullet=focus_bullet,
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
        print(f"[taxonomy_agent] orchestrator stream raised: {e!r} — flushing "
              f"partial state and exiting.")

    artifact_path = os.path.join(output_dir, "taxonomy.json")
    out: dict = {"run_id": run_id, "output_dir": output_dir, "artifact_path": artifact_path}
    if os.path.exists(artifact_path) and stream_error is None:
        with open(artifact_path) as f:
            out["artifact"] = json.load(f)
        out["status"] = "ok"
        print(f"[taxonomy_agent] done → {artifact_path}")
    else:
        # The orchestrator may have walked off without calling finalize_classify
        # (e.g. monotonic-add loops that never trip the don't-fit threshold).
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
                print(f"[taxonomy_agent] orchestrator did not call "
                      f"finalize_classify; auto-finalized against the current "
                      f"on-disk taxonomy → {artifact_path}")
        except Exception as ff_err:
            print(f"[taxonomy_agent] auto-finalize fallback raised: {ff_err!r}")
        if not auto_finalized:
            out["status"] = "incomplete" if stream_error is None else "error"
            if stream_error is not None:
                out["error"] = repr(stream_error)
            print(f"[taxonomy_agent] WARNING: no artifact at {artifact_path}. "
                  f"Status={out['status']}. The orchestrator may have hit the "
                  f"recursion limit, the classify budget, or an LLM error "
                  f"mid-run, and the auto-finalize fallback could not produce "
                  f"a taxonomy either (likely an empty taxonomy state). "
                  f"Partial state is in {output_dir}/taxonomy_state.json and "
                  f"{output_dir}/classifications.jsonl.")

    cost.write()
    cost_snapshot = cost.snapshot()
    out["cost"] = cost_snapshot
    meta["status"] = out["status"]
    meta["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta["cost"] = cost_snapshot
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if cost_snapshot["total_usd"] is not None:
        print(f"[taxonomy_agent] cost: ${cost_snapshot['total_usd']:.4f} "
              f"(orch={cost_snapshot['orchestrator']['n_calls']} calls, "
              f"judge={cost_snapshot['judge']['n_calls']} calls)")
    else:
        print(f"[taxonomy_agent] tokens recorded; USD unknown for one or both "
              f"models (not in cost.MODEL_PRICES). See {output_dir}/cost.json.")

    return out
