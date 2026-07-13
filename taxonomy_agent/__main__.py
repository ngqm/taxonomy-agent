"""CLI for taxonomy_agent.

Quick start:
    taxonomy demo                          # bundled 48-item DarkBench slice
    taxonomy run corpus.jsonl              # sensible defaults; cheap models
    taxonomy run corpus.jsonl -g "Goal"    # positional corpus + goal flag
    taxonomy run --quality corpus.jsonl    # Sonnet orchestrator for better quality
    taxonomy ui                            # launch the Streamlit walk-up app
    taxonomy inspect ./run_dir/            # text summary of a finished run

Power-user mode (still supported):
    taxonomy --config config.yaml          # YAML config with every knob
    taxonomy --input foo.jsonl --instruction "..." --output-dir out/  ...

Precedence for the discovery run: CLI flag > config file > built-in default.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .agent import run


# Defaults — keep aligned with the eval pair and the install default.
DEFAULT_ORCHESTRATOR_CHEAP = "deepseek/deepseek-v4-flash"
DEFAULT_ORCHESTRATOR_QUALITY = "anthropic/claude-sonnet-4.6"
DEFAULT_JUDGE = "deepseek/deepseek-v4-flash"
DEFAULT_GOAL = "Identify the topic of each text."


def _launch_ui(extra_args: list[str]) -> None:
    """Launch the Streamlit app, forwarding any extra args to streamlit itself."""
    streamlit = shutil.which("streamlit")
    if streamlit is None:
        sys.exit(
            "streamlit is not installed. Install it with:\n"
            "  pip install streamlit\n"
            "or, for the full stack:\n"
            "  pip install -r taxonomy_agent/requirements.txt"
        )
    app_path = Path(__file__).resolve().parent / "app.py"
    if not app_path.exists():
        sys.exit(f"could not find Streamlit app at {app_path}")
    cmd = [streamlit, "run", str(app_path), *extra_args]
    raise SystemExit(subprocess.call(cmd))


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        sys.exit(f"config file not found: {path}")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            sys.exit("PyYAML not installed. `pip install pyyaml` or use a JSON config.")
        with open(p) as f:
            return yaml.safe_load(f) or {}
    if p.suffix.lower() == ".json":
        with open(p) as f:
            return json.load(f)
    sys.exit(f"unrecognized config extension: {p.suffix}. Use .yaml/.yml/.json.")


def _cmd_demo(argv: list[str]) -> None:
    """Run the bundled DarkBench-manipulation demo with default models."""
    p = argparse.ArgumentParser(
        prog="taxonomy demo",
        description=(
            "Run the bundled demo — a balanced 48-prompt slice of DarkBench — "
            "with the install default models, recovering the manipulation "
            "patterns from an unlabelled corpus. Five to ten minutes on a warm "
            "OpenRouter connection. Requires OPENROUTER_API_KEY in env."
        ),
    )
    p.add_argument("-o", "--output", default=None,
                   help="Output directory (default: ./taxonomy_demo_run/).")
    p.add_argument("--quality", action="store_true",
                   help="Use the Sonnet orchestrator instead of the cheap default.")
    args = p.parse_args(argv)

    pkg_root = Path(__file__).resolve().parent
    items_path = pkg_root / "example" / "items.jsonl"
    instr_path = pkg_root / "example" / "instruction.txt"
    if not items_path.exists() or not instr_path.exists():
        sys.exit(f"bundled demo files missing under {pkg_root / 'example'}")

    out = args.output or "./taxonomy_demo_run"
    instruction = instr_path.read_text().strip()
    orchestrator = (DEFAULT_ORCHESTRATOR_QUALITY if args.quality
                    else DEFAULT_ORCHESTRATOR_CHEAP)
    n_items = sum(1 for line in items_path.open() if line.strip())
    print(f"[demo] corpus={items_path} ({n_items} items)")
    print(f"[demo] orchestrator={orchestrator}")
    print(f"[demo] judge={DEFAULT_JUDGE}")
    print(f"[demo] output={out}")
    run(
        items=str(items_path),
        instruction=instruction,
        output_dir=out,
        orchestrator_model=orchestrator,
        judge_model=DEFAULT_JUDGE,
        min_iterations=1,
        probe_size=5,
    )
    print(f"[demo] done. Inspect with: taxonomy inspect {out}")


def _cmd_run(argv: list[str]) -> None:
    """Friendly run mode: `taxonomy run corpus.jsonl [flags]`."""
    p = argparse.ArgumentParser(
        prog="taxonomy run",
        description=(
            "Discover an interpretable taxonomy in a JSONL corpus. "
            "Defaults to the cheap model pair for both roles."
        ),
    )
    p.add_argument("corpus", nargs="?",
                   help="Path to a .jsonl / .json / .csv corpus (a text field "
                        "or column per item; id optional).")
    p.add_argument("-g", "--goal", default=None,
                   help="One-sentence goal instruction (default: topic discovery).")
    p.add_argument("-o", "--output", default=None,
                   help="Output directory (default: ./taxonomy_run_<timestamp>/).")
    p.add_argument("--quality", action="store_true",
                   help="Use the Sonnet orchestrator (~$1-2 per 500-item run).")
    p.add_argument("--orchestrator", default=None,
                   help="Override the orchestrator model id.")
    p.add_argument("--judge", default=None,
                   help="Override the judge model id.")
    p.add_argument("--size", default="4-10", dest="size_hint",
                   help="Target taxonomy size (e.g. '4-10', '3'). '' to disable.")
    p.add_argument("--max-iters", type=int, default=10)
    p.add_argument("--min-iters", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("--probe-size", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args(argv)

    if not args.corpus:
        p.error("missing positional argument: corpus (.jsonl / .json / .csv path)")
    if not Path(args.corpus).exists():
        sys.exit(f"corpus file not found: {args.corpus}")

    out = args.output
    if out is None:
        from datetime import datetime
        out = f"./taxonomy_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    orchestrator = args.orchestrator or (
        DEFAULT_ORCHESTRATOR_QUALITY if args.quality
        else DEFAULT_ORCHESTRATOR_CHEAP
    )
    judge = args.judge or DEFAULT_JUDGE
    instruction = args.goal or DEFAULT_GOAL

    print(f"[run] corpus={args.corpus}")
    print(f"[run] goal={instruction}")
    print(f"[run] orchestrator={orchestrator}")
    print(f"[run] judge={judge}")
    print(f"[run] output={out}")
    run(
        items=args.corpus,
        instruction=instruction,
        output_dir=out,
        orchestrator_model=orchestrator,
        judge_model=judge,
        max_iterations=args.max_iters,
        min_iterations=args.min_iters,
        converge_below=args.threshold,
        probe_size=args.probe_size,
        concurrency=args.concurrency,
        size_hint=args.size_hint or None,
    )
    print(f"[run] done. Inspect with: taxonomy inspect {out}")


def _cmd_inspect(argv: list[str]) -> None:
    """Print a text summary of a finished run directory."""
    p = argparse.ArgumentParser(
        prog="taxonomy inspect",
        description="Print a text summary of a finished taxonomy_agent run directory.",
    )
    p.add_argument("run_dir", help="Path to a run output directory.")
    p.add_argument("--full", action="store_true",
                   help="Also print every category description and item count.")
    args = p.parse_args(argv)

    rd = Path(args.run_dir)
    if not rd.exists():
        sys.exit(f"run directory not found: {rd}")

    state_path = rd / "taxonomy_state.json"
    cost_path = rd / "cost.json"
    cls_path = rd / "classifications.jsonl"
    meta_path = rd / "meta.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"=== run {meta.get('run_id', rd.name)} ===")
        print(f"status:       {meta.get('status', '?')}")
        print(f"instruction:  {meta.get('instruction', '?')}")
        print(f"orchestrator: {meta.get('orchestrator_model', '?')}")
        print(f"judge:        {meta.get('judge_model', '?')}")
        print(f"n_items:      {meta.get('n_items_input', '?')}")
    else:
        print(f"=== run {rd.name} ===")
        print("(no meta.json — run may be in progress or pre-2026 layout)")

    if cost_path.exists():
        cost = json.loads(cost_path.read_text())
        total = cost.get("total_usd", 0)
        orch = cost.get("orchestrator", {})
        judge = cost.get("judge", {})
        print(f"\ncost:         USD {total:.4f}")
        if orch:
            print(f"  orchestrator: USD {orch.get('usd', 0):.4f} ({orch.get('n_calls', 0)} calls)")
        if judge:
            print(f"  judge:        USD {judge.get('usd', 0):.4f} ({judge.get('n_calls', 0)} calls)")
    else:
        print("\ncost:         (no cost.json)")

    if state_path.exists():
        state = json.loads(state_path.read_text())
        tax = state.get("taxonomy") or state.get("categories") or []
        print(f"\ntaxonomy: {len(tax)} categories")
        for i, cat in enumerate(tax, 1):
            name = cat.get("name", f"cat_{i}")
            print(f"  {i:2d}. {name}")
            if args.full:
                desc = cat.get("description", "")
                if desc:
                    print(f"      {desc}")
    else:
        print("\ntaxonomy: (no taxonomy_state.json — run incomplete)")

    if cls_path.exists():
        lines = cls_path.read_text().splitlines()
        print(f"\nclassifications: {len(lines)} items labelled")
        if args.full and lines:
            cats: dict[str, int] = {}
            for ln in lines:
                row = json.loads(ln)
                c = row.get("category", "other")
                cats[c] = cats.get(c, 0) + 1
            for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
                print(f"  {n:4d}  {cat}")
    else:
        print("\nclassifications: (none yet)")

    print()


def _cmd_legacy(argv: list[str]) -> None:
    """Original flag-based interface, preserved for config and scripted use."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None,
                     help="Path to a YAML or JSON config file.")
    pre_args, _ = pre.parse_known_args(argv)

    cfg: dict = _load_config(pre_args.config) if pre_args.config else {}

    p = argparse.ArgumentParser(
        prog="taxonomy",
        description=(
            "Discover and apply an interpretable taxonomy in a text corpus. "
            "Use `taxonomy demo` for an instant try, `taxonomy run corpus.jsonl` "
            "for a friendlier interface, or `taxonomy ui` for the Streamlit app."
        ),
    )
    p.add_argument("--config", default=pre_args.config,
                   help="Path to a YAML/JSON config file (CLI flags override its values).")

    p.add_argument("--input", default=cfg.get("input"),
                   help="Path to a JSONL file. Each line: {\"id\": ..., \"text\": ..., ...}")
    p.add_argument("--instruction", default=cfg.get("instruction"),
                   help="Natural-language classification goal.")
    p.add_argument("--instruction-file", default=cfg.get("instruction_file"),
                   help="Path to a text file containing the instruction.")
    p.add_argument("--output-dir", default=cfg.get("output_dir"),
                   help="Directory for outputs.")

    p.add_argument("--orchestrator",
                   default=cfg.get("orchestrator", DEFAULT_ORCHESTRATOR_CHEAP))
    p.add_argument("--judge",
                   default=cfg.get("judge", DEFAULT_JUDGE))
    p.add_argument("--max-iters", type=int,
                   default=cfg.get("max_iters", 10))
    p.add_argument("--min-iters", type=int,
                   default=cfg.get("min_iters", 3),
                   help="Minimum classify_with_judge rounds before "
                        "finalize_classify is allowed (default 3). Prevents "
                        "premature convergence on a lucky early probe.")
    p.add_argument("--threshold", type=float,
                   default=cfg.get("threshold", 0.10),
                   help="Don't-fit rate for early stop (default 0.10 = 10%%)")
    p.add_argument("--probe-size", type=int,
                   default=cfg.get("probe_size", 20),
                   help="K — items per discovery probe batch")
    p.add_argument("--pool-limit", type=int,
                   default=cfg.get("pool_limit"),
                   help="Cap pool size (smoke testing)")
    p.add_argument("--recursion-limit", type=int,
                   default=cfg.get("recursion_limit", 80))
    p.add_argument("--concurrency", type=int,
                   default=cfg.get("concurrency", 8))
    p.add_argument("--size-hint",
                   default=cfg.get("size_hint", "4-10"),
                   help="Target taxonomy size for the orchestrator prompt "
                        "(e.g. '4-10', 'around 6', '3'). Pass an empty string "
                        "for no target.")
    p.add_argument("--category-focus",
                   default=cfg.get("category_focus"),
                   help="What the taxonomy's categories should describe.")

    args = p.parse_args(argv)

    missing = []
    if not args.input:
        missing.append("input")
    if not args.output_dir:
        missing.append("output_dir")
    if not args.instruction and not args.instruction_file:
        missing.append("instruction or instruction_file")
    if missing:
        p.error("missing required field(s): " + ", ".join(missing)
                + ". Set them in --config or via CLI flags, "
                "or try `taxonomy run corpus.jsonl` for a friendlier interface.")

    instruction = args.instruction
    if not instruction and args.instruction_file:
        with open(args.instruction_file) as f:
            instruction = f.read().strip()

    run(
        items=args.input,
        instruction=instruction,
        output_dir=args.output_dir,
        orchestrator_model=args.orchestrator,
        judge_model=args.judge,
        max_iterations=args.max_iters,
        min_iterations=args.min_iters,
        converge_below=args.threshold,
        probe_size=args.probe_size,
        pool_limit=args.pool_limit,
        recursion_limit=args.recursion_limit,
        concurrency=args.concurrency,
        size_hint=args.size_hint,
        category_focus=args.category_focus,
    )


SUBCOMMANDS = {
    "ui": _launch_ui,
    "demo": _cmd_demo,
    "run": _cmd_run,
    "inspect": _cmd_inspect,
}


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in SUBCOMMANDS:
        SUBCOMMANDS[sys.argv[1]](sys.argv[2:])
        return
    _cmd_legacy(sys.argv[1:])


if __name__ == "__main__":
    main()
