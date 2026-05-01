"""CLI: `python -m taxonomy_agent --config config.yaml` (or all flags).

Subcommands:
- `taxonomy ui`  → launches the Streamlit UI
- `taxonomy ...` → discovery run (the existing flag-based interface)

Precedence for the discovery run: CLI flag > config file > built-in default.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .agent import run


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
    # Inherit stdio so the user sees the URL Streamlit prints on startup.
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
        import json
        with open(p) as f:
            return json.load(f)
    sys.exit(f"unrecognized config extension: {p.suffix}. Use .yaml/.yml/.json.")


def main() -> None:
    # Subcommand dispatch — `taxonomy ui` (or `python -m taxonomy_agent ui`)
    # launches Streamlit and forwards any further args to it. Done before
    # argparse so streamlit-specific flags (e.g. --server.port) don't trip
    # the discovery-run argument parser.
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        _launch_ui(sys.argv[2:])
        return

    # Two-pass parse: pull --config first, then build the real parser using
    # config values as defaults so CLI flags override them.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None,
                     help="Path to a YAML or JSON config file.")
    pre_args, _ = pre.parse_known_args()

    cfg: dict = _load_config(pre_args.config) if pre_args.config else {}

    p = argparse.ArgumentParser(
        prog="taxonomy",
        description=(
            "Discover and apply an interpretable taxonomy in a text corpus. "
            "Run `taxonomy ui` to launch the Streamlit UI instead."
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
                   default=cfg.get("orchestrator", "anthropic/claude-sonnet-4.6"))
    p.add_argument("--judge",
                   default=cfg.get("judge", "meta-llama/llama-3.3-70b-instruct"))
    p.add_argument("--max-iters", type=int,
                   default=cfg.get("max_iters", 10))
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
                   default=cfg.get("size_hint", "4–10"),
                   help="Target taxonomy size for the orchestrator prompt "
                        "(e.g. '4–10', 'around 6', '3'). Pass an empty string "
                        "to drop the guidance entirely.")
    p.add_argument("--category-focus",
                   default=cfg.get("category_focus"),
                   help="What the taxonomy's categories should describe "
                        "(e.g. 'what each text is about' for topic modeling, "
                        "'the reasoning strategy each chain of thought uses' "
                        "for CoT analysis). Default: none — the instruction "
                        "carries the meaning on its own.")

    args = p.parse_args()

    # Required-field validation (config or CLI must supply).
    missing = []
    if not args.input:
        missing.append("input")
    if not args.output_dir:
        missing.append("output_dir")
    if not args.instruction and not args.instruction_file:
        missing.append("instruction or instruction_file")
    if missing:
        p.error("missing required field(s): " + ", ".join(missing)
                + ". Set them in --config or via CLI flags.")

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
        converge_below=args.threshold,
        probe_size=args.probe_size,
        pool_limit=args.pool_limit,
        recursion_limit=args.recursion_limit,
        concurrency=args.concurrency,
        size_hint=args.size_hint,
        category_focus=args.category_focus,
    )


if __name__ == "__main__":
    main()
