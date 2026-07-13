"""CLI entry point: python -m taxonomy_agent.eval --corpus 20ng ..."""
from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from .runner import benchmark


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)
    p = argparse.ArgumentParser(prog="taxonomy_agent.eval")
    p.add_argument("--corpus", default="20ng")
    p.add_argument("--methods", default="taxonomy_agent,bertopic,topicgpt_style,single_shot",
                   help="comma-separated method names")
    p.add_argument("--seeds", default="42,43,44",
                   help="comma-separated seeds")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--instruction", default="Identify the topic of each text.")
    p.add_argument("--model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--orchestrator", default="anthropic/claude-sonnet-4.6",
                   help="orchestrator model for taxonomy_agent method")
    p.add_argument("--n-per-class", type=int, default=50)
    p.add_argument("--reasoning-path", default=None,
                   help="path to reasoning-strategies JSONL "
                        "(only used when --corpus reasoning)")
    p.add_argument("--size-hint", default=None,
                   help="taxonomy size hint for taxonomy_agent; pass '' for no "
                        "hint (let it discover the category count freely)")
    p.add_argument("--dry-run", action="store_true",
                   help="skip LLM/heavy calls, use canned predictions")
    args = p.parse_args(argv)

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    extra: dict = {}
    if args.reasoning_path:
        extra["synth_path"] = args.reasoning_path
    if args.size_hint is not None:
        extra["size_hint"] = args.size_hint

    res = benchmark(
        corpus_name=args.corpus,
        methods=methods,
        seeds=seeds,
        output_dir=args.output_dir,
        instruction=args.instruction,
        model=args.model,
        orchestrator_model=args.orchestrator,
        api_key=api_key,
        n_per_class=args.n_per_class,
        dry_run=args.dry_run,
        **extra,
    )
    print(f"wrote {len(res['rows'])} runs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
