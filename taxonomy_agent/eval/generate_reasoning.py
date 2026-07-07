"""Standalone CLI: build the synthetic CoT-patterns corpus via OpenRouter.

Example:
    python -m taxonomy_agent.eval.generate_reasoning \\
        --out eval_data/cot_patterns.jsonl \\
        --n-per-source 30 \\
        --generator google/gemini-3.1-flash-lite \\
        --verifier deepseek/deepseek-v4-flash \\
        --seed 42 \\
        --concurrency 8
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from .synth_reasoning import (
    DEFAULT_CORPUS_PATH,
    GENERATOR_MODEL_DEFAULT,
    VERIFIER_MODEL_DEFAULT,
    generate_corpus,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="taxonomy_agent.eval.generate_reasoning")
    p.add_argument("--out", default=DEFAULT_CORPUS_PATH,
                   help="output JSONL path")
    p.add_argument("--n-per-source", type=int, default=30,
                   help="problems sampled per corpus source (3 sources total)")
    p.add_argument("--generator", default=GENERATOR_MODEL_DEFAULT,
                   help="OpenRouter model id for CoT generation")
    p.add_argument("--verifier", default=VERIFIER_MODEL_DEFAULT,
                   help="OpenRouter model id for strategy-blind verification")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for sampling + prompt variation")
    p.add_argument("--concurrency", type=int, default=8,
                   help="parallel OpenRouter requests")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    args = p.parse_args(argv)

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set (env or .env). Aborting.",
              file=sys.stderr)
        return 2

    try:
        summary = generate_corpus(
            out_path=args.out,
            n_per_source=args.n_per_source,
            generator_model=args.generator,
            verifier_model=args.verifier,
            api_key=api_key,
            base_url=args.base_url,
            seed=args.seed,
            concurrency=args.concurrency,
        )
    except Exception as e:
        print(f"generate_corpus failed: {e!r}", file=sys.stderr)
        return 1

    print()
    print("=== summary ===")
    print(f"candidates: {summary['n_candidates']}")
    print(f"survived:   {summary['n_survived']}")
    print(f"kappa:      {summary['agreement_rate']:.3f}")
    print(f"cost:       ${summary['total_cost']:.4f}")
    print(f"out:        {summary['out_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
