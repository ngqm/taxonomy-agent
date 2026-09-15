"""Agentic taxonomy discovery and text classification.

An orchestrator LLM iteratively probes random batches of a corpus, edits a
structured taxonomy through six tools, and once the unmatched rate converges a
cheaper, stateless judge LLM labels every item. Both roles default to
DeepSeek-v4-Flash over OpenRouter, so a run on a few hundred items costs well
under $0.20.

Public API:
    from taxonomy_agent import run, refine, RunResult

    result = run(items, instruction, output_dir="out/")
    result.taxonomy          # [{"name", "description"}, ...]
    result.definitions       # {category: definition}
    result.classifications   # [{id, text, category, rationale}, ...]  (streams)
    result.to_dataframe()    # id, text, category, rationale, definition
    result.save_csv("labels.csv")
    result.cost_usd          # OpenRouter spend, in USD

    RunResult.from_dir("out/")                 # reload a finished run, no re-spend
    better = refine(result, "merge the two flattery categories")

See the run() docstring for the full parameter set: scaling via
finalize="embed"/"finetune", multi_label, opt-in web search, and more.
"""
from .agent import run, RunResult, open_corpus
from .refinement import refine, interpret_feedback

__all__ = ["run", "RunResult", "refine", "interpret_feedback", "open_corpus"]
__version__ = "0.1.0"
