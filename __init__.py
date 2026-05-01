"""Reusable agentic taxonomy-discovery + text-classification pipeline.

The orchestrator (a strong LLM, e.g. Claude Sonnet) iteratively probes random
batches of texts, edits a structured taxonomy via six tools, and once it
converges, runs a final classification of the entire pool through a cheaper
judge LLM (e.g. Llama 3.3 70B).

Public API:
    from taxonomy_agent import run

The package is fully self-contained — no imports from outside the folder.
"""
from .agent import run

__all__ = ["run"]
__version__ = "0.1.0"
