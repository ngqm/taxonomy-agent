"""Baseline taxonomy/clustering methods for benchmark comparison.

Each baseline subclasses `Baseline` and is registered by name. The runner looks
one up with `get_baseline(name)` instead of a hardcoded dispatch chain, so a new
baseline is added by writing the class and listing it below. Importing the
modules is cheap; their heavy dependencies load lazily inside `run`.
"""
from __future__ import annotations

from .base import Baseline
from .bertopic_baseline import BERTopicBaseline
from .embed_cluster_llm import EmbedClusterLabelBaseline
from .lda_baseline import LDABaseline
from .single_shot_llm import SingleShotBaseline
from .topicgpt_style import TopicGPTStyleBaseline

_BASELINES: list[Baseline] = [
    BERTopicBaseline(),
    LDABaseline(),
    SingleShotBaseline(),
    TopicGPTStyleBaseline(),
    EmbedClusterLabelBaseline(),
]
REGISTRY: dict[str, Baseline] = {b.name: b for b in _BASELINES}


def get_baseline(name: str) -> Baseline:
    """Return the registered baseline, or raise KeyError listing the known ones."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown baseline {name!r}; known: {sorted(REGISTRY)}") from None


__all__ = [
    "Baseline", "REGISTRY", "get_baseline",
    "BERTopicBaseline", "LDABaseline", "SingleShotBaseline",
    "TopicGPTStyleBaseline", "EmbedClusterLabelBaseline",
]
