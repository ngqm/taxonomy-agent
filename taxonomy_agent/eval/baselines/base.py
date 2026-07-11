"""Common interface for the taxonomy-discovery baselines.

Each baseline is a strategy that, given a corpus (and, for the LLM baselines,
a goal instruction), produces a taxonomy and a per-item assignment. Subclasses
set ``name`` and implement ``run``; the runner dispatches through the registry
in ``baselines/__init__.py`` instead of a hardcoded if/elif chain.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Baseline(ABC):
    #: registry key, e.g. "bertopic"
    name: str = ""
    #: True if run() reads the goal instruction (the LLM baselines)
    uses_instruction: bool = False

    @abstractmethod
    def run(self, items: list[dict], *, instruction: str = "", seed: int = 42,
            model: str = "", api_key: str | None = None, **kwargs) -> dict:
        """Discover a taxonomy over ``items`` and label each one.

        Returns a dict with keys ``taxonomy``, ``assignments``, ``cost_usd``,
        and ``wall_time_s``. Baselines ignore keyword arguments they do not use
        (e.g. the classical models ignore ``instruction`` / ``model``).
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
