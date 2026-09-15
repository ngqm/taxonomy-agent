"""DynaMiTE baseline (Balepur et al., Findings of ACL 2023, "DynaMiTE:
Discovering Explosive Topic Evolutions with User Guidance").

STATUS: BLOCKED for this benchmark. DynaMiTE is a *dynamic* (temporal) topic
method and does not produce the static per-document category assignment this
harness scores. This adapter is intentionally a documented stub: it is wired
into the registry so the method is accounted for, but ``run`` raises with the
reasoning rather than fabricating a misleading result.

Upstream: https://github.com/nbalepur/DynaMiTE (commit e205241).

Why it does not fit a static single-corpus assignment
-----------------------------------------------------
1. Temporal by construction. The model requires every document to carry a
   ``time_discrete`` ordinal (the paper uses T = 2012..2022, 11 steps). Training
   learns *dynamic* word embeddings across time with a temporal-smoothness
   regularizer (``tau``) that couples U/V at t-1, t, t+1. Collapsing the corpus
   to a single time step makes the temporal contrastive objective degenerate --
   the method reduces to a static, seed-guided embedding, which is exactly what
   the CatE baseline already provides.

2. No document -> category output. DynaMiTE's output is *topic-word
   evolutions*: per time step, per (seed-guided) category, a ranked list of
   expanded seed words (``TopicRanker.print_seeds``). ``eval.py`` computes NPMI
   over those word lists and ``shift_study.py`` studies category drift. Nothing
   in the pipeline assigns a document to a category, so producing the
   ``assignments`` this harness needs would require bolting on the same
   mean-word-embedding -> nearest-category readout that CatE already does --
   again collapsing DynaMiTE to a redundant CatE.

3. Heavy, non-reproducible preprocessing. The pipeline requires AutoPhrase (a
   separate Java/C++ tool, ``AutoPhrase.zip`` from the CatE repo; the README
   quotes ~15 min/dataset) plus per-time PPMI matrices, temporal TF-IDF, and
   static-init embeddings. requirements.txt pins Python 3.8.10 with
   scikit-learn==0.22, numpy==1.23, torch==1.13, tensorflow==2.11, spherecluster,
   rapids/cudf, etc. -- incompatible with this project's Python 3.12 stack.

Like CatE, DynaMiTE is SEED-ADVANTAGED: it is given the gold category names as
user-provided seeds. If a temporal variant is ever wanted, it would need a
corpus with real timestamps and would be evaluated on topic-word coherence
(NPMI) over time, not on this harness's static assignment metrics.
"""
from __future__ import annotations

from .base import Baseline

_BLOCKER_MSG = (
    "DynaMiTE is a temporal (dynamic) topic-evolution method and does not "
    "produce a static per-document category assignment. It requires a "
    "'time_discrete' timestamp per document, trains dynamic word embeddings "
    "across time with a temporal regularizer, and outputs ranked topic-word "
    "evolutions per time step (not document labels). Forcing it into a single "
    "time step + nearest-category readout would collapse it to a redundant "
    "CatE. It also depends on AutoPhrase (Java) and a Python-3.8 / "
    "scikit-learn-0.22 / torch-1.13 stack incompatible with this environment. "
    "See taxonomy_agent/eval/baselines/dynamite_baseline.py for the full "
    "reasoning. Blocked by design; not run for this benchmark."
)


class DynaMiTEBaseline(Baseline):
    """Seed-guided *temporal* topic evolution (Balepur et al., ACL Findings 2023).

    BLOCKED: fundamentally temporal, emits topic-word evolutions rather than
    document assignments. See module docstring. SEED-ADVANTAGED by design.
    """
    name = "dynamite"
    uses_instruction = False

    def run(self, items, *, instruction="", seed=42, model="", api_key=None,
            **kwargs):
        raise NotImplementedError(_BLOCKER_MSG)
