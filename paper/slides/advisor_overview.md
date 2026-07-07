# taxonomy_agent — advisor overview

A 10-slide outline. Each `---` separates a slide. Drop into Keynote / Google
Slides / Marp / Pandoc-Beamer as you prefer; numbers match the Beamer
deck at `advisor_overview.tex` (commit `dae9bd6`).

---

## Slide 1 — Title

**taxonomy_agent: An Orchestrator–Judge Agent for Taxonomy Discovery**

EMNLP 2026 System Demonstrations (deadline 2026-07-10)

Quang Minh Nguyen — KAIST

---

## Slide 2 — Problem

- Many applied NLP workflows start with a labelled taxonomy: annotation
  guidelines, qualitative-research codebooks, dashboard schemas,
  moderation policies.
- Building one means reading hundreds of docs, drafting categories,
  arguing, iterating. Slow, expensive, brittle.
- LLMs can replace much of that labour — but published pipelines lump
  the proposer and the labeller into a single agent that can
  hallucinate and force its own labels.

**Our question:** can structural choices about *how* the proposer and
the labeller interact give us a cheap, honest discovery system?

---

## Slide 3 — Two structural commitments

1. **Typed dispatcher** with validate-before-mutate and per-operation
   failure isolation.
   - Six taxonomy verbs: add / rename / edit / merge / split / drop.
   - Malformed edit returns unchanged taxonomy + log entry the
     orchestrator reads next step. No silent corruption.

2. **Stateless judge LLM.**
   - Labels each item against the current taxonomy from item text alone.
   - Never sees orchestrator's tool-call history or prior labels.
   - Runs on the cheap model, called once per item — linear in corpus.

---

## Slide 4 — Architecture

```
       ┌─────────────────┐                    ┌──────────────────┐
       │ Orchestrator    │ ── typed edits ──> │ Working taxonomy │
       │ LLM (ReAct,     │                    │ (closure state)  │
       │  6 tools)       │                    └──────────┬───────┘
       └────┬────────────┘                               │
            ▲                                  snapshot  ▼
            │ don't-fit rate                  ┌──────────────────┐
            │                                 │ Judge LLM        │
            │                                 │ stateless        │
            │                                 │ per item         │
            │                                 └────┬─────────────┘
            │                                      │
            │                              item text│  label
            │                                      ▼
            │                                 ┌─────────┐
            └─────────────────────────────────│ Corpus  │
                                              └─────────┘
```

The judge re-derives every label from item text against the
orchestrator's current taxonomy. The orchestrator cannot force a
label past it.

---

## Slide 5 — 20NG results (5 seeds, install default)

| Method | NMI ↑ | ARI ↑ | Cost (USD) |
|---|---|---|---|
| **taxonomy_agent (cheap pair)** | **0.740 ± .055** | **0.489 ± .085** | 0.17 |
| taxonomy_agent (Sonnet, 3 seeds) | 0.656 ± .012 | 0.336 ± .035 | 1.86 |
| TopicGPT *(Pham et al. NAACL 2024, official code)* | 0.606 ± .026 | 0.297 ± .047 | 0.12 |
| embed-cluster-LLM (TnT-LLM family) | 0.488 ± .013 | 0.243 ± .010 | 0.02 |
| single-shot LLM | 0.345 ± .056 | 0.101 ± .041 | 0.02 |
| iterative-proposal LLM | 0.222 ± .128 | 0.043 ± .043 | 0.03 |
| Brady-Islam 2025 (their code) | 0.164 ± .092 | 0.022 ± .002 | <0.01 |
| BERTopic, LDA | ≤ 0.16 | ≤ 0.02 | 0.00 |

- Cheap pair leads strongest LLM baseline (TopicGPT) by **1.22× NMI / 1.65× ARI**.
- Sonnet at 11× the cost does not improve — orchestrator quality is
  not the bottleneck.

---

## Slide 6 — CoT-Pattern results (149 items, 5 failure modes)

| Method | NMI ↑ | Purity ↑ |
|---|---|---|
| **taxonomy_agent (cheap pair, 3 seeds)** | **0.944 ± .081** | **0.975 ± .043** |
| taxonomy_agent (Sonnet, 1 seed) | 0.762 | 0.866 |
| TopicGPT *(Pham et al., 3 seeds)* | 0.001 ± .002 | 0.206 ± .008 |
| embed-cluster-LLM | 0.039 | 0.268 |
| single-shot LLM | 0.143 | 0.336 |
| LR@surface *(supervised oracle, 14 features)* | n/a | 0.730 |
| random floor | n/a | 0.200 |

- taxonomy_agent labels on average **142 / 149** items at **100% accuracy**
  on the labelled subset.
- **TopicGPT collapses**: in 2 of 3 seeds it discovers a single
  "Mathematics" surface topic. No goal-instruction mechanism to direct
  discovery toward the reasoning-pattern axis.

---

## Slide 7 — Cheap pair dominates

- Install default is DeepSeek-v4-Flash in *both* roles (orchestrator
  and judge).
- Sonnet 4.6 orchestrator pair: NMI **0.656** at \$1.86 per 20NG run.
- Cheap pair: NMI **0.740** at \$0.17 per run.
- Same model in both roles, **11× less spend**, better quality.
- Publication and install configuration are the same — no "try the
  expensive version" upgrade path needed.

(Visualisation: scatter of cost vs NMI showing cheap pair top-left of
Sonnet — the Beamer deck has this as a TikZ plot.)

---

## Slide 8 — Demo: Streamlit Inspect tab

Show the figure at `paper/figures/streamlit_05_cot_trace.png`.

- Iteration timeline parses `trace.jsonl` into one row per tool call.
- Don't-fit rates as bars, cost split by role.
- Partial state recoverable after process kill.

CLI install path: `pip install taxonomy-agent`, then `taxonomy demo`.

---

## Slide 9 — Honest weaknesses

- **20NG contamination.** 20 Newsgroups predates current LLM training
  cutoffs; gold structure may be memorised.
- **Single-source CoT-Pattern.** Synthetic corpus from
  Gemini-3.1-Flash-Lite, verified by a Claude Opus 4.8 ensemble (same
  family as the Sonnet evaluator).
- **Single-seed typed-dispatcher ablation.** Contribution rests on
  one seed; reported as preliminary.
- **One Sonnet seed on CoT-Pattern.** "1/30 of the cost" comparison
  hangs on n=1 for the Sonnet side.
- **TnT-LLM not run directly.** No canonical Microsoft code release;
  the embed-cluster-LLM family is represented by our own minimal
  pipeline (no distillation phase — orthogonal scaling step).

---

## Slide 10 — Open questions for you

1. Is the structural-commitments framing strong enough as the headline
   contribution for a demo paper, or should we lead with the
   cheap-model-dominance empirical claim?
2. Drop `iterative_proposal` now that the actual TopicGPT is in the
   table? It looks like a strictly worse version of TopicGPT next to it.
3. Is the Sonnet n=1 on CoT-Pattern a submission blocker, or fixable
   as a footnote + camera-ready follow-up?
4. Streamlit demo URL: deploy as part of submission, or only for
   camera-ready?
5. Any other baselines worth running before 2026-07-10 submission?

---

## Status

- Paper draft: 6 body pages + 2 appendix pages at commit `dae9bd6` on `main`.
- Recent passes: adversarial proofread (80+ confirmed findings landed),
  bib audit against arXiv/ACL anthology (8 author corrections + duplicate
  cleanup), faithful TopicGPT runs (5+3 seeds, \$0.85 spend), framing
  pass for baselines.
- Repo: `github.com/ngqm/taxonomy-agent`. CLI: `pip install
  taxonomy-agent`, then `taxonomy demo`.
- Outstanding pre-submission: deploy demo URL, screencast, decision on
  `iterative_proposal` row.
