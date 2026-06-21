# paper/

LaTeX source for the EMNLP 2026 System Demonstration paper for
`taxonomy_agent`.

## Build

The skeleton targets the ACL/EMNLP shared style files. Before the first
build, download `acl.sty` and `acl_natbib.bst` from
<https://github.com/acl-org/acl-style-files> and drop them next to
`main.tex`, then uncomment the `\usepackage[demo]{acl}` line at the top
of `main.tex`. Without the style files the document still compiles via
the fallback `article` class.

```
latexmk -pdf main.tex
```

or, manually:

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Layout

- `main.tex` — document root, `\input`s each section.
- `sections/abstract.tex` — abstract (stub, awaits headline number).
- `sections/introduction.tex` — real draft (~0.7 page).
- `sections/system.tex` — architecture, six tools, revise DSL,
  convergence; stub + outline.
- `sections/implementation.tex` — LangGraph, judge fan-out,
  OpenRouter, cost, partial progress, UI; stub + outline.
- `sections/evaluation.tex` — 20 Newsgroups setup, baselines, metrics,
  convergence-validation scatter; stub + outline, no numbers yet.
- `sections/related.tex` — full ~1 page of prose with positioning.
- `sections/limitations.tex` — real para on self-reported convergence,
  cost, English-only single-seed.
- `sections/conclusion.tex` — stub + outline.
- `references.bib` — BibTeX entries.
- `figures/` — figure assets (currently only `.gitkeep`).

## What is still `\todo`

- Headline results in the abstract and conclusion.
- Section 3 (System): tool I/O signatures, revise-op table,
  convergence threshold formula.
- Section 4 (Implementation): codebase map figure.
- Section 5 (Evaluation): all numbers, main results table, convergence
  scatter plot, ablations.
- Architecture diagram (`figures/architecture.pdf`).
- BibTeX entries flagged `\todo` (TaxoAdapt, LLM-guided clustering,
  EMNLP 2024 annotation survey, Co-DETECT) need confirmed author lists
  and bibkeys before camera-ready.

`\todo` markers render visibly via `todonotes`; switch to
`\usepackage[disable]{todonotes}` for a clean PDF.
