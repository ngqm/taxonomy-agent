# TaxonomyAgent

[![CI](https://github.com/ngqm/taxonomy-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ngqm/taxonomy-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

TaxonomyAgent discovers an interpretable taxonomy over an unlabelled text corpus
along an axis you choose, then labels every item against it. An orchestrator LLM
proposes typed edits to a working taxonomy while a cheaper judge LLM classifies
items; both run through OpenRouter and default to DeepSeek-v4-Flash.

You give it a corpus and one sentence describing the axis of interest, for
example "group these prompts by the manipulation tactic each uses", and get back
the discovered categories, a label and rationale for every item, and a
replayable trace.

## Installation

```bash
git clone https://github.com/ngqm/taxonomy-agent
cd taxonomy-agent
pip install -e .
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
```

Requires Python 3.10+.

## Quickstart

```python
from taxonomy_agent import run

result = run(
    items=["first text", "second text"],   # or {id, text} dicts, or a .jsonl/.json/.csv path
    instruction="Group these prompts by the manipulation tactic each uses.",
    output_dir="out/",
)

result.definitions        # {category: definition}
result.to_dataframe()     # id, text, category, rationale, definition
result.save_csv("labels.csv")
result.cost_usd           # OpenRouter spend, in USD
```

Or from the command line:

```bash
taxonomy run corpus.csv -g "Group these by the manipulation tactic each uses." -o out/
taxonomy demo     # one-command run on a bundled DarkBench slice
taxonomy ui       # Streamlit app (or: streamlit run app.py)
```

`RunResult.from_dir("out/")` reloads a finished run offline. Runnable scripts
live in [`examples/`](examples/), and `notebooks/quickstart.ipynb` is a full
walkthrough.

## Key options

`items`, `instruction`, and `output_dir` are required; everything else has a
sensible default. The ones you reach for most:

| Parameter | Default | Description |
| --- | --- | --- |
| `orchestrator_model` / `judge_model` | `deepseek/deepseek-v4-flash` | Any OpenRouter `provider/model` slug. Pairing a stronger orchestrator with the cheap judge helps on hard corpora. |
| `max_iterations` | `10` | Cap on discovery rounds. |
| `converge_below` | `0.10` | Stop once fewer than this fraction of items fit no category. |
| `size_hint` | `"4–10"` | Free-form target taxonomy size; `""` for no target. |
| `finalize` | `"judge"` | How to label the full corpus once discovery converges — see [Scaling](#scaling-to-large-corpora). |
| `concurrency` | `8` | Parallel judge calls. |

Run `help(run)` for the complete parameter reference, or `taxonomy run --help`
for the CLI flags. The result's `iteration_stats()` and `plot_iterations()`
expose per-round diagnostics (`pip install 'taxonomy-agent[viz]'` for the plot).

## Scaling to large corpora

Labelling every item with the judge is one LLM call per item, fine for thousands
but costly for millions. `finalize=` picks how the corpus is labelled once
discovery converges:

- **`"judge"`** (default) — the LLM labels every item. Highest fidelity, O(N) cost.
- **`"embed"`** — re-judge a sample, label the confident majority by embedding
  nearest-centroid, and send only the low-confidence tail to the judge.
- **`"finetune"`** — same, but fine-tune a BERT-family model instead of
  centroids (a little more accurate, heavier).
- **`"none"`** — discovery only: ship the taxonomy plus the items judged during
  discovery, and label the rest later.

```python
result = run(items, instruction, output_dir="out/",
             finalize="embed", coverage=0.85, calibration_size=200)
```

`embed`/`finetune` cut the judge bill from N to the low-confidence tail; on an
embedding-separable corpus (e.g. DarkBench) they reproduce ~98% of the
full-judge labels while judging ~20% of items. Fidelity is corpus-dependent, so
each run reports its own estimate in `result.labeling["val_accuracy"]`. Needs
`pip install 'taxonomy-agent[scale]'`.

## Refining a taxonomy

Steer a finished result in natural language instead of re-running from scratch.
`refine()` interprets feedback into typed edits (`add`, `rename`, `edit`,
`merge`, `split`, `drop`) and re-labels only the items an edit could have moved:

```python
from taxonomy_agent import refine

# Natural language (one cheap judge call to interpret):
better = refine(result, "merge the two flattery categories and split 'harmful' by severity")

# Or exact edits — no LLM, fully deterministic:
better = refine(result, operations=[
    {"op": "rename", "old_name": "sycophancy", "new_name": "flattery"},
    {"op": "merge",  "into": "manipulation", "from": ["sneaking", "brand_bias"]},
])
```

It writes a new run and leaves the original untouched. A rename or merge is a
pure relabel with no judge calls; open-ended feedback warm-starts a short
re-discovery loop.

## Input and output

Inputs: a list of strings, a list of `{id, text}` dicts, or a path to a local
`.jsonl` (one object or string per line), `.json` (an array), or `.csv` (a
`text` column, optional `id`). A `.jsonl` path is read out-of-core (indexed by
byte offset, rows loaded on demand), so corpora larger than memory work.
Identifiers are assigned by position when absent.

Each run writes to its output directory:

- `taxonomy.json` — the taxonomy and per-category counts (compact; no per-item rows)
- `classifications.jsonl` — per-item labels and rationales, streamed row by row
- `trace.jsonl` — every revise, classify, and proposal call
- `taxonomy_state.json` — the working taxonomy after each revision
- `cost.json` / `meta.json` — running spend and run metadata (config, status, final cost)

## Cost

With DeepSeek-v4-Flash in both roles, a 500-item run costs roughly \$0.17 and
takes about ten minutes; smaller corpora cost a few cents. A stronger
orchestrator improves quality on difficult corpora at higher cost while the
judge stays cheap.

## Testing

```bash
python -m pytest tests/
```

The suite stubs the judge, so it runs offline in seconds without an API key.

## Reproducing the benchmarks

```bash
pip install -e ".[eval]"
python -m taxonomy_agent.eval --corpus 20ng \
    --methods taxonomy_agent,bertopic,lda --seeds 42,43,44 \
    --orchestrator deepseek/deepseek-v4-flash \
    --instruction "Identify the topic of each text."
```

Writes `results.json` with purity, NMI, ARI, and cost per method and seed. Pass
`--orchestrator deepseek/deepseek-v4-flash` to match the paper's cheap config
(the CLI otherwise defaults the orchestrator to Claude Sonnet).

## Citation

```bibtex
@misc{nguyen2026taxonomyagent,
  title  = {TaxonomyAgent: An Agent for Iterative Taxonomy Discovery},
  author = {Nguyen, Quang Minh and Ahmed, Uzair and Kim, Taegyoon},
  year   = {2026},
  note   = {Preprint},
  howpublished = {\url{https://github.com/ngqm/taxonomy-agent}}
}
```

## License

MIT — see [LICENSE](LICENSE).
