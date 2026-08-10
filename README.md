# TaxonomyAgent

[![CI](https://github.com/ngqm/taxonomy-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ngqm/taxonomy-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Live demo](https://img.shields.io/badge/demo-live-brightgreen.svg)](https://ngqm--taxonomyagent.modal.run)

TaxonomyAgent discovers an interpretable taxonomy over an unlabelled text
corpus along an axis you choose, then labels every item against it. An
orchestrator LLM proposes typed edits to a working taxonomy while a cheaper
judge LLM classifies items. Both roles run through OpenRouter and default to
DeepSeek-v4-Flash.

You supply a corpus and one sentence describing the axis of interest, for
example "group these prompts by the manipulation tactic each uses."
TaxonomyAgent returns the discovered categories, a label and rationale for
every item, and a replayable trace of the run.

A hosted demo is available at https://ngqm--taxonomyagent.modal.run.

## Installation

TaxonomyAgent requires Python 3.10 or later.

```bash
git clone https://github.com/ngqm/taxonomy-agent
cd taxonomy-agent
pip install -e .
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
```

## Usage

The same engine is available as a Python library, a command-line tool, and a
web application.

### Python

```python
from taxonomy_agent import run

result = run(
    items=["first text", "second text"],   # or {id, text} dicts, or a
                                            # .jsonl / .json / .csv file path
    instruction="Group these prompts by the manipulation tactic each uses.",
    output_dir="out/",
    orchestrator_model="deepseek/deepseek-v4-flash",  # drives the loop
    judge_model="deepseek/deepseek-v4-flash",         # labels each item
    api_key="sk-or-...",   # or set OPENROUTER_API_KEY in the environment
)

result.definitions            # {category: definition}
result.to_dataframe()         # id, text, category, rationale, definition
result.save_csv("labels.csv")
result.cost_usd               # OpenRouter spend, in USD

result.iteration_stats()      # per-iteration DataFrame: categories, unmatched rate, ...
result.plot_iterations()      # matplotlib figure of those statistics over the run
```

`iteration_stats()` reads the run's `trace.jsonl` and returns one row per loop
event (step, kind, category count, unmatched rate, proposed count, judge
errors). `plot_iterations()` renders the taxonomy size and unmatched rate over
the iterations and returns a matplotlib `Figure` (`pip install
'taxonomy-agent[viz]'` for matplotlib; pass `save_path=` to write it to disk).

`orchestrator_model` and `judge_model` are independent and each accept any
OpenRouter model slug (`provider/model`). Both default to
`deepseek/deepseek-v4-flash`, so you can omit them to run both roles on that
inexpensive model; a common alternative pairs a stronger orchestrator with the
cheap judge, for example `orchestrator_model="anthropic/claude-sonnet-4.6"`. To
use a different OpenAI-compatible endpoint, also pass `base_url=`.

`RunResult.from_dir("out/")` reloads a completed run offline. See
`notebooks/quickstart.ipynb` for a runnable walkthrough.

#### Parameters

`items`, `instruction`, and `output_dir` are required; everything else is an
optional keyword argument with a sensible default:

| Parameter | Default | Description |
| --- | --- | --- |
| `orchestrator_model` | `deepseek/deepseek-v4-flash` | LLM that drives the discovery loop (proposes and revises categories). |
| `judge_model` | `deepseek/deepseek-v4-flash` | LLM that labels each item against the taxonomy. |
| `max_iterations` | `10` | Hard cap on the number of discovery rounds. |
| `min_iterations` | `3` | Minimum judge rounds before the run may finalize; guards against stopping on a lucky early probe. Must be `<= max_iterations`. |
| `converge_below` | `0.10` | Early-stop threshold: finish once the fraction of items that fit no category falls below this (`0.10` = 10%). |
| `probe_size` | `20` | Number of items sampled per discovery probe. |
| `size_hint` | `"4–10"` | Free-form target taxonomy size given to the orchestrator; `None` or `""` means no target. |
| `category_focus` | `None` | Optional sentence describing what the categories should capture (e.g. "the reasoning strategy each chain of thought uses"). |
| `concurrency` | `8` | Number of parallel judge calls. |
| `pool_limit` | `None` | Cap the number of items used (handy for smoke tests); `None` uses all of them. |
| `seed` | `42` | Seeds probe sampling for reproducibility; vary it for independent replicates. |
| `temperature` | `0.2` | Orchestrator sampling temperature. |
| `orchestrator_max_tokens` | `None` | Cap on orchestrator output tokens per step; `None` uses the model's own default. |
| `judge_max_tokens` | `300` | Max tokens per judge classification reply (label + rationale). Lower to trim cost on the O(N) pass; raise if rationales truncate. |
| `reasoning_effort` | `None` | Reasoning effort for a reasoning-capable orchestrator (`low`/`medium`/`high`), forwarded as OpenRouter `reasoning.effort`. The judge stays a cheap non-reasoning labeller. |
| `recursion_limit` | `80` | LangGraph cap on agent super-steps. |
| `finalize` | `"judge"` | How to label the full corpus: `"judge"` (LLM per item), `"embed"` (re-judge a sample, then embedding nearest-centroid), or `"finetune"` (re-judge, then fine-tune a BERT-family model). See below. |
| `coverage` | `0.85` | With `finalize="embed"/"finetune"`, the fraction of items to accept from the classifier; the rest go to the judge. |
| `calibration_size` | `200` | With `finalize="embed"/"finetune"`, items to re-judge against the final taxonomy for training labels (`0` = probes only); each is an extra judge call. |
| `embed_model` | `all-MiniLM-L6-v2` | sentence-transformers model for `finalize="embed"`. |
| `api_key` | `OPENROUTER_API_KEY` | OpenRouter key; read from the environment if omitted. |
| `base_url` | OpenRouter | OpenAI-compatible endpoint to call. |

The `taxonomy run` CLI exposes the most-used knobs as flags (`--max-iters`,
`--min-iters`, `--threshold`, `--probe-size`, `--concurrency`, `--seed`,
`--orchestrator`, `--judge`, `--size`, `--judge-max-tokens`,
`--orchestrator-max-tokens`, `--reasoning-effort`; see `taxonomy run --help`).
Run `help(run)` in Python for the full docstring.

#### Scaling to large corpora (`finalize=`)

Labeling every item with the judge is one LLM call per item — fine for
thousands, costly for millions. The `finalize` argument picks how the corpus is
labeled once discovery converges — three options:

1. **`finalize="judge"`** (default) — the LLM judge labels every item. O(N)
   calls; highest fidelity, highest cost.
2. **`finalize="embed"`** — re-judge a sample against the final taxonomy, then
   label the confident majority by **embedding nearest-centroid** and send only
   the low-confidence tail to the judge.
3. **`finalize="finetune"`** — same, but **fine-tune a BERT-family model** on the
   re-judged sample instead of using centroids (a little more accurate, heavier).

```python
result = run(items, instruction, output_dir="out/",
             finalize="finetune",           # or "embed" / "judge"
             calibration_size=200,   # items to re-judge for training (default 200)
             coverage=0.85)          # keep the top 85% by confidence, judge the rest
```

For `embed`/`finetune`, the classifier trains on the discovery probes plus
`calibration_size` fresh items **re-judged against the final taxonomy**
(clean labels — this is the main lever on fidelity), then labels every item; the
judge bill drops from N to the size of the low-confidence tail. Fidelity is
corpus-dependent and degrades gracefully — on an embedding-separable corpus
(e.g. DarkBench manipulation tactics) the classifier reproduces ~98% of the
full-judge labels while judging only ~20% of items (a 5× cut); on harder corpora
the gate routes more to the judge to hold accuracy. `finetune` edges out `embed`
but usually by little (DarkBench 95.0% vs 93.5% at a 30% split), so `embed` is
often the better cost/quality trade. Lower `coverage` for higher
fidelity, raise it (up to `1.0`, a pure `$0` labeling pass) for lower cost.
`embed`/`finetune` need the extra: `pip install 'taxonomy-agent[scale]'`.

Because fidelity is corpus-dependent, each `embed`/`finetune` run **measures its
own**: a slice of the re-judged calibration is held out, and the classifier's
agreement with the judge on it is reported as `result.labeling["val_accuracy"]`
(and logged) — this run's fidelity estimate on your data. A low value warns that
the cheap labels are noisy for your corpus, so you can raise the calibration
size, lower `coverage`, or fall back to `finalize="judge"`.

#### Refining a taxonomy with feedback

Not happy with the result? Steer it in natural language instead of re-running
from scratch. `refine()` interprets your feedback into typed edits (`add`,
`rename`, `edit`, `merge`, `split`, `drop`), applies them, and re-labels only
the items an edit could have moved:

```python
from taxonomy_agent import refine

# Natural language: interpreted into edits by a cheap judge call.
better = refine(result, "merge the two flattery categories and split 'harmful' by severity")

# Or apply exact edits directly — no LLM, fully deterministic.
better = refine(result, operations=[
    {"op": "rename", "old_name": "sycophancy", "new_name": "flattery"},
    {"op": "merge",  "into": "manipulation", "from": ["sneaking", "brand_bias"]},
])

better.definitions        # the revised taxonomy
better.to_dataframe()     # re-labelled items
better["refine"]          # the feedback, applied ops, and how many items were re-labelled
```

`refine()` accepts a `RunResult` or a run directory and writes a **new** run
(the original is untouched). It's cheap because a rename or merge is a pure
relabel with **no** judge calls; an `add` re-judges only the `other` bucket; a
`drop` / `split` / `edit` re-judges just that category's items. Control this with
`reclassify=`: `"affected"` (default), `"all"` (relabel everything), or `"none"`
(deterministic relabels only). Open-ended feedback ("too fine-grained") warm-
starts a short re-discovery loop from the current taxonomy.

### Command line

```bash
taxonomy run corpus.csv -g "Group these by the manipulation tactic each uses." -o out/
taxonomy demo    # one-command run on a bundled DarkBench slice
```

### Web

```bash
taxonomy ui      # or: streamlit run app.py
```

## Input formats

The library and the CLI accept a list of strings, a list of `{id, text}`
dictionaries, or a path to a local file (URLs are not fetched):

- `.jsonl` — one JSON object, or a bare string, per line
- `.json` — an array of objects or strings
- `.csv` — a `text` column, with an optional `id` column

A `.jsonl` path is read **out-of-core**: the file is indexed by byte offset and
rows are loaded on demand (sampled by seek, streamed for the final labeling
pass), so a corpus far larger than memory can be processed. The other inputs
are loaded into memory.

Identifiers are assigned by position when absent.

## Output

Each run writes to its output directory:

- `taxonomy.json` — the discovered taxonomy and per-category counts (a compact
  summary; the per-item rows are not embedded, so it stays small on a
  million-item run)
- `classifications.jsonl` — the per-item labels and rationales, streamed row by
  row; the source of truth `RunResult` reads (and `save_csv` streams from)
- `trace.jsonl` — every revise, classify, and novelty-proposal call
- `taxonomy_state.json` — the working taxonomy, rewritten after each revision
- `cost.json` — running spend, from OpenRouter's native usage cost

## Cost

With DeepSeek-v4-Flash in both roles, a 500-item run costs roughly \$0.17 and
takes about ten minutes; smaller corpora cost a few cents. A stronger
orchestrator such as Claude Sonnet, GPT-5, or Gemini Pro improves quality on
difficult corpora at higher cost, while the judge can remain inexpensive.

## Testing

```bash
python -m pytest tests/
```

The suite stubs the judge, so it runs offline in a few seconds without an API
key.

## Reproducing the benchmarks

The evaluation harness reproduces the benchmark numbers:

```bash
pip install -e ".[eval]"
python -m taxonomy_agent.eval --corpus 20ng \
    --methods taxonomy_agent,bertopic,lda --seeds 42,43,44 \
    --instruction "Identify the topic of each text."
```

It writes `results.json` with purity, NMI, ARI, and cost per method and seed.

## Citation

```bibtex
@misc{nguyen2026taxonomyagent,
  title  = {TaxonomyAgent: An Agent for Iterative Taxonomy Discovery},
  author = {Nguyen, Quang Minh and Ahmed, Uzair and Kim, Taegyoon},
  year   = {2026},
  note   = {Preprint},
  howpublished = {\url{https://ngqm--taxonomyagent.modal.run}}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
