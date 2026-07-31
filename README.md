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
```

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
| `recursion_limit` | `80` | LangGraph cap on agent super-steps. |
| `api_key` | `OPENROUTER_API_KEY` | OpenRouter key; read from the environment if omitted. |
| `base_url` | OpenRouter | OpenAI-compatible endpoint to call. |

The `taxonomy run` CLI exposes the most-used knobs as flags (`--max-iters`,
`--min-iters`, `--threshold`, `--probe-size`, `--concurrency`, `--seed`,
`--orchestrator`, `--judge`, `--size`; see `taxonomy run --help`). Run
`help(run)` in Python for the full docstring.

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

Identifiers are assigned by position when absent.

## Output

Each run writes to its output directory:

- `taxonomy.json` — the taxonomy, the per-item classifications, and the
  per-category counts
- `classifications.jsonl` — per-item labels, streamed row by row
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
