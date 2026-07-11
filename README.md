# taxonomy_agent

Discover a taxonomy over an unlabelled corpus and label every item, along an
axis you choose. An orchestrator LLM proposes typed edits to a working taxonomy;
a cheaper judge LLM labels each item. Both run through OpenRouter, defaulting to
DeepSeek-v4-Flash for both roles.

Give it a corpus and one sentence naming the axis (for example, "group these
prompts by the manipulation tactic each uses"). It returns the categories, a
label and a rationale for every item, and a replayable trace.

Live demo: https://ngqm--taxonomyagent.modal.run

## Install

Requires Python 3.10 or newer.

```bash
git clone https://github.com/ngqm/taxonomy-agent.git
cd taxonomy-agent
pip install -e .                      # registers the `taxonomy` command
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
```

## Run it

Three interfaces share one engine.

### Python

```python
from taxonomy_agent import run

result = run(
    items=["first text", "second text"],   # list of strings, list of
                                            # {id, text} dicts, or a path to a
                                            # .jsonl / .json / .csv file
    instruction="Group these prompts by the manipulation tactic each uses.",
    output_dir="out/",
)

result.definitions          # {category: one-line definition}
result.classifications      # [{id, text, category, rationale}, ...]
result.category_counts      # {category: n_items}
result.cost_usd             # OpenRouter spend, in USD
result.to_dataframe()       # id, text, category, rationale, definition
result.save_csv("labels.csv")
```

`run()` returns a `RunResult`, a `dict` subclass, so `result["artifact"]` and
`result["cost"]` still work.

### Command line

```bash
taxonomy run corpus.csv -g "Group these by the manipulation tactic each uses." -o out/
taxonomy demo                         # one-command run on a bundled DarkBench slice
taxonomy --config my.yaml             # config file; CLI flags override its values
```

### Web

```bash
taxonomy ui                           # or: streamlit run app.py
```

Five tabs: **Start** (how to use it), **Run** (launch, with a live trace),
**History** (past runs), **Inspect** (taxonomy, corpus map, cost, per-item
labels), and **Compare** (two runs side by side). The bundled runs under
`example_runs/` render in Inspect and Compare with no API key.

## Input

Pass a list of strings, a list of `{id, text}` dicts, or a path to a file:

- `.jsonl`: one JSON object (or a bare string) per line
- `.json`: an array of objects or strings
- `.csv`: a `text` column, with an optional `id` column

Ids are assigned by position when absent. Extra keys on a dict item are passed
to the judge as context and copied into the output rows.

## Instruction

One sentence naming the axis to group by. Be specific about the dimension of
variation you care about:

- "Identify the dominant topic of each text."
- "Group these prompts by the manipulation tactic each uses."
- "Identify the failure mode in each transcript (hallucination, refusal, format break)."

Set `category_focus` for an extra constraint on what the categories should
capture, and `size_hint` for a target count (for example `"4-8"`).

## How it works

```
items -> [orchestrator: DeepSeek-v4-Flash]
    sample a batch of K items
    classify_with_judge          the judge labels the batch, or "other"
    don't-fit rate below threshold?
        no  -> propose_novelties + revise_taxonomy, then loop
        yes -> finalize_classify   the judge labels every item
-> taxonomy.json + trace.jsonl
```

The orchestrator drives the loop through six tools: `sample_items`,
`get_taxonomy`, `revise_taxonomy`, `classify_with_judge`,
`propose_novelties_with_judge`, and `finalize_classify`. `revise_taxonomy`
applies typed ops: `add`, `rename`, `edit`, `merge`, `split`, `drop`.

## Output

- `taxonomy.json`: the taxonomy, per-item classifications, and per-category counts.
- `trace.jsonl`: every revise, classify, and novelty call.

```json
{
  "taxonomy": [{"name": "sneaking", "description": "..."}],
  "category_counts": {"sneaking": 87, "other": 4},
  "classifications": [
    {"id": "1", "text": "...", "category": "sneaking", "rationale": "..."}
  ]
}
```

## Configuration

Every setting can be a `--config` file (YAML or JSON) or a CLI flag; flags
override config values.

| Flag | Default | Description |
| --- | --- | --- |
| `--input` | required | Path to a `.jsonl` / `.json` / `.csv` corpus |
| `--instruction` / `--instruction-file` | required | The goal instruction |
| `--output-dir` | required | Where to write outputs |
| `--orchestrator` | `deepseek/deepseek-v4-flash` | Orchestrator model id |
| `--judge` | `deepseek/deepseek-v4-flash` | Judge model id |
| `--max-iters` | `10` | Hard cap on the discovery loop |
| `--min-iters` | `3` | Floor on classify rounds before finalizing |
| `--threshold` | `0.10` | Stop when the don't-fit rate falls below this |
| `--probe-size` | `20` | Items per probe batch |
| `--size-hint` | `4-10` | Target taxonomy size in the prompt (`""` to drop) |
| `--category-focus` | unset | What the categories should describe |
| `--pool-limit` | unset | Cap the corpus size (smoke testing) |
| `--concurrency` | `8` | Parallel judge calls |

## Cost

With DeepSeek-v4-Flash for both roles, a 500-item run costs about $0.17 and
takes roughly 10 minutes; a run of a few dozen items costs a few cents. Cost
scales with the pool size and the iteration count. A stronger orchestrator
(Claude Sonnet, GPT-5, Gemini Pro) raises quality on hard corpora at higher
cost; keep the judge cheap.

`cost.json` is rewritten after each orchestrator step from OpenRouter's native
`usage.cost`, with a static price table as a fallback.

## Repository layout

```
taxonomy_agent/       the package: agent, tools, judge, prompts, cost
  eval/               benchmark harness: runner, metrics, corpora
    baselines/        Baseline subclasses and a name registry
demo/                 Streamlit app: helpers, theme, viz, and tabs/
tests/                pytest suite (offline, stubbed judges)
paper/                LaTeX sources; reproduction scripts under data/
example/              the bundled demo corpus
example_runs/         precomputed runs for Inspect and Compare
app.py, modal_app.py  entrypoints for Streamlit and Modal
```

## Tests

```bash
pip install pytest
python -m pytest tests/
```

154 tests cover input parsing, taxonomy ops, JSON extraction, the six tools, the
judge retry path, cost accounting, and the `RunResult` API. They stub the judge,
so the suite runs offline in a few seconds with no API key.

## Safeguards

- Escape hatch: every classify prompt lets the judge answer `"other"` when
  nothing fits.
- Coercion: an invented label is remapped to `"other"` and preserved in the
  rationale (counted as `n_coerced`).
- Judge errors: transient failures retry with backoff; persistent ones are
  tagged, counted as `n_judge_errors`, and kept out of the don't-fit rate.
- Partial progress: the working taxonomy and the streamed per-item labels
  survive a crash. The Inspect tab loads them when `taxonomy.json` is missing.

## Citation

```bibtex
@inproceedings{nguyen2026taxonomyagent,
  title     = {TaxonomyAgent: Iterative LLM-Driven Taxonomy Discovery},
  author    = {Nguyen, Quang Minh},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing: System Demonstrations},
  year      = {2026}
}
```

A machine-readable `CITATION.cff` is in the repo root.
