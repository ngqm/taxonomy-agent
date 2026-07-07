# Demo handoff

This is a working guide for inspecting and improving the **TaxonomyAgent demo**
(the Streamlit app and its Modal deployment). For the research/library details
see `README.md`; for deploy mechanics see `DEPLOY.md`.

- **Live demo:** https://ngqm--taxonomy-agent.modal.run
- **Repo:** https://github.com/ngqm/taxonomy-agent (branch `main`)
- **What it is:** point it at a corpus and a one-line goal; an orchestrator LLM
  discovers a taxonomy and a judge LLM labels every item. Default models are
  DeepSeek-v4-Flash for both roles, via OpenRouter.

## Run it locally in two minutes

```bash
git clone https://github.com/ngqm/taxonomy-agent.git
cd taxonomy-agent
pip install -e .            # or: pip install -r requirements.txt
streamlit run app.py       # opens the four-tab UI
```

The **Inspect** and **Compare** tabs work with zero setup (they read the bundled
`example_runs/`). To launch a real run you need an OpenRouter key: put it in
`.env` as `OPENROUTER_API_KEY=...`, or paste it in the sidebar.

`pytest` runs the backend unit tests (fast, offline, no key). Note: the app UI
itself has no automated tests yet (see backlog).

## Codebase map

**Backend (the agent, unchanged by the demo work):**
- `agent.py`: `run()`, the orchestrator loop entry point.
- `__main__.py`: CLI (`python -m taxonomy_agent` / the `taxonomy` script).
- `tools.py`, `judge.py`, `prompts.py`, `cost.py`: the six tools, judge caller,
  prompt templates, cost tracking.
- `tests/`: backend unit tests (ops, cost, metrics, judge, parsing, tools).

**Demo (where most inspection/improvement will happen):**
- `app.py` (~1700 lines): the Streamlit UI. Section banners mark the parts:
  `# ── Sidebar`, `# ── Run tab`, `# ── Runs tab` (History), `# ── Inspect tab`,
  `# ── Compare tab`. Helpers up top: `_pkg_root_for_subprocess`,
  `_instruction_block_reason`, `_category_colors`, `_corpus_umap`,
  `_representative_examples`, `_render_run_card`, `_render_progress`.
- `modal_app.py`: the Modal deployment (serves `app.py` in one container).
- `scripts/precompute_example_viz.py`: regenerates the bundled runs' map/example
  JSON.
- `example_runs/`: three finished runs shipped in the repo so the results tabs
  have content out of the box (two DarkBench, one 20 Newsgroups).
- `.streamlit/config.toml`: the theme (pine-green).

## How the non-obvious demo pieces work

- **Explore-first.** `app.py` preloads `example_runs/darkbench_manipulation` into
  the Inspect tab, and a banner points keyless visitors there. This is the main
  demo experience; live runs are secondary.
- **Light deploy (no torch at serve).** The corpus map and per-category examples
  for bundled runs render from precomputed `map_coords.json` +
  `representative.json` (see the `scripts/` file). The app only imports
  `sentence-transformers`/`umap-learn` when computing a map for a *user's own*
  run, which the hosted image does not install. Keep it this way to stay on a
  small image.
- **Import shim.** The package is `taxonomy_agent` (underscore) but the repo
  clones as `taxonomy-agent` (hyphen), so `python -m taxonomy_agent` would fail
  on a hosted checkout. `_pkg_root_for_subprocess()` creates a symlink so it
  resolves. Do not remove it without testing on a hyphenated checkout.
- **Keys.** The sidebar key is optional. If blank, runs fall back to
  `OPENROUTER_API_KEY` from the environment; on Modal that comes from the
  `openrouter-demo-key` secret (never in the repo/image/UI). See DEPLOY.md.
- **Hosted guards** (active only when `TAXONOMY_DEMO_HOSTED=1`, set by the Modal
  image): reject corpora over 2,000 rows, cap runs at 8 iterations / pool 100,
  and content-filter the instruction (length, prompt-injection, blocklist). Local
  installs are unrestricted.

## Deploy (summary; full details in DEPLOY.md)

```bash
modal app stop taxonomy-agent-demo -y   # clear the warm container (avoids stale code)
modal deploy modal_app.py               # rebuild + publish
```

Redeploy after any `app.py`/`modal_app.py`/`example_runs` change. Warm the URL
once before demoing (cold start is ~20–30 s from scale-to-zero).

## Access your teammate will need

- **GitHub** write access to `ngqm/taxonomy-agent`.
- **Modal** access to the `ngqm` workspace to redeploy (or their own workspace;
  the app name/label determine the URL).
- **OpenRouter**: the shared demo key. It was handled over chat once, so rotate
  it and set a per-key **credit limit** in the OpenRouter dashboard; store the
  new value only in the Modal secret `openrouter-demo-key`.

## Known limitations / improvement backlog

Roughly prioritized; each is a real place to improve the demo.

1. **Live-run robustness on Modal.** Runs are a detached subprocess writing to
   the container's ephemeral disk, so History does not persist across restarts
   and a container recycle can kill an in-flight run. Options: mount a Modal
   Volume for `taxonomy_runs/`, or run the agent as a Modal function per run
   (isolated, autoscaling) and stream progress back.
2. **Concurrency.** One container serves all sessions; several simultaneous runs
   contend for its CPU. The per-run-function approach in (1) fixes this.
3. **Corpus map for user runs on the hosted demo.** Only bundled runs have a
   precomputed map; a reviewer's own run gets none (no torch on the image).
   Could compute it in a Modal function on demand.
4. **Content filter is heuristic.** `_instruction_block_reason` is a regex
   blocklist (length + injection + slurs). An LLM moderation call would be more
   robust; the row/iteration caps are fixed constants worth revisiting.
5. **No UI tests.** Backend has unit tests; the app has none. A small Playwright
   smoke test (load, open Inspect, one keyless run) in CI would catch
   regressions. This session found several only by screenshotting.
6. **Cold start + keep-warm.** Scale-to-zero saves money but the first visitor
   waits. A keep-warm during demo windows (`min_containers=1`) or a scheduled
   ping is a cheap fix.
7. **Spend safety.** Consider a per-session run cap and a visible spend meter for
   the shared key, on top of the OpenRouter credit limit.
8. **UI polish.** Accessibility/contrast audit; History pagination/search for
   many runs; optional side-by-side corpus maps in Compare; mobile layout.
