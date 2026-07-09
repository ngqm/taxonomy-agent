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
- `app.py` (~85 lines): thin entry point. Page bootstrap (`set_page_config`,
  the masthead eyebrow / title / subtitle, `st.session_state` init — these MUST
  stay here; module-level code in demo/ runs once per process, not per Streamlit
  rerun), then `settings = render_sidebar()`, then `inject_theme(ss["theme"])`
  (the Day/Night CSS, injected *after* the sidebar's Appearance toggle has set
  the theme so the switch takes effect the same rerun), then `st.tabs(...)`
  dispatching to the tab renderers. (The "New?" onboarding banner now lives in
  the Run tab, not here.)
- `demo/`: the app, as a package (names re-exported via `from demo import *`).
  - `sidebar.py`: `render_sidebar()` renders the config sidebar and returns a
    `Settings` NamedTuple.
  - `tabs/{run,history,inspect,compare}.py`: one `render(...)` per tab. `run`
    and `inspect` take the `Settings`; `history`/`compare` take nothing.
  - `helpers.py` (paths, run discovery/scanning, cost estimate, trace/cost
    renderers, presets, model lists), `guards.py` (hosted caps + content
    filter), `viz.py` (per-category colour *mapper*, embeddings/UMAP, run-card
    renderer).
  - `theme.py`: the visual design system ("The Journal" editorial look) —
    Day/Night palette tokens, the shared category-colour palette, the font +
    widget CSS (`inject_theme`), and the bespoke HTML component builders
    (specimen cards, stat ledger, distribution bars, map legend, filter-chip
    tinting). Purely presentational; imported by `viz.py`, `helpers.py`, and the
    tabs.
- `modal_app.py`: the Modal deployment (serves `app.py` in one container).
- `scripts/precompute_example_viz.py`: regenerates the bundled runs' map/example
  JSON.
- `example_runs/`: three finished runs shipped in the repo so the results tabs
  have content out of the box (two DarkBench, one 20 Newsgroups).
- `.streamlit/config.toml`: the static base theme Streamlit paints first (the
  warm-paper "Day" palette + vermilion primary). `demo/theme.py` layers the full
  "The Journal" editorial look and the runtime Day/Night switch on top.

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
- **Package layout.** The library lives in the `taxonomy_agent/` subdirectory
  (a normal package). app.py runs the agent as a subprocess with
  `cwd=repo-root`, so `python -m taxonomy_agent` resolves regardless of the
  clone directory name. (An earlier flat layout needed a symlink shim; that is
  gone.)
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
   smoke test (load, open Inspect, one keyless run, and a *second* session to
   catch session-state init bugs) in CI would catch regressions. This session
   found several only by screenshotting.
6. **Run-tab loop robustness.** `demo/tabs/run.py` streams a detached
   subprocess by tailing its log/trace files in a `while proc.poll()` loop.
   It works, but it is the most fragile piece (state, streaming, Stop button).
   Any change here should be checked with a real run in a *second* session,
   not just a first-load render.
6. **Cold start + keep-warm.** Scale-to-zero saves money but the first visitor
   waits. A keep-warm during demo windows (`min_containers=1`) or a scheduled
   ping is a cheap fix.
7. **Spend safety.** Consider a per-session run cap and a visible spend meter for
   the shared key, on top of the OpenRouter credit limit.
8. **UI polish.** Accessibility/contrast audit; History pagination/search for
   many runs; optional side-by-side corpus maps in Compare; mobile layout.
