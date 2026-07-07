# Deploying the demo on Modal

The Streamlit demo is served on [Modal](https://modal.com) from a single
container. Streamlit multiplexes browser sessions in-process, so one container
handles a poster session's traffic; Modal gives far more CPU/RAM headroom than
the free Streamlit Cloud / HF Spaces tiers, plus scale-to-zero and free credits.

**Live URL:** https://ngqm--taxonomy-agent-demo-serve.modal.run

## One-time setup

```bash
pip install modal
modal setup                 # browser auth, or:
# modal token set --token-id <id> --token-secret <secret>
```

## Deploy / redeploy

```bash
modal deploy modal_app.py   # builds the image, publishes the persistent URL
```

Re-run after any code change. For local iteration with hot reload and an
ephemeral URL, use `modal serve modal_app.py`.

## What the config does (`modal_app.py`)

- **Image:** `requirements.txt` only (no `torch`/`umap`) plus the repo code and
  bundled `example_runs/`. The corpus map and per-category examples for the
  bundled runs render from precomputed JSON, so exploring needs no heavy extras.
- **Package resolution:** the app auto-creates a symlink shim so
  `python -m taxonomy_agent` resolves even though the code lives at `/root/app`
  in the container (the package is `taxonomy_agent`, the dir is not).
- **Resources:** `cpu=2.0, memory=4096`. Bump these in `modal_app.py` if you
  expect heavy concurrent live runs.
- **Scaling:** `min_containers=0` scales to zero (no idle cost). For a live demo,
  either warm it first (open the URL ~30 s before presenting) or set
  `min_containers=1` temporarily to keep a container hot (bills credits
  continuously).

## API keys

Reviewers paste their **own** OpenRouter key in the sidebar — nothing is baked
into the image, and no key means only the bundled examples work (which is the
safe default for a public URL).

To let visitors run without their own key (they will spend **your** OpenRouter
credits — use with care), attach a Modal secret:

```bash
modal secret create openrouter-key OPENROUTER_API_KEY=sk-or-...
```

then add `secrets=[modal.Secret.from_name("openrouter-key")]` to the
`@app.function(...)` decorator and redeploy.

## Costs

- **Modal** covers compute. Scale-to-zero means you pay only while a container
  is up; free monthly credits comfortably cover a demo. A warm container
  (`min_containers=1`) bills continuously, so keep it at 0 outside demo windows.
- **OpenRouter** bills LLM calls separately, on whoever's key is used. A typical
  small run is well under $0.05.

## Safeguards on the public URL

- **No secrets in the image.** `modal_app.py` excludes `.env`, `.modal.toml`,
  `*.pem`, `*.key` from the build, and `agent.py` uses `load_dotenv(override=False)`
  so a stray `.env` can never override a reviewer's key. Verify with a throwaway
  `modal run` that checks `/root/app/.env` is absent.
- **Input + per-run caps.** The image sets `TAXONOMY_DEMO_HOSTED=1`, which makes
  the app reject corpora over **2,000 rows**, cap a run at **8 iterations**
  (pool ≤ 100), and run a lightweight **content filter** on the goal instruction
  (length limit, prompt-injection and inappropriate-content blocklist). Local
  installs are uncapped and unfiltered.
- **BYO-key.** Runs use the reviewer's own OpenRouter key; no shared key unless
  you explicitly attach one (above).
- **Bound the blast radius on Modal:** keep `min_containers=0`, and set an
  account **spending limit** and a **max container count** in the Modal
  dashboard so a burst of traffic can't drain credits.
- **If a key was ever exposed** (e.g. an early build that baked `.env`), rotate
  it in the OpenRouter dashboard — treat it as compromised.

## Notes

- Container storage is ephemeral: a reviewer's run persists for their session,
  not across container restarts. The bundled `example_runs/` are always present
  (baked into the image).
- Manage the deployment at https://modal.com/apps/ngqm/main/deployed/taxonomy-agent-demo
  (`modal app stop taxonomy-agent-demo` to take it down).
