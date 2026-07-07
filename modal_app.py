"""Serve the TaxonomyAgent Streamlit demo on Modal.

    pip install modal
    modal setup                 # one-time auth (or `modal token set ...`)
    modal serve  modal_app.py   # dev: ephemeral URL, hot reload
    modal deploy modal_app.py   # persistent public URL

A single generously-sized container serves the app; Streamlit multiplexes
browser sessions in-process, so one warm container handles a poster session's
traffic. Reviewers paste their own OpenRouter key in the sidebar, so no key is
baked into the image (see DEPLOY.md for an optional shared-key setup).

The image installs only the light serve dependencies (requirements.txt, no
torch/umap) — the bundled example runs render their corpus map from precomputed
coordinates, so exploring needs no heavy extras.
"""
import shlex
import subprocess

import modal

APP_DIR = "/root/app"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    # Bake the repo (code + bundled example_runs) into the image. Skip large or
    # irrelevant dirs so the image stays small and builds fast.
    .add_local_dir(
        ".",
        remote_path=APP_DIR,
        copy=True,
        ignore=[
            "eval_runs", "taxonomy_runs", ".git", "paper", "slides",
            "scripts", "__pycache__", "*.pyc", "*.egg-info", ".venv",
        ],
    )
)

app = modal.App("taxonomy-agent-demo", image=image)


@app.function(
    cpu=2.0,
    memory=4096,
    # min_containers=1 keeps a warm container (no cold start) but bills credits
    # continuously; default 0 scales to zero — warm it before a live demo.
    min_containers=0,
    scaledown_window=600,
    timeout=3600,
)
@modal.concurrent(max_inputs=100)  # one container serves many browser sessions
@modal.web_server(port=8501, startup_timeout=180)
def serve():
    cmd = (
        f"streamlit run {APP_DIR}/app.py "
        "--server.port 8501 --server.address 0.0.0.0 --server.headless true "
        "--server.enableCORS false --server.enableXsrfProtection false "
        "--browser.gatherUsageStats false"
    )
    # cwd = APP_DIR so Streamlit picks up .streamlit/config.toml (the theme).
    subprocess.Popen(shlex.split(cmd), cwd=APP_DIR)
