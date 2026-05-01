"""Stateless OpenRouter call helpers for the judge LLM."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def make_judge_caller(api_key: str, model: str,
                      base_url: str = "https://openrouter.ai/api/v1"):
    """Return (single_call, parallel_call) closures bound to the given judge model.

    `_call` retries once on any failure (network, HTTP, or malformed body) before
    returning None. Callers must distinguish None from a successful "other"
    classification — see tools.classify_with_judge for the bookkeeping."""

    def _call(prompt: str, max_tokens: int = 400, temperature: float = 0.0) -> str | None:
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps({
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }),
                    timeout=90,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                if attempt == 0:
                    time.sleep(1.0)
        print(f"[judge error after retry] {last_err}")
        return None

    def _parallel(prompts: list[str], concurrency: int = 8,
                  max_tokens: int = 400) -> list[str | None]:
        out: list[str | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(_call, p, max_tokens): i for i, p in enumerate(prompts)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    out[i] = fut.result()
                except Exception as e:
                    print(f"[judge parallel error] {e}")
        return out

    return _call, _parallel
