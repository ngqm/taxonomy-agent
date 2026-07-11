"""OpenRouter client for the judge LLM."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import requests


class JudgeAuthError(RuntimeError):
    """Raised on 401/403 from the judge endpoint — auth bugs, not transient."""


class Judge:
    """A stateless OpenRouter judge bound to one model.

    ``call`` labels a single prompt; ``parallel`` fans a batch over a thread
    pool, preserving order. HTTP failures are discriminated: 401/403 raise
    ``JudgeAuthError``; 429 backs off (1s, 2s, 4s) up to 3 retries; 5xx and
    network errors retry once; a malformed body returns ``None``.

    ``usage_sink``, if set, is called with each successful response's ``usage``
    dict (augmented with ``http_status``) from worker threads, so it must be
    thread-safe (see :class:`cost.CostTracker`).
    """

    def __init__(self, api_key: str, model: str,
                 base_url: str = "https://openrouter.ai/api/v1",
                 usage_sink: Callable[[dict], None] | None = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.usage_sink = usage_sink

    def _post(self, prompt: str, max_tokens: int, temperature: float):
        return requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                # Ask OpenRouter to include the actual charge under usage.cost;
                # cost.CostTracker prefers it over the static price table.
                "usage": {"include": True},
            }),
            timeout=90,
        )

    def _handle_ok(self, resp) -> str | None:
        try:
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            print(f"[judge malformed body] {e}")
            return None
        if self.usage_sink is not None:
            try:
                usage = dict(body.get("usage") or {})
                usage["http_status"] = resp.status_code
                self.usage_sink(usage)
            except Exception as e:  # never let accounting break a call
                print(f"[judge usage_sink error] {e}")
        return content

    def call(self, prompt: str, max_tokens: int = 400,
             temperature: float = 0.0) -> str | None:
        """Label a single prompt. Returns the reply text, or None on failure."""
        rate_backoff = [1.0, 2.0, 4.0]
        rate_attempt = 0
        net_retry_used = False
        server_retry_used = False
        while True:
            try:
                resp = self._post(prompt, max_tokens, temperature)
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status in (401, 403):
                    raise JudgeAuthError(
                        f"judge auth failed (HTTP {status}) — check OPENROUTER_API_KEY"
                    ) from e
                if status == 429:
                    if rate_attempt >= len(rate_backoff):
                        print(f"[judge rate-limited after {rate_attempt} retries] {e}")
                        return None
                    time.sleep(rate_backoff[rate_attempt])
                    rate_attempt += 1
                    continue
                if status is not None and 500 <= status < 600:
                    if server_retry_used:
                        print(f"[judge 5xx after retry] {e}")
                        return None
                    server_retry_used = True
                    time.sleep(1.0)
                    continue
                print(f"[judge HTTP error] {e}")
                return None
            except requests.exceptions.RequestException as e:
                if net_retry_used:
                    print(f"[judge network error after retry] {e}")
                    return None
                net_retry_used = True
                time.sleep(1.0)
                continue
            except Exception as e:
                # Anything that slipped past the request layer (e.g. mocks
                # raising bare RuntimeError) gets one retry, then None.
                if net_retry_used:
                    print(f"[judge error after retry] {e}")
                    return None
                net_retry_used = True
                time.sleep(1.0)
                continue
            return self._handle_ok(resp)

    def parallel(self, prompts: list[str], concurrency: int = 8,
                 max_tokens: int = 400,
                 on_reply: Callable[[int, str | None], None] | None = None,
                 ) -> list[str | None]:
        """Fan ``prompts`` out over a thread pool, preserving order.

        If ``on_reply`` is set, it fires as ``(index, reply)`` the moment each
        future completes, so ``finalize_classify`` can stream rows to disk and a
        crash mid-run does not lose every label."""
        out: list[str | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(self.call, p, max_tokens): i
                    for i, p in enumerate(prompts)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    reply = fut.result()
                except Exception as e:
                    print(f"[judge parallel error] {e}")
                    reply = None
                out[i] = reply
                if on_reply is not None:
                    try:
                        on_reply(i, reply)
                    except Exception as e:
                        print(f"[judge on_reply error] {e}")
        return out
