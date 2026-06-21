"""make_judge_caller — retry behaviour with mocked OpenRouter calls."""
from __future__ import annotations

import json as _json
from unittest.mock import MagicMock, patch

import pytest
import requests

from taxonomy_agent.judge import JudgeAuthError, make_judge_caller


def _http_err_resp(status: int) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    err = requests.exceptions.HTTPError(f"HTTP {status}")
    err.response = r
    r.raise_for_status.side_effect = err
    return r


def _ok_resp(content: str = "ok") -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.raise_for_status = lambda: None
    r.json.return_value = {"choices": [{"message": {"content": content}}]}
    return r


def _err_resp() -> MagicMock:
    r = MagicMock()
    r.raise_for_status.side_effect = RuntimeError("HTTP 500")
    return r


def test_call_returns_content_on_success():
    call, _ = make_judge_caller("k", "model")
    with patch("taxonomy_agent.judge.requests.post", return_value=_ok_resp("hi")):
        assert call("prompt") == "hi"


def test_call_retries_once_then_returns_none():
    call, _ = make_judge_caller("k", "model")
    posts: list[int] = []

    def fake_post(*a, **k):
        posts.append(1)
        raise RuntimeError("network")

    with patch("taxonomy_agent.judge.requests.post", side_effect=fake_post), \
         patch("taxonomy_agent.judge.time.sleep") as fake_sleep:
        out = call("prompt")
    assert out is None
    assert len(posts) == 2  # original + 1 retry
    fake_sleep.assert_called_once_with(1.0)


def test_call_retry_succeeds_after_initial_failure():
    call, _ = make_judge_caller("k", "model")
    responses = [_err_resp(), _ok_resp("recovered")]

    def fake_post(*a, **k):
        return responses.pop(0)

    with patch("taxonomy_agent.judge.requests.post", side_effect=fake_post), \
         patch("taxonomy_agent.judge.time.sleep"):
        assert call("prompt") == "recovered"


def test_parallel_preserves_order():
    _, parallel = make_judge_caller("k", "model")

    def fake_post(*a, data=None, **kw):
        body = _json.loads(data)
        prompt = body["messages"][0]["content"]
        return _ok_resp(content=f"reply_to_{prompt}")

    with patch("taxonomy_agent.judge.requests.post", side_effect=fake_post):
        out = parallel(["p1", "p2", "p3"], concurrency=2)
    assert out == ["reply_to_p1", "reply_to_p2", "reply_to_p3"]


def test_parallel_isolates_per_prompt_failures():
    """A failure on one prompt must not poison the rest of the batch."""
    _, parallel = make_judge_caller("k", "model")

    def fake_post(*a, data=None, **kw):
        body = _json.loads(data)
        if "fail" in body["messages"][0]["content"]:
            raise RuntimeError("network")
        return _ok_resp(content="ok")

    with patch("taxonomy_agent.judge.requests.post", side_effect=fake_post), \
         patch("taxonomy_agent.judge.time.sleep"):
        out = parallel(["good1", "fail", "good2"], concurrency=2)
    assert out == ["ok", None, "ok"]


def _ok_resp_with_usage(content: str = "ok",
                         prompt_tokens: int = 10,
                         completion_tokens: int = 5,
                         cost: float | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.raise_for_status = lambda: None
    usage = {"prompt_tokens": prompt_tokens,
             "completion_tokens": completion_tokens,
             "total_tokens": prompt_tokens + completion_tokens}
    if cost is not None:
        usage["cost"] = cost
    r.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": usage,
    }
    return r


def test_usage_sink_called_per_successful_call():
    captured: list[dict] = []
    call, _ = make_judge_caller("k", "model", usage_sink=captured.append)
    with patch("taxonomy_agent.judge.requests.post",
               return_value=_ok_resp_with_usage(prompt_tokens=42, completion_tokens=7)):
        call("prompt")
    assert captured == [{"prompt_tokens": 42, "completion_tokens": 7,
                          "total_tokens": 49, "http_status": 200}]


def test_usage_sink_not_called_on_failure():
    captured: list[dict] = []
    call, _ = make_judge_caller("k", "model", usage_sink=captured.append)

    def fake_post(*a, **k):
        raise RuntimeError("network")

    with patch("taxonomy_agent.judge.requests.post", side_effect=fake_post), \
         patch("taxonomy_agent.judge.time.sleep"):
        out = call("prompt")
    assert out is None
    assert captured == []


def test_request_body_asks_openrouter_for_native_cost():
    """Every call must include `usage: {include: true}` in the request body —
    that's how OpenRouter knows to return usage.cost."""
    call, _ = make_judge_caller("k", "model")
    captured: dict = {}

    def fake_post(*a, data=None, **k):
        captured["body"] = _json.loads(data)
        return _ok_resp("ok")

    with patch("taxonomy_agent.judge.requests.post", side_effect=fake_post):
        call("prompt")

    assert captured["body"]["usage"] == {"include": True}


def test_usage_sink_forwards_native_cost():
    captured: list[dict] = []
    call, _ = make_judge_caller("k", "model", usage_sink=captured.append)
    with patch("taxonomy_agent.judge.requests.post",
               return_value=_ok_resp_with_usage(prompt_tokens=100,
                                                  completion_tokens=20,
                                                  cost=0.0042)):
        call("prompt")
    assert captured == [{"prompt_tokens": 100, "completion_tokens": 20,
                          "total_tokens": 120, "cost": 0.0042,
                          "http_status": 200}]


def test_usage_sink_error_does_not_break_call():
    """A buggy sink must not poison the judge result — accounting is best-effort."""
    def boom(usage):
        raise RuntimeError("sink exploded")

    call, _ = make_judge_caller("k", "model", usage_sink=boom)
    with patch("taxonomy_agent.judge.requests.post",
               return_value=_ok_resp_with_usage()):
        assert call("prompt") == "ok"


def test_parallel_on_reply_fires_per_completion():
    """on_reply must be called once per future result, with (index, reply)."""
    _, parallel = make_judge_caller("k", "model")

    def fake_post(*a, data=None, **kw):
        body = _json.loads(data)
        return _ok_resp(content=f"reply_to_{body['messages'][0]['content']}")

    received: list[tuple[int, str | None]] = []

    def on_reply(i, rep):
        received.append((i, rep))

    with patch("taxonomy_agent.judge.requests.post", side_effect=fake_post):
        out = parallel(["a", "b", "c"], concurrency=2, on_reply=on_reply)

    assert out == ["reply_to_a", "reply_to_b", "reply_to_c"]
    # All three indices appear (order across threads isn't guaranteed).
    assert sorted(received) == [(0, "reply_to_a"), (1, "reply_to_b"), (2, "reply_to_c")]


def test_call_raises_on_401():
    call, _ = make_judge_caller("k", "model")
    with patch("taxonomy_agent.judge.requests.post",
               return_value=_http_err_resp(401)):
        with pytest.raises(JudgeAuthError):
            call("prompt")


def test_call_raises_on_403():
    call, _ = make_judge_caller("k", "model")
    with patch("taxonomy_agent.judge.requests.post",
               return_value=_http_err_resp(403)):
        with pytest.raises(JudgeAuthError):
            call("prompt")


def test_call_retries_with_backoff_on_429():
    call, _ = make_judge_caller("k", "model")
    posts: list[int] = []

    def fake_post(*a, **k):
        posts.append(1)
        return _http_err_resp(429)

    with patch("taxonomy_agent.judge.requests.post", side_effect=fake_post), \
         patch("taxonomy_agent.judge.time.sleep") as fake_sleep:
        out = call("prompt")

    assert out is None
    # original + 3 retries = 4 posts
    assert len(posts) == 4
    sleeps = [c.args[0] for c in fake_sleep.call_args_list]
    assert sleeps == [1.0, 2.0, 4.0]


def test_call_retries_on_500_then_succeeds():
    call, _ = make_judge_caller("k", "model")
    responses = [_http_err_resp(500), _ok_resp("recovered")]

    def fake_post(*a, **k):
        return responses.pop(0)

    with patch("taxonomy_agent.judge.requests.post", side_effect=fake_post), \
         patch("taxonomy_agent.judge.time.sleep") as fake_sleep:
        assert call("prompt") == "recovered"
    fake_sleep.assert_called_once_with(1.0)


def test_usage_sink_receives_http_status():
    captured: list[dict] = []
    call, _ = make_judge_caller("k", "model", usage_sink=captured.append)
    resp = _ok_resp_with_usage(prompt_tokens=10, completion_tokens=5)
    resp.status_code = 200
    with patch("taxonomy_agent.judge.requests.post", return_value=resp):
        call("prompt")
    assert len(captured) == 1
    assert captured[0]["http_status"] == 200


def test_parallel_on_reply_fires_for_failures_too():
    """A None result must still trigger on_reply so the streaming sink can
    persist a placeholder."""
    _, parallel = make_judge_caller("k", "model")

    def fake_post(*a, data=None, **kw):
        body = _json.loads(data)
        if "fail" in body["messages"][0]["content"]:
            raise RuntimeError("network")
        return _ok_resp(content="ok")

    received: list[tuple[int, str | None]] = []

    with patch("taxonomy_agent.judge.requests.post", side_effect=fake_post), \
         patch("taxonomy_agent.judge.time.sleep"):
        parallel(["good", "fail"], concurrency=2,
                 on_reply=lambda i, r: received.append((i, r)))

    assert sorted(received) == [(0, "ok"), (1, None)]
