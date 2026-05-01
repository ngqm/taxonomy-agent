"""make_judge_caller — retry behaviour with mocked OpenRouter calls."""
from __future__ import annotations

import json as _json
from unittest.mock import MagicMock, patch

from taxonomy_agent.judge import make_judge_caller


def _ok_resp(content: str = "ok") -> MagicMock:
    r = MagicMock()
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
