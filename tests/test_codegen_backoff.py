"""The code generator waits out rate limits instead of losing the program."""
import os
import sys

import httpx
import openai
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from saturn.codegen.generator import CodeGenerator  # noqa: E402


class _Completions:
    def __init__(self, failures):
        self.failures, self.calls = failures, 0

    def create(self, **kw):
        self.calls += 1
        if self.calls <= self.failures:
            req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            raise openai.RateLimitError("rate-limited upstream", response=httpx.Response(429, request=req), body=None)
        return "ok"


def _gen(failures, monkeypatch):
    g = CodeGenerator.__new__(CodeGenerator)
    comp = _Completions(failures)
    g.client = type("C", (), {"chat": type("Ch", (), {"completions": comp})()})()
    monkeypatch.setattr("time.sleep", lambda s: None)
    return g, comp


def test_rate_limits_are_retried_until_the_call_succeeds(monkeypatch):
    g, comp = _gen(3, monkeypatch)
    assert g._create({}) == "ok" and comp.calls == 4


def test_a_persistent_rate_limit_still_raises(monkeypatch):
    g, comp = _gen(100, monkeypatch)
    with pytest.raises(openai.RateLimitError):
        g._create({})
    assert comp.calls == len(CodeGenerator.RETRY_WAITS) + 1
