"""The codegen seed sent with every code-LLM request."""

import json

import pytest

from saturn.codegen.generator import CodeGenerator


@pytest.fixture()
def gen(tmp_path, monkeypatch):
    cache = tmp_path / "cache.json"
    objects = tmp_path / "cache_objects.json"
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{query}")

    def make(program_cache, objects_cache):
        cache.write_text(json.dumps(program_cache))
        objects.write_text(json.dumps(objects_cache))
        g = CodeGenerator(
            api_key="test-key-unused",
            model_name="test-model",
            program_cache_path=str(cache),
            code_prompt_path=str(prompt),
            write_program_cache=False,
            use_cache=True,
            provider="deepseek",
        )
        # Any cache miss reaching the API is a test failure, not a network call.
        class _Boom:
            def __getattr__(self, name):
                raise AssertionError("cache miss reached the API client")
        g.client = _Boom()
        return g

    return make


# ---------------------------------------------------------------- codegen seed
def _capture_kwargs(gen):
    captured = {}

    class _Completions:
        def create(self, **kw):
            captured.update(kw)
            raise RuntimeError("stop-after-capture")

    class _Chat:
        completions = _Completions()

    gen.client = type("C", (), {"chat": _Chat()})()
    try:
        gen._chat_completion([{"role": "user", "content": "hi"}])
    except RuntimeError:
        pass
    return captured


def test_codegen_seed_zero_by_default(gen, monkeypatch):
    """Deterministic decoding by default: seed 0 when SAPY_CODEGEN_SEED is unset."""
    monkeypatch.delenv("SAPY_CODEGEN_SEED", raising=False)
    assert _capture_kwargs(gen({}, {}))["seed"] == 0


def test_codegen_seed_passed_through(gen, monkeypatch):
    monkeypatch.setenv("SAPY_CODEGEN_SEED", "7")
    assert _capture_kwargs(gen({}, {}))["seed"] == 7


def test_codegen_seed_non_integer_ignored(gen, monkeypatch):
    """A typo must not abort a multi-hour run."""
    monkeypatch.setenv("SAPY_CODEGEN_SEED", "abc")
    assert "seed" not in _capture_kwargs(gen({}, {}))
