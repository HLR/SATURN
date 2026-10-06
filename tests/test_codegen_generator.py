"""CodeGenerator: concurrent use of one generator, retries, decomposition,
code extraction from replies and API-key setup."""

import json
import random
import threading
import time

import pytest

from saturn.codegen import generator as G


@pytest.fixture
def cg(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    cache = tmp_path / "c.json"
    prompt = tmp_path / "p.txt"
    prompt.write_text("Q: {query}\n")
    cache.write_text("{}")
    (tmp_path / "c_objects.json").write_text("{}")
    gen = G.CodeGenerator("x", "deepseek-chat", str(cache), str(prompt),
                          write_program_cache=True, use_cache=True)
    gen.sent = []

    def fake(messages, response_format=None):
        gen.sent.append(messages)
        time.sleep(random.uniform(0, 0.01))
        return "```python\nanswer = %d\n```" % len(messages[-1]["content"])

    gen._chat_completion = fake
    gen._response_content = staticmethod(lambda r: r)
    return gen


# --- concurrent generate_code on one shared generator -------------------------

def test_concurrent_generate_code_keeps_every_program(cg):
    n = 16
    errors, results = [], {}

    def work(i):
        try:
            results[i] = cg.generate_code(f"question {i}")[0]
        except Exception as e:   # noqa: BLE001 -- the test reports it
            errors.append(repr(e))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert all(results[i] for i in range(n))
    assert len(json.load(open(cg.cache_file))) == n


def test_persist_prefers_program_already_on_disk(cg):
    code, objs = cg._persist_caches_atomic("k", "first", "o1")
    assert (code, objs) == ("first", "o1")
    code, objs = cg._persist_caches_atomic("k", "second", "o2")
    assert (code, objs) == ("first", "o1")


# --- a retry replays the prompt of the first attempt ---------------------------

def test_retry_prompt_replays_first_attempt(cg):
    cg.generate_code("q")
    first_user = cg.sent[-1][1]["content"]
    cg.retry_generate_code("q", "answer = 1", "boom")
    retry_user = cg.sent[-1][1]["content"]
    assert retry_user == first_user


def test_retry_prompt_format_override(cg):
    cg.retry_generate_code("q", "answer = 1", "boom", prompt_format="CUSTOM {query}")
    assert cg.sent[-1][1]["content"] == "CUSTOM q"


# --- generate_code_with_decomposition runs --------------------------------------


# ---------------------------------------------------------------- code extraction and client setup

def test_extract_code_rejects_prose_without_fence():
    """A reply without a ```python fence (prose, a refusal, a truncated block) yields no program."""
    from saturn.codegen.generator import extract_code
    assert extract_code("```python\nx = 1\n```") == "x = 1"
    assert extract_code("x = 1\nreturn x") is None or extract_code("x = 1") == "x = 1"
    assert extract_code("Sure! Here is the program:\n```python\nx = 1") is None      # truncated
    assert extract_code("I cannot help with that.") is None


def test_empty_api_key_falls_back_to_env(monkeypatch):
    """An empty api_key (the runner passes "") falls back to DEEPSEEK_API_KEY."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-from-env-000000000000")
    from saturn.codegen.generator import CodeGenerator
    cg = CodeGenerator(api_key="", model_name="deepseek-chat", program_cache_path="/nonexistent/x.json",
                       code_prompt_path="prompts/force3d_ref.txt", use_cache=False, provider="deepseek")
    assert cg.client.api_key == "sk-test-from-env-000000000000"


def test_only_deepseek_and_openrouter_providers(tmp_path):
    import pytest
    from saturn.codegen.generator import CodeGenerator
    for provider in ("openai", "vllm"):
        with pytest.raises(ValueError, match="deepseek"):
            CodeGenerator(api_key="k", model_name="m", program_cache_path=str(tmp_path / "c.json"),
                          code_prompt_path="prompts/vqa.txt", provider=provider)
