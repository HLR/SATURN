"""Codegen determinism: content-addressed program cache, fixed seed, provider pin,
minimal retry prompt, and SCENE FACTS without the per-object index table."""

import json

import pytest

from saturn.codegen.generator import PROGRAM_KEY_PREFIX, CodeGenerator

Q = "Is the chair left of the table?"
PROG = "def logic_executor(query, score_fn, query_fn, scene, images, history):\n    return 'MARKER'"
RESPONSE = f"<objects>chair, table</objects>\n<code>\n{PROG}\n</code>"


class _FakeClient:
    """Records every request; answers with a fixed response."""

    def __init__(self, content=RESPONSE):
        self.calls = []
        client = self

        class _Completions:
            def create(self, **kw):
                client.calls.append(kw)
                msg = type("M", (), {"content": content, "reasoning_content": None})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

        self.chat = type("Chat", (), {"completions": _Completions()})()


@pytest.fixture()
def make_gen(tmp_path, monkeypatch):
    monkeypatch.delenv("SAPY_CODEGEN_SEED", raising=False)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("TEMPLATE v1\n{scene_facts_block}\nQ: {query}")

    def make(program_cache=None, objects_cache=None, model="test-model", write_program_cache=True):
        cache = tmp_path / "cache.json"
        cache.write_text(json.dumps(program_cache or {}))
        (tmp_path / "cache_objects.json").write_text(json.dumps(objects_cache or {}))
        g = CodeGenerator(api_key="unused", model_name=model, program_cache_path=str(cache),
                          code_prompt_path=str(prompt), write_program_cache=write_program_cache,
                          use_cache=True, provider="deepseek")
        g.client = _FakeClient()
        return g

    make.prompt = prompt
    make.cache = tmp_path / "cache.json"
    return make


# ------------------------------------------------------------- cache key
def test_new_programs_written_under_content_key_only(make_gen):
    g = make_gen()
    code, objs = g.generate_code(Q, scene_facts_block="SCENE FACTS: a")
    assert "MARKER" in code and objs == "chair, table"
    stored = json.loads(make_gen.cache.read_text())
    assert list(stored) and all(k.startswith(PROGRAM_KEY_PREFIX) for k in stored)
    assert list(stored.values()) == [PROG]  # raw snippet, not template-wrapped


def test_second_call_replays_without_api(make_gen):
    g = make_gen()
    first = g.generate_code(Q, scene_facts_block="SCENE FACTS: a")
    again = g.generate_code(Q, scene_facts_block="SCENE FACTS: a")
    assert first == again and len(g.client.calls) == 1


@pytest.mark.parametrize("change", ["template", "model", "seed", "facts"])
def test_any_input_change_misses_the_cache(make_gen, monkeypatch, change):
    g = make_gen()
    g.generate_code(Q, scene_facts_block="SCENE FACTS: a")
    facts = "SCENE FACTS: a"
    if change == "template":
        make_gen.prompt.write_text("TEMPLATE v2\n{scene_facts_block}\nQ: {query}")
    elif change == "model":
        g = make_gen(program_cache=json.loads(make_gen.cache.read_text()), model="other-model")
    elif change == "seed":
        monkeypatch.setenv("SAPY_CODEGEN_SEED", "1")
    else:
        facts = "SCENE FACTS: b"
    g.generate_code(Q, scene_facts_block=facts)
    assert len(json.loads(make_gen.cache.read_text())) == 2


def test_fresh_and_replayed_program_are_identical(make_gen):
    """Fresh output and a cache replay both pass through fix_framework_code."""
    g = make_gen()
    g.client = _FakeClient("<code>\nx = score('green bounding box is red', 1)  # c\n</code>")
    fresh, _ = g.generate_code(Q)
    replay, _ = g.generate_code(Q)
    assert fresh == replay and "red bounding box" in fresh


def test_unparseable_response_is_not_cached(make_gen):
    g = make_gen()
    g.client = _FakeClient("I cannot help with that.")
    assert g.generate_code(Q)[0] is None
    assert json.loads(make_gen.cache.read_text()) == {}


# ------------------------------------------------------------- decoding
def test_seed_zero_and_temperature_zero_by_default(make_gen):
    kw = make_gen()._request_kwargs([{"role": "user", "content": "hi"}])
    assert kw["seed"] == 0 and kw["temperature"] == 0.0


def test_explicit_seed_is_sent(make_gen, monkeypatch):
    monkeypatch.setenv("SAPY_CODEGEN_SEED", "7")
    assert make_gen()._request_kwargs([])["seed"] == 7


def _openrouter(monkeypatch, provider):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("SAPY_OPENROUTER_PROVIDER", provider)
    return CodeGenerator(api_key="", model_name="qwen/qwen3-coder", program_cache_path="/nonexistent/x.json",
                         code_prompt_path="/nonexistent/p.txt", use_cache=False, provider="openrouter")


def test_openrouter_empty_provider_sends_no_pin(monkeypatch):
    body = _openrouter(monkeypatch, "")._request_kwargs([])["extra_body"]
    assert body == {"reasoning": {"enabled": False}}


def test_openrouter_provider_pin_and_no_thinking(monkeypatch):
    body = _openrouter(monkeypatch, "StreamLake")._request_kwargs([])["extra_body"]
    assert body["reasoning"] == {"enabled": False}
    assert body["provider"]["order"] == ["StreamLake"] and body["provider"]["allow_fallbacks"] is False


# ------------------------------------------------------------- retry prompt
def test_retry_prompt_is_minimal(make_gen):
    g = make_gen()
    code, _ = g.retry_generate_code(Q, failed_code="x = 1/0", error_message="ZeroDivisionError")
    assert "MARKER" in code
    msgs = g.client.calls[0]["messages"]
    assert msgs[2] == {"role": "assistant", "content": "```python\nx = 1/0\n```"}
    last = msgs[3]["content"]
    assert "ZeroDivisionError" in last and "same output format" in last
    assert "Handle cases" not in last and "3 objects" not in last
    assert json.loads(make_gen.cache.read_text()) == {}  # retries are never cached


# ------------------------------------------------------------- scene facts
def _scene():
    from saturn.scene.scene import Scene
    scene = Scene.__new__(Scene)
    obj = type("O", (), {"label": "chair", "views": [0, 1]})()
    scene.objects, scene.cameras = [obj], [object(), object()]
    scene._setup_caption = "a living room"
    scene._axis_convention_M = None
    return scene


def test_scene_facts_omit_object_index_table_by_default():
    from saturn.scene.scene import Scene
    facts = Scene.dump_facts_str(_scene())
    assert "setup_caption: a living room" in facts and "objects_count: 1" in facts
    assert "cameras_count: 2" in facts and "axis_convention" in facts
    assert "[0]" not in facts and "merged=" not in facts


def test_scene_facts_object_table_opt_in():
    from saturn.scene.scene import Scene
    facts = Scene.dump_facts_str(_scene(), objects_table=True)
    assert "[0] 'chair'" in facts and "merged=True" in facts


def test_question_key_is_not_replayed(make_gen):
    g = make_gen({Q: PROG}, {Q: "chair"})
    g.generate_code(Q)
    assert len(g.client.calls) == 1  # regenerated: only content-addressed keys are replayed
