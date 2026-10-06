"""The vLLM scoring agent: crop numbering in single-view prompts, the multiview
prompt for class-type scores, and case-insensitive yes/no token buckets."""
import asyncio
import types

import torch
from PIL import Image

from saturn.vlm.client import ProbabilisticVLMClient
from saturn.vlm.qwen_vllm import QwenVLvLLM


class FakeClient:
    def __init__(self):
        self.calls = []

    async def score(self, images, text):
        self.calls.append((len(images), text))
        return 0.9

    async def generate(self, *a, **k):
        return ""


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _agent():
    client = FakeClient()
    return QwenVLvLLM(client), client


IMG = Image.new("RGB", (100, 100))


def test_singleview_prompt_numbers_crops_from_image_2():
    agent, client = _agent()
    _run(agent._score_async(IMG, "Is it red?", [[10, 10, 50, 50]], type="default"))
    n_images, text = client.calls[-1]
    assert n_images == 2
    assert "Image-2 is the zoomed-in" in text
    assert "Image-3" not in text

    _run(agent._score_async(
        IMG, "Is A left of B?", [[10, 10, 50, 50], [50, 50, 90, 90]], type="default"))
    n_images, text = client.calls[-1]
    assert n_images == 3
    assert "Image-2 is the zoomed-in" in text and "Image-3 is the zoomed-in" in text
    assert "Image-4" not in text


def test_class_type_score_uses_multiview_prompt():
    agent, client = _agent()
    out = _run(agent._score_async(
        IMG, "chair", [[10, 10, 50, 50]], type="class",
        all_view_images=[IMG, IMG, IMG], view_index=1,
    ))
    assert abs(float(out) - 0.9) < 1e-6
    n_images, text = client.calls[-1]
    assert n_images == 4  # 3 views + crop
    assert "Image-2 shows the scene with the object highlighted" in text
    assert "Image-4 is a zoomed-in crop" in text
    assert "Is the object in the red bounding box in Image-2 a chair?" in text


def _resp(*entries):
    E = lambda t, l: types.SimpleNamespace(token=t, logprob=l)
    top = [E(t, l) for t, l in entries]
    return types.SimpleNamespace(choices=[types.SimpleNamespace(
        logprobs=types.SimpleNamespace(content=[types.SimpleNamespace(top_logprobs=top)]))])


def test_yes_no_buckets_are_case_insensitive():
    c = ProbabilisticVLMClient(base_url="http://x", model="m")
    assert c._extract_yes_prob(_resp(("yes", -0.05), ("No", -4.0))) > 0.9
    assert c._extract_yes_prob(_resp(("Yes", -4.0), (" no", -0.05))) < 0.1
    # Canonical tokens are unchanged.
    p = c._extract_yes_prob(_resp(("Yes", -0.1), ("No", -2.4)))
    assert abs(p - 1 / (1 + 2.718281828 ** -2.3)) < 1e-6
