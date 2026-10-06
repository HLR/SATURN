"""Agent.score_multiview: the batched and sequential paths receive the same arguments; num_objects is checked."""
import types

import pytest
import torch

from saturn.vlm.agent import Agent


class _RecordingAgent(Agent):
    """Records every ``_score`` call; ``batched`` installs a ``_score_many``."""

    def __init__(self, batched):
        super().__init__()
        self.device = torch.device("cpu")
        self.calls = []
        if batched:
            self._score_many = lambda jobs: [self._score(**job) for job in jobs]

    def _score(self, image, question, bboxes, **kwargs):
        self.calls.append({"image": image, "question": question, "bboxes": list(bboxes), **kwargs})
        return torch.tensor(0.5)


def _scene(n_objects=1):
    objects = [
        types.SimpleNamespace(
            views=[0], per_view_bboxes={0: [1, 2, 3, 4]}, per_view_masks={},
        )
        for _ in range(n_objects)
    ]
    return types.SimpleNamespace(objects=objects, cameras=[], images=["view0"])


def _normalized(call):
    call = dict(call)
    if call.get("masks") is not None:
        call["masks"] = list(call["masks"])
    return call


@pytest.mark.parametrize("type_", [None, "class"])
def test_batched_and_sequential_paths_send_the_same_score_arguments(type_):
    extra = {"candidates": ["chair", "table"], "history": ["earlier turn"]}
    batched, sequential = _RecordingAgent(True), _RecordingAgent(False)
    out_b = batched.score_multiview("chair", type=type_, scene=_scene(), **extra)
    out_s = sequential.score_multiview("chair", type=type_, scene=_scene(), **extra)
    assert torch.equal(out_b, out_s)
    assert len(batched.calls) == len(sequential.calls) == 1
    assert _normalized(batched.calls[0]) == _normalized(sequential.calls[0])
    assert batched.calls[0]["candidates"] == extra["candidates"]
    assert batched.calls[0]["history"] == extra["history"]


@pytest.mark.parametrize("n_objects", [0, 1])
@pytest.mark.parametrize("num_objects", [0, 3])
def test_unsupported_num_objects_raises_with_and_without_objects(n_objects, num_objects):
    agent = _RecordingAgent(True)
    with pytest.raises(ValueError, match="num_objects=1 or 2"):
        agent.score_multiview("chair", num_objects=num_objects, scene=_scene(n_objects))


def test_empty_scene_still_returns_zero_scores_for_one_and_two_objects():
    agent = _RecordingAgent(True)
    scene = _scene(0)
    scene.cameras = [object(), object()]
    assert torch.equal(agent.score_multiview("chair", scene=scene), torch.zeros(2))
    assert torch.equal(
        agent.score_multiview("on top of", num_objects=2, scene=scene), torch.zeros(2, 2)
    )
    assert agent.calls == []
