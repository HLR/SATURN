"""Agent.cache is isolated between concurrent execute phases."""
import asyncio
from saturn.vlm.agent import Agent


def _phase(agent, tag):
    agent.clean_cache()
    agent.cache["scores"][("phrase", 12, "vlm")] = [tag]
    agent.cache["images"][12] = [f"img-{tag}"]
    return agent.cache["scores"][("phrase", 12, "vlm")][0], agent.cache["images"][12][0]


def test_concurrent_phases_are_isolated():
    a = Agent()
    async def run():
        return await asyncio.gather(*(asyncio.to_thread(_phase, a, t) for t in ("A", "B", "C", "D")))
    out = asyncio.run(run())
    assert out == [("A", "img-A"), ("B", "img-B"), ("C", "img-C"), ("D", "img-D")]


def test_clean_cache_does_not_wipe_another_context():
    a = Agent()
    import contextvars
    ctx = contextvars.copy_context()
    ctx.run(lambda: (a.clean_cache(), a.cache["images"].__setitem__(5, ["x"])))
    a.clean_cache()                       # main context
    assert a.cache["images"].get(5) in (None, [])   # main sees nothing
    assert ctx.run(lambda: a.cache["images"][5]) == ["x"]   # other context intact
