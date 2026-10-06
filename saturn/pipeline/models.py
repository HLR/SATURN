"""Model container + service wiring + sync bridges for the async runner."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from saturn.codegen import CodeGenerator
from saturn.scene.build.load_async import _run_sync_on_loop
from saturn.log import get_logger, progress

log = get_logger(__name__)


def load_pinned_programs(path):
    """{sample id: program} to replay (``--programs_by_id``), or None."""
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        programs = json.load(f)
    log.info(f"[replay] {len(programs)} pinned programs from {path}")
    return {str(k): v for k, v in programs.items()}


class Models:
    """Container for all loaded models — avoids passing 8 variables everywhere."""

    def __init__(self):
        self.code_generator: CodeGenerator = None
        self.pinned_programs = None  # {sample id: program} from --programs_by_id
        self.vl_model = None
        self.sam3 = None
        self.orientation_provider = None
        self.vggt_reconstructor = None
        self.planner = None


def _install_vlm_sync_bridge(agent: Any, loop: asyncio.AbstractEventLoop) -> Any:
    def _query_sync(image, text, max_new_tokens: int = 128):
        return _run_sync_on_loop(
            loop,
            agent._query_async(image, text, max_new_tokens=max_new_tokens),
        )

    def _ground_sync(image, phrase):
        return _run_sync_on_loop(loop, agent.ground_async(image, phrase))

    def _score_sync(*args, **kwargs):
        return _run_sync_on_loop(loop, agent._score_async(*args, **kwargs))

    def _score_many_sync(jobs):
        # jobs: list of kwargs dicts; each is forwarded to agent._score_async.
        # Dispatched in parallel so the score-loop's (object × view) fan-out
        # saturates the vLLM DP replicas instead of serializing.
        async def _gather():
            return await asyncio.gather(*[agent._score_async(**kw) for kw in jobs])
        import time as _time
        _t0 = _time.perf_counter()
        result = _run_sync_on_loop(loop, _gather())
        _dt = _time.perf_counter() - _t0
        log.debug(f"[_score_many] {len(jobs):>3} jobs in {_dt*1000:.0f} ms ({len(jobs)/_dt:.1f} req/s)")
        return result

    def _score_simple_sync(images, text, target_token="Yes", temperature=1.0):
        return _run_sync_on_loop(
            loop,
            agent._score_simple_async(
                images, text, target_token=target_token, temperature=temperature,
            ),
        )

    # Pose-constraint extraction bridge: always installed (cheap), used only
    # with --use_pose_constraints. The import is deferred to first use.
    def _extract_pose_constraints_sync(question, images, num_cameras=None):
        from saturn.planning.constraint_extractor import (
            extract_camera_constraints_async,
        )
        return _run_sync_on_loop(
            loop,
            extract_camera_constraints_async(
                question=question,
                images=images,
                vlm_generate=agent.client.generate,
                num_cameras=num_cameras,
            ),
        )

    agent._query = _query_sync
    agent.ground = _ground_sync
    agent._score = _score_sync
    agent._score_many = _score_many_sync
    agent._score_simple = _score_simple_sync
    agent.extract_pose_constraints = _extract_pose_constraints_sync
    return agent


def _install_sam3_sync_bridge(provider: Any, loop: asyncio.AbstractEventLoop) -> Any:
    """Polymorphic shim around the SAM3 provider's hot methods.

    Two distinct caller classes share the same attribute name:
      * Sync callers run on a worker thread spawned by ``asyncio.to_thread``
        — they need a fully resolved value.
      * Async callers (``saturn/scene/build/load_async.py`` →
        ``await _call_maybe_async(sam3.predict_with_masks, ...)``) run on
        the event-loop thread — they need an awaitable.

    The shim picks at call time: on the loop thread we return the original
    coroutine (so ``await`` works); from a worker thread we run sync via the
    cross-thread bridge. A pure sync wrapper would make the async path on the
    event loop hit ``_run_sync_on_loop``'s deadlock guard.
    """

    async_predict_with_masks = provider.predict_with_masks
    async_predict_masks = provider.predict_masks

    def _on_loop_thread() -> bool:
        try:
            return asyncio.get_running_loop() is loop
        except RuntimeError:
            return False

    def _predict_with_masks_dual(image, text: str = "", threshold: float = 0.0):
        coro = async_predict_with_masks(image, text, threshold)
        if _on_loop_thread():
            return coro  # let the awaiting async caller drive it
        return _run_sync_on_loop(loop, coro)

    def _mask_from_boxes_dual(image, bboxes):
        coro = async_predict_masks(image, list(bboxes))
        if _on_loop_thread():
            return coro
        return _run_sync_on_loop(loop, coro)

    provider.predict_with_masks = _predict_with_masks_dual
    # The sync shim in RayServeSAM3MaskProvider uses ``run_until_complete``,
    # which crashes when invoked from a worker thread (loop already running).
    provider.mask_from_boxes = _mask_from_boxes_dual
    return provider


async def bring_up_services(args: argparse.Namespace) -> dict:
    from saturn.vlm.client import ProbabilisticVLMClient

    client = ProbabilisticVLMClient(
        base_url=args.vlm_base_url,
        model=args.vlm_model_name,
        max_concurrency=args.vlm_max_concurrency,
    )

    handles: dict[str, Any] = {}
    if not args.skip_ray_handles:
        from saturn.serving.clients.ray_handles import get_handle

        for name in ("sam3", "vggt", "oriany"):
            handles[name] = get_handle(name)

    return {"vlm_client": client, "ray_handles": handles}


def build_models_async(
    args: argparse.Namespace,
    services: dict,
    loop: asyncio.AbstractEventLoop,
    dataset: Any = None,
) -> Models:
    from saturn.soft_logic import ProbabilisticTensor
    from saturn.planning.query_planner import QueryPlanner
    from saturn.vlm.qwen_vllm import QwenVLvLLM
    from saturn.perception.masks.rayserve import (
        RayServeSAM3MaskProvider,
    )
    from saturn.perception.orientation.rayserve import (
        RayServeOriAnyOrientationProvider,
    )

    m = Models()
    m.pinned_programs = load_pinned_programs(getattr(args, "programs_by_id", None))
    m.code_generator = CodeGenerator(
        api_key="",   # each provider reads its own key from the environment (.env)
        model_name=args.code_gen_model_name,
        program_cache_path=args.program_cache,
        code_prompt_path=args.code_prompt,
        write_program_cache=args.write_program_cache,
        provider=args.code_gen_provider,
    )

    vl_model = QwenVLvLLM(services["vlm_client"], wrapper=ProbabilisticTensor)
    m.vl_model = _install_vlm_sync_bridge(vl_model, loop)

    if services["ray_handles"]:
        m.sam3 = RayServeSAM3MaskProvider(services["ray_handles"]["sam3"])
        # Bridge the async SAM3 provider so sync callers on worker threads can
        # use it without deadlocking on the event loop.
        _install_sam3_sync_bridge(m.sam3, loop)
        m.orientation_provider = RayServeOriAnyOrientationProvider(
            services["ray_handles"]["oriany"]
        )
        m.vggt_reconstructor = services["ray_handles"].get("vggt")

    cache_path = getattr(
        args, "planner_cache", "cache/planner.json"
    )
    m.planner = QueryPlanner(
        vl_model=m.vl_model,
        cache_path=cache_path,
        write_cache=True,
        # The planner writes its sketch before `object_groundings`; a tight
        # budget truncates the JSON, which the recovery regex can only
        # partly salvage.
        # Budget: vLLM max_model_len=16384, prompt ~3000, 2-img ~512 →
        # ~12.8k available; 4096 leaves ~8.7k headroom.
        max_new_tokens=4096,
        verbose=True,
    )
    progress(f"Planner ready (cache={cache_path}).")

    return m
