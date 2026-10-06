"""Pipeline step 2: Build scene (async-bridged; the runner's ONE build_scene).

``loop``, ``args``, and ``scene_cache`` are bound by the runner via
``functools.partial``.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, List

from saturn.pipeline.models import Models
from saturn.scene.build.load_async import _run_sync_on_loop
from saturn.log import get_logger

log = get_logger(__name__)


def build_scene(
    images: List[Any],
    models: Models,
    *,
    loop: asyncio.AbstractEventLoop,
    args: argparse.Namespace,
    scene_cache,
    keywords=None,
    unique_keywords=None,
):
    from saturn.scene import load_scene_async
    from saturn.scene.build.cache import scene_key
    _scene_cache = scene_cache

    def _fresh():
        return _run_sync_on_loop(
            loop,
            load_scene_async(
                images=images,
                keywords=keywords,
                unique_keywords=unique_keywords,
                sam3=models.sam3,
                orientation_provider=models.orientation_provider,
                vggt_reconstructor=models.vggt_reconstructor,
            ),
        )

    # In-process cache per image set: perception is identical for every
    # question on the same images. Cache the
    # PRE-MUTATION scene as a dict and give each question its own copy;
    # the raw reconstruction attributes are carried along so post-build
    # grounding (planner phrases) still works on a cached scene.
    paths = [getattr(im, "_sapy_path", None) for im in images]
    if not all(paths):
        return _fresh()
    key = scene_key(paths, keywords, None,
                    unique_keywords=tuple(sorted(unique_keywords)) if unique_keywords else ())
    _AUX = ("_vggt_result", "_depth_predictions", "_world_geometries", "ground_info")

    def _build_entry():
        scene = _fresh()
        return scene.to_dict(include_points=True), {a: getattr(scene, a, None) for a in _AUX}

    data, aux = _scene_cache.get_or_build(key, _build_entry)
    from saturn.scene.scene import Scene
    scene = Scene.from_dict(data, images=images)
    for a, v in aux.items():
        if v is not None:
            setattr(scene, a, v)
    # from_dict drops the non-serializable runtime hooks (detect_fn etc.);
    # planner-driven grounding needs them on EVERY question, cached or not.
    from saturn.scene.build.load_async import attach_runtime_hooks
    attach_runtime_hooks(
        scene, loop=loop, sam3=models.sam3,
        orientation_provider=models.orientation_provider,
    )
    return scene
