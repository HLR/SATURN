"""Cached accessors for Ray Serve deployment handles.

Usage:

    from saturn.serving.clients.ray_handles import get_handle
    handle = get_handle("sam3")
    result = await handle.predict_with_masks.remote(image_bytes, text, 0.0)

Handles are resolved once per process. Each deployment is published as its
own Serve application (app names: ``sam3``, ``vggt``, ``oriany``), so
``get_handle("sam3")`` resolves app ``sam3`` first and falls back to
``app_name="sapy"``.
"""

from __future__ import annotations

import functools
from typing import Any

APP_NAME_FALLBACK = "sapy"

DEPLOYMENT_NAMES = {"sam3", "vggt", "oriany"}


@functools.lru_cache(maxsize=None)
def get_handle(name: str) -> Any:
    if name not in DEPLOYMENT_NAMES:
        raise KeyError(
            f"Unknown deployment {name!r}; valid options: {sorted(DEPLOYMENT_NAMES)}"
        )
    from ray import serve  # type: ignore

    for app_name in (name, APP_NAME_FALLBACK):
        try:
            return serve.get_deployment_handle(name, app_name=app_name)
        except KeyError:
            continue

    raise KeyError(
        f"Deployment {name!r} was not found in app {name!r} or "
        f"fallback app {APP_NAME_FALLBACK!r}."
    )


