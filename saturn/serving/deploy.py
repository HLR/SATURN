"""Entrypoint: deploy the Ray Serve apps (Orient-Anything, VGGT, SAM3).

Run:

    python -m saturn.serving.deploy

Expects Ray to already be running (``ray start --head ...``). Reads env vars:

* ``SAPY_ENABLE_VGGT`` (default ``1``)
* ``SAPY_ENABLE_SAM3`` (default ``1``)
"""

from __future__ import annotations

from saturn.settings import env
import sys
import time
from saturn.log import configure, get_logger, progress

log = get_logger(__name__)


APP_NAME = "sapy"


def select_apps(enable_vggt: bool, enable_sam3: bool) -> set[str]:
    """Names of the apps main() deploys for these flags. oriany is always deployed."""
    wanted = {"oriany"}
    for name, on in (("vggt", enable_vggt), ("sam3", enable_sam3)):
        if on:
            wanted.add(name)
    return wanted



def main() -> int:
    configure()
    from dotenv import load_dotenv
    load_dotenv()
    try:
        import ray
        from ray import serve
    except ImportError:
        log.error("[deploy] Ray is not installed in this environment.")
        return 2

    from saturn.serving.deployments import sam3_deployment, vggt_deployment, oriany_deployment

    if not ray.is_initialized():
        ray.init(address="auto", namespace=APP_NAME)
    # Ray Serve's HTTP proxy defaults to port 8000, which collides with vLLM's
    # OpenAI-compatible server on the same host; the port is configurable
    # (vLLM=8000, Serve=8001 by convention).
    serve_http_port = int(env("SAPY_SERVE_HTTP_PORT"))
    serve.start(
        detached=True,
        http_options={"host": "127.0.0.1", "port": serve_http_port},
    )

    enable_vggt = env("SAPY_ENABLE_VGGT") == "1"
    enable_sam3 = env("SAPY_ENABLE_SAM3") == "1"

    # Deploy order IS the Ray bin-packing order for fractional-GPU actors: the
    # app deployed first claims the lowest-index cards. oriany has the highest
    # call rate (once per detected object per view) and is sized to own whole
    # cards, so it is placed FIRST; otherwise vggt would hold half of card 0
    # and oriany's replicas would interleave across every card.
    selected = select_apps(enable_vggt, enable_sam3)
    apps = {}
    apps[oriany_deployment.NAME] = oriany_deployment.build()
    if "vggt" in selected:
        apps[vggt_deployment.NAME] = vggt_deployment.build()
    if "sam3" in selected:
        apps[sam3_deployment.NAME] = sam3_deployment.build()

    progress(
        "[deploy] config: "
        f"enable_vggt={enable_vggt} enable_sam3={enable_sam3}"
    )

    for name, app in apps.items():
        progress(f"[deploy] Deploying {name}…")
        serve.run(app, name=name, route_prefix=None, blocking=False)

    # Wait until all requested apps are healthy.
    deadline = time.time() + 600
    while time.time() < deadline:
        status = serve.status()
        healthy = {n: s.status for n, s in status.applications.items()}
        progress(f"[deploy] status = {healthy}")
        if all(v == "RUNNING" for v in healthy.values()) and set(healthy) >= set(apps):
            progress("[deploy] All apps RUNNING.")
            return 0
        time.sleep(5)

    log.error("[deploy] Timed out waiting for apps to become RUNNING.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
