"""Object orientation providers (Orient-Anything local and via Ray Serve).

Model-side; imported only by pipeline.models and serving.
"""
from saturn.perception.orientation.orientanything_serve import OrientAnythingProvider

__all__ = ["OrientAnythingProvider"]
