"""Image wire codec shared by the Ray Serve deployments and their clients."""

from __future__ import annotations

import io

from PIL import Image


def encode_image(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def decode_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
