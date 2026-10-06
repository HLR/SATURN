"""QwenVLvLLM — Qwen VL agent backed by an external vLLM server.

Implements the ``Agent`` hooks (``_score``, ``_query``, ``ground``) by routing every call through an async ``ProbabilisticVLMClient``.

* ``_score`` returns CPU tensors (``requires_grad=True``)
  derived from vLLM top-logprobs.
* ``ground`` parses Qwen's 0-1000 ``bbox_2d`` reply (a fenced JSON block, a
  bare JSON list or dict, or the first four integers).

All public methods are ``async`` *except* the sync-bridged helpers used from
generated code, which are provided by the runner's sandbox wrapper.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch
from PIL import Image

import re
import json as _json

from .agent import Agent


# Grounding helpers.
def _build_grounding_prompt(phrase: str) -> str:
    label = " ".join(str(phrase).strip().split()) or "object"
    safe_label = label.replace('"', "\\\"")
    return (
        f'Locate the best matching instance of "{safe_label}" in this image. '
        f'If multiple instances are present, return the best matching one first. '
        'Report bbox coordinates in Qwen\'s 0-1000 coordinate scale using JSON '
        f'format like this: [{{"bbox_2d": [x1, y1, x2, y2], "label": "{safe_label}"}}]'
    )


def _extract_grounding_bbox(output_text: str, image_size) -> list[int] | None:
    orig_w, orig_h = image_size

    def _denorm(coords):
        x1, y1, x2, y2 = [float(c) for c in coords[:4]]
        x1 = int(x1 / 1000 * orig_w)
        y1 = int(y1 / 1000 * orig_h)
        x2 = int(x2 / 1000 * orig_w)
        y2 = int(y2 / 1000 * orig_h)

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(x1 + 1, min(x2, orig_w))
        y2 = max(y1 + 1, min(y2, orig_h))
        return [x1, y1, x2, y2]

    def _parse_candidate(candidate):
        try:
            data = _json.loads(candidate)
        except (TypeError, ValueError):
            return None

        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return None

        for item in data:
            if isinstance(item, dict) and "bbox_2d" in item:
                coords = item["bbox_2d"]
                if isinstance(coords, list) and len(coords) >= 4:
                    return _denorm(coords)
        return None

    json_block = re.search(r"```json\s*([\s\S]*?)\s*```", output_text, re.IGNORECASE)
    if json_block:
        bbox = _parse_candidate(json_block.group(1))
        if bbox is not None:
            return bbox

    json_match = re.search(r"\[.*\]", output_text, re.DOTALL)
    if json_match:
        bbox = _parse_candidate(json_match.group())
        if bbox is not None:
            return bbox

    dict_match = re.search(r"\{.*\}", output_text, re.DOTALL)
    if dict_match:
        bbox = _parse_candidate(dict_match.group())
        if bbox is not None:
            return bbox

    matches = re.findall(r"(\d+)", output_text)
    if len(matches) >= 4:
        return _denorm(matches[:4])
    return None


class QwenVLvLLM(Agent):
    """Async agent that delegates VLM calls to an external vLLM server."""

    def __init__(self, client: Any, wrapper: Any = None) -> None:
        super().__init__(wrapper=wrapper)
        self.client = client
        # The device is intentionally cpu — the shim returns CPU tensors.
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_singleview_prompt(
        self,
        image: Image.Image,
        bboxes: Sequence,
        masks: Optional[Sequence],
        question: str,
        type: Optional[str],
    ):
        """Build prompt with marked image + per-object crops."""
        marked_image = self.marker.mark_objects(
            image, bboxes,
            draw_bbox=True, draw_text=False,
            overlay_masks=False, masks=False,
        )

        if masks is not None:
            extra_images = [
                self.extract_object_with_mask(image, masks[i]).crop(
                    (bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3])
                )
                for i in range(len(bboxes))
            ]
        else:
            extra_images = [
                image.crop(
                    (int(bboxes[i][0]), int(bboxes[i][1]),
                     int(bboxes[i][2]), int(bboxes[i][3]))
                )
                for i in range(len(bboxes))
            ]

        image_paths = [marked_image, *extra_images]
        image_prompt = ""
        if extra_images:
            image_prompt += "Image-1 is the source image."
        for i in range(len(extra_images)):
            image_prompt += (
                f"\nImage-{i + 2} is the zoomed-in of the extracted main object "
                f"in the {self.marker.colors[i % len(self.marker.colors)]} bounding box from Image-1."
            )

        image_prompt += f"\nQuestion: {question}\n"
        return image_paths, image_prompt

    def _build_multiview_prompt(
        self,
        image: Image.Image,
        bboxes: Sequence,
        masks: Optional[Sequence],
        question: str,
        all_view_images: Sequence[Image.Image],
        view_index: Optional[int],
    ):
        """Build prompt showing ALL scene views + crop of the target object."""
        n_views = len(all_view_images)
        view_idx = view_index if view_index is not None else 0

        image_paths = []
        image_prompt_parts = []
        for vi, vimg in enumerate(all_view_images):
            if vi == view_idx:
                marked = self.marker.mark_objects(
                    vimg, bboxes,
                    draw_bbox=True, draw_text=False,
                    overlay_masks=False, masks=False,
                )
                image_paths.append(marked)
                image_prompt_parts.append(
                    f"Image-{vi + 1} shows the scene with the object "
                    f"highlighted in a red bounding box."
                )
            else:
                image_paths.append(vimg)
                image_prompt_parts.append(
                    f"Image-{vi + 1} shows another view of the same scene."
                )

        # Zoomed crop of the object
        if masks is not None and masks[0] is not None:
            crop_img = self.extract_object_with_mask(image, masks[0]).crop(
                (bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3])
            )
        else:
            crop_img = image.crop(
                (int(bboxes[0][0]), int(bboxes[0][1]),
                 int(bboxes[0][2]), int(bboxes[0][3]))
            )
        image_paths.append(crop_img)
        image_prompt_parts.append(
            f"Image-{n_views + 1} is a zoomed-in crop of the object "
            f"in the red bounding box from Image-{view_idx + 1}."
        )

        if "red bounding box" not in question.lower():
            question = (
                f"Is the object in the red bounding box in "
                f"Image-{view_idx + 1} a {question}?"
            )

        image_prompt = "\n".join(image_prompt_parts)
        image_prompt += "\nAnswer Yes or No.\n"
        image_prompt += f"\nQuestion: {question}\n"
        return image_paths, image_prompt

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    async def _score_async(
        self,
        image: Image.Image,
        question: str,
        bboxes: Sequence,
        masks: Optional[Sequence] = None,
        candidates: Optional[list] = None,
        type: str = "default",
        history: Any = None,
        target_tokens: Optional[list] = None,
        all_view_images: Optional[Sequence[Image.Image]] = None,
        view_index: Optional[int] = None,
    ) -> torch.Tensor:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy().tolist()

        if all_view_images is not None and type in ("class", "property"):
            image_paths, image_prompt = self._build_multiview_prompt(
                image, bboxes, masks, question, all_view_images, view_index
            )
        else:
            image_paths, image_prompt = self._build_singleview_prompt(
                image, bboxes, masks, question, type
            )

        p = await self.client.score(image_paths, image_prompt)
        t = torch.tensor(float(p), device=self.device, dtype=self.dtype)
        t.requires_grad = True
        return t

    def _score(self, *args, **kwargs):
        # Synchronous fallback — used when a caller on the main thread wants
        # a blocking call. The runner's sandbox bridge replaces ``score`` at
        # the ``Agent.score`` level, so this path is only hit in tests.
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self._score_async(*args, **kwargs)
        )

    async def _score_simple_async(
        self,
        images: Sequence[Image.Image],
        text: str,
        target_token: str = "Yes",
        temperature: float = 1.0,
    ) -> float:
        """P(target_token) for a yes/no prompt — plain float for grounder verify.

        ObjectGrounder._score_candidate calls this; without it the
        ``hasattr(vlm, "_score_simple")`` check fails and disambiguation
        returns 1.0 unconditionally (every candidate ties → first wins).
        """
        imgs = list(images) if not isinstance(images, list) else images
        p_yes = await self.client.score(imgs, text)
        return float(1.0 - p_yes) if target_token != "Yes" else float(p_yes)

    def _score_simple(self, *args, **kwargs):
        # Sync fallback — the runner installs a thread-bridge shim that
        # overrides this attribute when the agent is wired into the loop.
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self._score_simple_async(*args, **kwargs)
        )

    # ------------------------------------------------------------------
    # Query / grounding
    # ------------------------------------------------------------------

    async def _query_async(
        self,
        image: Any,
        text: str,
        max_new_tokens: int = 128,
    ) -> str:
        images = image if isinstance(image, list) else [image]
        return await self.client.generate(images, text, max_tokens=max_new_tokens)

    def _query(self, image, text, max_new_tokens: int = 128) -> str:
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self._query_async(image, text, max_new_tokens=max_new_tokens)
        )

    async def ground_async(
        self, image: Image.Image | List[Image.Image], phrase: str
    ):
        images = image if isinstance(image, list) else [image]
        text = _build_grounding_prompt(phrase)
        output_text = await self.client.generate(images, text, max_tokens=256)
        return _extract_grounding_bbox(output_text, images[0].size)

    def ground(self, image, phrase):
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self.ground_async(image, phrase)
        )
