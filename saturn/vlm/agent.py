"""Agent: the VLM scoring client used by ``score(...)`` (batched, cached, per-task traced).

Model-side; imported only by pipeline.models, saturn.vlm and serving.
"""
import contextvars
import itertools
import math
from collections import defaultdict

import numpy as np
import PIL
import torch
from PIL import Image, ImageDraw

from saturn.perception.drawing import MarkerV2
from saturn.log import get_logger

log = get_logger(__name__)


class Assets:
    def __init__(self):
        self._marker = None

    @property
    def marker(self):
        if self._marker is None:
            self._marker = MarkerV2()
        return self._marker


    def extract_object_with_mask(
        self, image_pil: Image.Image, mask_input, bbox=None, bbox_color=None
    ) -> Image.Image:
        """
        Extracts an object from a PIL image using a mask.

        The resulting image will have the object pixels from the original image
        and a transparent background elsewhere.

        Args:
            image_pil: The input PIL Image (should ideally be RGB or RGBA).
            mask_input: The mask defining the object. Can be:
                - A PIL Image (mode 'L' or '1'). Assumed white object, black background.
                - A NumPy array (2D HxW, boolean or integer 0/1).
                - A PyTorch Tensor (2D HxW, boolean or numeric 0/1).

        Returns:
            A new PIL Image (RGBA) with the extracted object on a transparent background.
        """
        # 1. Convert input PIL image to NumPy array
        # Ensure image is RGBA for transparency handling
        image_rgba = image_pil.convert("RGBA")
        image_np = np.array(image_rgba)
        H, W, _ = image_np.shape

        # 2. Process the mask_input into a 2D boolean NumPy array
        mask_np = None
        if isinstance(mask_input, Image.Image):
            mask_pil = mask_input.convert("L")  # Convert to grayscale
            if mask_pil.size != (W, H):
                mask_pil = mask_pil.resize((W, H), Image.NEAREST)  # Ensure same size
            mask_np = np.array(mask_pil)
            mask_bool = mask_np > 128  # Threshold grayscale mask (adjust if needed)
        elif isinstance(mask_input, np.ndarray):
            if mask_input.ndim == 3:  # Handle masks like (H, W, 1)
                mask_input = mask_input.squeeze()
            if mask_input.shape != (H, W):
                # Basic resize if shape mismatch, might need interpolation adjustment
                mask_img = Image.fromarray(
                    mask_input.astype(np.uint8) * 255
                )  # Convert to PIL for resize
                mask_img = mask_img.resize((W, H), Image.NEAREST)
                mask_np = np.array(mask_img)
            else:
                mask_np = mask_input
            # Convert to boolean (assuming non-zero means True)
            mask_bool = mask_np.astype(bool)
        elif "torch" in globals() and isinstance(mask_input, torch.Tensor):
            mask_tensor = mask_input.cpu().detach()
            if mask_tensor.ndim == 3:  # Handle masks like (1, H, W) or (H, W, 1)
                mask_tensor = mask_tensor.squeeze()
            mask_np = mask_tensor.numpy()
            if mask_np.shape != (H, W):
                # Basic resize
                mask_img = Image.fromarray(
                    mask_np.astype(np.uint8) * 255
                )  # Convert to PIL for resize
                mask_img = mask_img.resize((W, H), Image.NEAREST)
                mask_np = np.array(mask_img)
            # Convert to boolean (handle probabilities > 0.5 or integer masks > 0)
            if np.issubdtype(mask_np.dtype, np.floating):
                mask_bool = mask_np > 0.5
            else:
                mask_bool = mask_np > 0
        else:
            raise TypeError(
                "Unsupported mask type. Provide PIL Image, NumPy array, or PyTorch Tensor."
            )

        if mask_bool.shape != (H, W):
            raise ValueError(
                f"Mask shape {mask_bool.shape} doesn't match image shape {(H, W)}"
            )

        # 3. Create a transparent background canvas
        output_np = np.zeros_like(image_np)  # Shape (H, W, 4), filled with 0s

        # 4. Copy original pixels where mask is True
        output_np[mask_bool] = image_np[mask_bool]

        # 5. Alpha is already 255 under the mask and 0 elsewhere: image_np is
        # RGBA and output_np was initialized to zeros.

        # 6. Convert the result back to a PIL Image
        extracted_image_pil = Image.fromarray(output_np, "RGBA")
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # Draw the bounding box on the extracted image
            draw = ImageDraw.Draw(extracted_image_pil)
            draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=3)
        return extracted_image_pil


class Agent(Assets):
    def __init__(self, wrapper=None):
        super().__init__()
        self.wrapper = wrapper
        # Per-sample cache of marked images and VLM scores, held in a ContextVar
        # so concurrent execute phases (asyncio.to_thread copies the task
        # context) never see each other's entries. The keys carry object COUNT,
        # not scene identity, so a shared dict would let one sample score
        # another scene's images whenever counts coincide.
        self._cache_var: contextvars.ContextVar = contextvars.ContextVar(
            f"agent_cache_{id(self)}", default=None
        )
        self.clean_cache()

    def _expand_bbox_min_size(self, bbox, image_width, image_height, min_size=28):
        x1, y1, x2, y2 = map(float, bbox)

        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Ensure at least min_size around center
        new_w = max(w, float(min_size))
        new_h = max(h, float(min_size))

        nx1 = cx - new_w / 2.0
        nx2 = cx + new_w / 2.0
        ny1 = cy - new_h / 2.0
        ny2 = cy + new_h / 2.0

        # Round (no shrink)
        nx1 = math.floor(nx1)
        ny1 = math.floor(ny1)
        nx2 = math.ceil(nx2)
        ny2 = math.ceil(ny2)

        # Clamp by sliding (keep size)
        bw = nx2 - nx1
        bh = ny2 - ny1

        # If image is smaller than min_size, just clamp to image
        bw = min(bw, image_width)
        bh = min(bh, image_height)

        # Slide in X
        if nx1 < 0:
            nx2 -= nx1
            nx1 = 0
        if nx2 > image_width:
            nx1 -= nx2 - image_width
            nx2 = image_width
        nx1 = max(0, nx1)
        nx2 = min(image_width, nx1 + bw)

        # Slide in Y
        if ny1 < 0:
            ny2 -= ny1
            ny1 = 0
        if ny2 > image_height:
            ny1 -= ny2 - image_height
            ny2 = image_height
        ny1 = max(0, ny1)
        ny2 = min(image_height, ny1 + bh)

        # Final ints
        return int(nx1), int(ny1), int(nx2), int(ny2)

    def score(
        self,
        image,
        bboxes,
        question,
        num_objects=1,
        masks=None,
        candidates=None,
        type=None,
        history=None,
        all_view_images=None,
        view_index=None,
        # extra keyword arguments are accepted and ignored
        **kwargs,
    ):
        question += "\nAnswer the question using a single word or phrase."
        original_question = question

        all_combinations = list(itertools.product(*[range(len(bboxes))] * num_objects))
        probabilities = []

        for comb_index, combination in enumerate(all_combinations):
            items = tuple([bboxes[x] for x in combination])
            mask_items = tuple([masks[x] for x in combination]) if masks else None
            if len(set(combination)) != num_objects and num_objects > 0:
                probability = torch.zeros(1).to(self.device).squeeze(0)
                probability.requires_grad = True
            else:
                probability = self._score(
                    image,
                    question,
                    items,
                    masks=mask_items,
                    candidates=candidates,
                    type=type,
                    history=history,
                    all_view_images=all_view_images,
                    view_index=view_index,
                )
            probabilities.append(probability)

        if isinstance(probabilities, list):
            probabilities = torch.stack(probabilities)

        if num_objects > 0:
            shape = int(math.ceil(probabilities.shape[0] ** (1 / num_objects)))
            probabilities = probabilities.reshape(*([shape] * num_objects))
        else:
            probabilities = probabilities.squeeze(0)

        if self.wrapper is not None:
            if num_objects == 0:
                return self.wrapper(
                    probabilities, extra_info=original_question, vars=[]
                )
            else:
                return self.wrapper(probabilities, extra_info=original_question)
        return probabilities

    @property
    def cache(self):
        c = self._cache_var.get()
        if c is None:
            c = self._fresh_cache()
            self._cache_var.set(c)
        return c

    @staticmethod
    def _fresh_cache():
        return {"images": defaultdict(lambda: list()), "scores": defaultdict(lambda: list())}

    def clean_cache(self):
        self._cache_var.set(self._fresh_cache())

    def query(
        self,
        image,
        boxes,
        bbox_id,
        text,
        mask=False,
        draw_bbox=False,
        pass_og_image=False,
        overlay_masks=False,
        type="default",
        crop=True,
    ):
        new_query = text + "\nAnswer the question using a single word or phrase."

        if bbox_id is not None:
            box = [boxes[bbox_id]]
            marked_image = self.marker.mark_objects(
                image,
                box,
                draw_bbox=draw_bbox,
                overlay_masks=overlay_masks,
                masks=mask,
            )
            if crop is True and type != "context":
                x1, y1, x2, y2 = self._expand_bbox_min_size(
                    box[0], marked_image.width, marked_image.height, min_size=28
                )
                marked_image = marked_image.crop((x1, y1, x2, y2))
            if pass_og_image:
                new_query = (
                    "These are two similar images, second one is marked with bounding boxes"
                    + new_query
                )
                marked_image = [image, marked_image]
            else:
                marked_image = [marked_image]

            answer = self._query(marked_image, new_query)
        else:
            answer = self._query(image, text)
        return answer

    def _score(
        self,
        image: "PIL.Image",
        question: str,
        bboxes: list,
        masks: list = None,
        candidates: list = None,
        type: str = "default",
        history=None,
        target_tokens: list = None,
        all_view_images: list = None,
        view_index: int = None,
    ):
        """Score bboxes against a yes/no question. Must be overridden by subclass."""
        raise NotImplementedError

    def _query(self, image, text, max_new_tokens=1):
        raise NotImplementedError


    def ground(self, image, phrase):
        raise NotImplementedError


    # ==================== Multi-View Score / Query ====================

    def score_multiview(
        self,
        question,
        num_objects=1,
        type=None,
        scene=None,
        images=None,
        cam_id=None,
        **kwargs,
    ):
        """Multi-view score: evaluate a VLM question across views, take max.

        For each object (or object pair), this method evaluates the VLM
        question in every view where the object(s) are visible and returns
        the maximum score across views.

        Parameters
        ----------
        question : str
            VLM question (uses red/green bounding box markers).
        num_objects : int
            1 or 2. Number of objects per evaluation.
        type : str, optional
            "class", "property", "relation", "context".
        scene : Scene
            Multi-view scene (from ``load_scene_async``).
        images : list[Image], optional
            Explicit image list. Falls back to ``scene.images``.
        cam_id : int, optional
            If provided, only score objects in this camera/view index.
            Objects not visible in this view get score 0.
        **kwargs
            Forwarded to the underlying ``self.score()`` call, which passes
            ``candidates`` and ``history`` on to ``_score`` and ignores the
            rest; the batched path passes the same two.

        Returns
        -------
        ProbabilisticTensor or torch.Tensor
            Shape ``(N,)`` for ``num_objects=1``, ``(N, N)`` for
            ``num_objects=2``.
        """
        if scene is None:
            raise ValueError("score_multiview requires a scene argument.")
        if num_objects not in (1, 2):
            raise ValueError(
                f"score_multiview supports num_objects=1 or 2, got {num_objects}"
            )
        type = type or "default"

        view_images = images if images is not None else scene.images
        N = len(scene.objects)
        # Number of cameras — appended to entity space so spatial tensors
        # (left/right/front/behind, distance, ...) can refer to cameras as
        # entities.  Object-class scores must therefore return 0 for camera
        # slots so queries like ``camera("x1") & view.behind("x1", "x2") &
        # car("x2")`` only fire when x2 is actually an object.
        C = len(scene.cameras)
        total = N + C

        # No objects: an all-zero tensor, still padded for cameras so shapes
        # line up with directional tensors.
        if N == 0:
            if num_objects == 1:
                empty = torch.zeros(total, device=self.device)
            else:
                empty = torch.zeros(total, total, device=self.device)
            return self._wrap_multiview(empty, question)

        if num_objects == 1:
            scores_per_object = self._best_object_scores(
                question, type, scene, view_images, cam_id, kwargs,
            )
            probabilities = torch.stack(scores_per_object)
            # Pad to (N+C,) — cameras get score 0 for object-class queries.
            if C > 0:
                pad = torch.zeros(C, device=self.device)
                probabilities = torch.cat([probabilities, pad], dim=0)
            return self._wrap_multiview(probabilities, question)

        scores_matrix = self._best_pair_scores(
            question, type, scene, view_images, cam_id, kwargs,
        )
        # Pad to (N+C, N+C) — entries involving any camera slot are 0
        # for object-class queries.
        if C > 0:
            padded = torch.zeros(total, total, device=self.device)
            padded[:N, :N] = scores_matrix
            scores_matrix = padded
        return self._wrap_multiview(scores_matrix, question)

    def _wrap_multiview(self, scores, question):
        if self.wrapper is not None:
            return self.wrapper(scores, extra_info=question)
        return scores

    def _zero_score(self, requires_grad=False):
        zero = torch.zeros(1, device=self.device).squeeze(0)
        if requires_grad:
            zero.requires_grad = True
        return zero

    @staticmethod
    def _score_value(result, entry=None):
        """Detached 0-d tensor from a score result (``entry`` indexes a matrix)."""
        if hasattr(result, "tensor"):
            val = result.tensor if entry is None else result.tensor[entry]
            val = val.detach()
        elif isinstance(result, torch.Tensor):
            val = (result if entry is None else result[entry]).detach()
        else:
            val = torch.tensor(float(result))
        return val.squeeze()

    @staticmethod
    def _object_views(obj, cam_id):
        """Views that show *obj*, restricted to ``cam_id`` when it is given."""
        visible_views = obj.views
        if cam_id is not None:
            visible_views = [v for v in visible_views if v == cam_id]
        return visible_views

    @staticmethod
    def _object_view_inputs(obj, v, view_images):
        """``(image, bbox, masks)`` for *obj* in view *v*, or None if it is not boxed there."""
        if v >= len(view_images):
            return None
        img = view_images[v]
        bbox = obj.per_view_bboxes.get(v)
        if bbox is None:
            return None
        mask_v = obj.per_view_masks.get(v)
        masks_arg = [mask_v] if mask_v is not None else None
        return img, bbox, masks_arg

    def _best_object_scores(self, question, type, scene, view_images, cam_id, kwargs):
        """Per object, the maximum single-object score over the views that show it.

        Builds the full (object, view) job list first, then dispatches it in
        one batch through ``_score_many`` when available, so the fan-out
        saturates the vLLM replicas instead of awaiting one call at a time.
        """
        score_jobs = []  # list of (obj_idx, kwargs-for-_score_async)
        for obj_idx in range(len(scene.objects)):
            obj = scene.objects[obj_idx]
            for v in self._object_views(obj, cam_id):
                inputs = self._object_view_inputs(obj, v, view_images)
                if inputs is None:
                    continue
                img, bbox, masks_arg = inputs
                # The arguments score() hands _score on the sequential path.
                score_jobs.append((obj_idx, {
                    "image": img,
                    "question": question + "\nAnswer the question using a single word or phrase.",
                    "bboxes": [bbox],
                    "masks": masks_arg,
                    "candidates": kwargs.get("candidates"),
                    "type": type,
                    "history": kwargs.get("history"),
                    "all_view_images": view_images,
                    "view_index": v,
                }))

        if hasattr(self, "_score_many") and score_jobs:
            return self._best_object_scores_batched(score_jobs, len(scene.objects))
        return self._best_object_scores_sequential(
            question, type, scene, view_images, cam_id, kwargs,
        )

    def _best_object_scores_batched(self, score_jobs, n_objects):
        raw_results = self._score_many([job[1] for job in score_jobs])
        per_obj_best = [None] * n_objects
        for (obj_idx, _), result in zip(score_jobs, raw_results):
            val = self._score_value(result)
            cur = per_obj_best[obj_idx]
            if cur is None or val.item() > cur.item():
                per_obj_best[obj_idx] = val
        return [
            s if s is not None else self._zero_score(requires_grad=True)
            for s in per_obj_best
        ]

    def _best_object_scores_sequential(
        self, question, type, scene, view_images, cam_id, kwargs,
    ):
        scores_per_object = []
        for obj_idx in range(len(scene.objects)):
            obj = scene.objects[obj_idx]
            best_score = None
            for v in self._object_views(obj, cam_id):
                inputs = self._object_view_inputs(obj, v, view_images)
                if inputs is None:
                    continue
                img, bbox, masks_arg = inputs
                result = self.score(
                    img,
                    [bbox],
                    question,
                    num_objects=1,
                    type=type,
                    masks=masks_arg,
                    all_view_images=view_images,
                    view_index=v,
                    **kwargs,
                )
                val = self._score_value(result)
                if best_score is None or val.item() > best_score.item():
                    best_score = val
            if best_score is None:
                best_score = self._zero_score(requires_grad=True)
            scores_per_object.append(best_score)
        return scores_per_object

    def _best_pair_scores(self, question, type, scene, view_images, cam_id, kwargs):
        """(N, N) matrix: per ordered object pair, the maximum score over shared views."""
        N = len(scene.objects)
        scores_matrix = torch.zeros(N, N, device=self.device)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                scores_matrix[i, j] = self._best_pair_score(
                    scene.objects[i], scene.objects[j],
                    question, type, view_images, cam_id, kwargs,
                )
        return scores_matrix

    def _best_pair_score(self, obj_i, obj_j, question, type, view_images, cam_id, kwargs):
        co_views = set(obj_i.views) & set(obj_j.views)
        if cam_id is not None:
            co_views = {v for v in co_views if v == cam_id}
        best_score = None
        for v in co_views:
            if v >= len(view_images):
                continue
            img = view_images[v]
            bbox_i = obj_i.per_view_bboxes.get(v)
            bbox_j = obj_j.per_view_bboxes.get(v)
            if bbox_i is None or bbox_j is None:
                continue
            mask_i = obj_i.per_view_masks.get(v)
            mask_j = obj_j.per_view_masks.get(v)
            if mask_i is not None and mask_j is not None:
                masks_arg = [mask_i, mask_j]
            else:
                masks_arg = None
            result = self.score(
                img,
                [bbox_i, bbox_j],
                question,
                num_objects=2,
                type=type,
                masks=masks_arg,
                all_view_images=view_images,
                view_index=v,
                **kwargs,
            )
            # The (0, 1) entry of the (2, 2) result is "obj at index 0 (red)
            # and obj at index 1 (green)".
            val = self._score_value(result, (0, 1))
            if best_score is None or val.item() > best_score.item():
                best_score = val
        if best_score is None:
            best_score = self._zero_score()
        return best_score

    def query_multiview(
        self,
        question,
        object_id=None,
        type="class",
        scene=None,
        images=None,
        camera_id=None,
        **kwargs,
    ):
        """Multi-view query: pick the best view and ask the VLM.

        View selection:
        1. If ``camera_id`` is specified and the object is visible there,
           use that view.
        2. Otherwise, fall back to the best visible view (highest detection
           confidence or largest bbox area).
        3. If ``object_id`` is ``None``, query the whole scene with every
           view's image.

        Parameters
        ----------
        question : str
            VLM question text.
        object_id : int, optional
            Object index to query about.
        type : str
            Question type.
        scene : Scene
            Multi-view scene.
        images : list[Image], optional
            Explicit image list. Falls back to ``scene.images``.
        camera_id : int, optional
            Preferred view for evaluation.
        **kwargs
            Forwarded to ``self.query()``.

        Returns
        -------
        str
        """
        if scene is None:
            raise ValueError("query_multiview requires a scene argument.")

        view_images = images if images is not None else scene.images

        if object_id is None:
            # Whole-scene query: pass ALL views to the VLM, since questions
            # that compare views need every image.
            if len(view_images) > 1:
                return self.query(
                    list(view_images), [], None, question, type=type, **kwargs
                )
            img = view_images[0]
            return self.query(img, [], None, question, type=type, **kwargs)

        obj = scene.objects[object_id]
        visible_views = obj.views

        # 1. Try preferred camera_id
        if camera_id is not None and camera_id in visible_views:
            bbox = obj.per_view_bboxes.get(camera_id)
            if bbox is not None and camera_id < len(view_images):
                img = view_images[camera_id]
                return self.query(img, [bbox], 0, question, type=type, **kwargs)

        # 2. Fall back to best visible view
        best_view = None
        best_metric = -1.0
        for v in visible_views:
            if v >= len(view_images):
                continue
            bbox = obj.per_view_bboxes.get(v)
            if bbox is None:
                continue
            # Prefer view with highest score; break ties by bbox area
            score_v = obj.per_view_scores.get(v, 0.0)
            if bbox:
                x1, y1, x2, y2 = bbox[:4]
                area = (x2 - x1) * (y2 - y1)
            else:
                area = 0.0
            # Combine: score as primary, area as secondary (normalized)
            metric = score_v + area * 1e-8
            if metric > best_metric:
                best_metric = metric
                best_view = v

        if best_view is not None:
            img = view_images[best_view]
            bbox = obj.per_view_bboxes[best_view]
            return self.query(img, [bbox], 0, question, type=type, **kwargs)

        # No visible view found — try camera 0 with no bbox
        return self.query(view_images[0], [], None, question, type=type, **kwargs)
