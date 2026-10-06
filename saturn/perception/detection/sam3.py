"""SAM3 open-vocabulary detection + segmentation (phrase -> boxes, masks, scores).

Model-side; imported only by pipeline.models and serving.
"""
import torch
import numpy as np


from PIL import Image

# Local SAM3 (image) API
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor as LocalSam3Processor

from saturn.perception.detection.base import apply_nms


class SAM3:
    """
    SAM3 (Segment Anything Model 3) wrapper for object detection via text prompts.
    Uses instance segmentation and extracts bounding boxes from the predicted masks.

    API:
      - .predict(image, text, apply_custom_nms=False, extra_bboxes=None)
          → torch.Tensor of shape (N,4): [x0,y0,x1,y1]
      - .predict_for_comparison(image, text)
          → torch.Tensor of raw boxes, no sorting/NMS/extra steps
      - .mark_objects, .predict_and_mark, .predict_mark_and_show
    """

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        gpu_number: int = 0,
    ):
        # Use local SAM3 image model + processor (set_image → set_text_prompt),
        # in fp32: LocalSam3Processor.set_image() builds fp32 input tensors
        # internally, so a bf16/fp16 model raises "Input type ... and weight
        # type ... should be the same" on every forward pass.
        self.sam3_model = build_sam3_image_model()
        self.processor = LocalSam3Processor(self.sam3_model)
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.nms_threshold = nms_threshold

    def _set_threshold(self, threshold: float | None = None) -> None:
        """Set the processor's confidence threshold for the next grounding call.

        ``Sam3Processor.confidence_threshold`` is shared instance state, so
        every inference method sets it explicitly (``self.threshold`` unless a
        per-call value is given); otherwise a replica would filter with
        whatever threshold its previous request left behind.
        """
        self.processor.confidence_threshold = (
            self.threshold if threshold is None else threshold
        )

    def predict(
        self,
        image: Image.Image | torch.Tensor,
        text: str = "",
        apply_custom_nms: bool = False,
        extra_bboxes: list[list[float]] = None,
    ) -> torch.Tensor:
        """
        Runs SAM3 instance segmentation with text prompt and returns bounding boxes.

        Args:
            image: Input image (PIL Image or torch.Tensor)
            text: Text prompt describing objects to segment
            apply_custom_nms: Whether to apply NMS to filter overlapping boxes
            extra_bboxes: Additional bounding boxes to include in results

        Returns:
            torch.Tensor: Bounding boxes in [x0, y0, x1, y1] format
        """
        # — prepare PIL image —
        if isinstance(image, torch.Tensor):
            arr = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(arr)
        else:
            pil = image.convert("RGB")

        # — run SAM3 inference (local API) —
        inference_state = self.processor.set_image(pil)
        self._set_threshold()
        output = self.processor.set_text_prompt(state=inference_state, prompt=text)
        results = output

        # — extract boxes and scores —
        boxes_list = []
        scores_list = []

        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        # Normalize to lists
        if isinstance(boxes, torch.Tensor):
            boxes_list = boxes.detach().cpu().tolist()
        else:
            boxes_list = list(boxes) if boxes is not None else []
        if isinstance(scores, torch.Tensor):
            scores_list = scores.detach().cpu().tolist()
        else:
            scores_list = (
                list(scores) if scores is not None else [1.0] * len(boxes_list)
            )

        # — append extra forced boxes —
        if extra_bboxes:
            boxes_list += [list(b) for b in extra_bboxes]
            scores_list += [0.95] * len(extra_bboxes)

        # — apply custom NMS if requested —
        if apply_custom_nms and boxes_list:
            keep = apply_nms(boxes_list, scores_list, self.nms_threshold)
            keep = sorted(keep, key=lambda i: scores_list[i], reverse=True)
            boxes_list = [boxes_list[i] for i in keep]
            scores_list = [scores_list[i] for i in keep]
        else:
            # sort by score descending
            if boxes_list:
                idxs = sorted(
                    range(len(scores_list)), key=lambda i: scores_list[i], reverse=True
                )
                boxes_list = [boxes_list[i] for i in idxs]

        if not boxes_list:
            return torch.empty((0, 4), dtype=torch.float32)

        return torch.tensor(boxes_list, dtype=torch.float32)


    def predict_with_masks(
        self, image: Image.Image | torch.Tensor, text: str = "", threshold: float = None
    ) -> dict:
        """
        Returns full SAM3 results including masks, boxes, and scores.

        Args:
            image: Input image
            text: Text prompt
            threshold: Confidence threshold for filtering detections
                (``None`` uses ``self.threshold``)
        Returns:
            dict with keys: 'masks', 'boxes', 'scores'
        """
        # — prepare PIL image —
        if isinstance(image, torch.Tensor):
            arr = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(arr)
        else:
            pil = image.convert("RGB")

        # — run SAM3 inference (local API) —
        inference_state = self.processor.set_image(pil)
        self._set_threshold(threshold)
        output = self.processor.set_text_prompt(state=inference_state, prompt=text)
        return {
            "masks": output.get("masks", []),
            "boxes": output.get("boxes", []),
            "scores": output.get("scores", []),
        }

    def mask_from_boxes(
        self,
        image: Image.Image | torch.Tensor,
        boxes,
    ) -> list:
        """Produce SAM3 masks from pre-specified bounding boxes.

        Use this as the second stage when the first-stage (text prompt)
        detection misses the object but an external source (e.g. a VLM)
        has provided a bounding box for it. The returned masks match the
        precision of text-prompted SAM3 masks, so downstream depth
        back-projection and region logic behave identically.

        Parameters
        ----------
        image : PIL.Image.Image or torch.Tensor
            The image to segment.
        boxes : list | np.ndarray | torch.Tensor
            One or more [x1, y1, x2, y2] boxes in pixel coordinates of the
            original image. Accepts a single 4-length sequence or a 2-D
            array of shape (N, 4).

        Returns
        -------
        list
            List of boolean masks aligned with ``boxes``. Each mask is a
            numpy ``bool`` array of shape ``(H, W)`` matching the input
            image's size. Returns ``[]`` when SAM3 produces no output.
        """
        # --- prepare PIL image ---
        if isinstance(image, torch.Tensor):
            arr = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(arr)
        else:
            pil = image.convert("RGB")

        # --- normalize boxes to tensor of shape (N, 4) ---
        if isinstance(boxes, torch.Tensor):
            boxes_tensor = boxes.float()
        else:
            boxes_arr = np.asarray(boxes, dtype=np.float32)
            if boxes_arr.ndim == 1:
                boxes_arr = boxes_arr.reshape(1, 4)
            boxes_tensor = torch.from_numpy(boxes_arr).float()

        if boxes_tensor.numel() == 0:
            return []

        # --- run SAM3 with one normalized box prompt at a time ---
        img_w, img_h = pil.size
        results = []
        for box in boxes_tensor:
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            cxcywh = [
                ((x1 + x2) * 0.5) / max(img_w, 1),
                ((y1 + y2) * 0.5) / max(img_h, 1),
                max(x2 - x1, 0.0) / max(img_w, 1),
                max(y2 - y1, 0.0) / max(img_h, 1),
            ]

            inference_state = self.processor.set_image(pil)
            self._set_threshold()
            output = self.processor.add_geometric_prompt(
                box=cxcywh, label=True, state=inference_state
            )
            raw_masks = output.get("masks", [])
            scores = output.get("scores")

            if len(raw_masks) == 0:
                m = np.zeros((img_h, img_w), dtype=bool)
                ix1 = max(0, min(img_w, int(round(x1))))
                iy1 = max(0, min(img_h, int(round(y1))))
                ix2 = max(0, min(img_w, int(round(x2))))
                iy2 = max(0, min(img_h, int(round(y2))))
                m[iy1:iy2, ix1:ix2] = True
                results.append(m)
                continue

            if hasattr(raw_masks, "cpu"):
                raw_masks = raw_masks.detach().cpu().numpy()
            raw_masks = np.asarray(raw_masks)

            best = 0
            if scores is not None:
                if hasattr(scores, "cpu"):
                    scores = scores.detach().cpu().numpy()
                scores = np.asarray(scores)
                if scores.size > 1:
                    best = int(np.argmax(scores))

            m = raw_masks[best]
            while m.ndim > 2:
                m = m[0]
            results.append(np.asarray(m).astype(bool))
        return results


def object_detector(model="sam3", threshold=0.2):
    if model in ["sam3", "SAM3"]:
        model = SAM3(
            model_name="facebook/sam3",
            threshold=threshold,
            mask_threshold=0.5,
            nms_threshold=0.6,
        )
    else:
        raise ValueError(f"Unknown model: {model}. Supported model: 'sam3'.")
    return model
