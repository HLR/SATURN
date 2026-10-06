"""Image annotation helpers (set-of-mark markers, boxes, mask overlays) for VLM prompting.

Pure PIL/numpy drawing; must not import saturn.vlm/serving.
"""
from copy import copy

import numpy as np
import torch
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageFont

from saturn.log import get_logger

log = get_logger(__name__)

# Local SAM3 imports
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor as LocalSam3Processor
except ImportError:
    build_sam3_image_model = None
    LocalSam3Processor = None
    log.warning("Local sam3 not installed. Falling back to HF transformers.")


class MaskerSAM3:
    """Masker class using local SAM3 image model for mask generation."""

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        device: str = "cuda",
        threshold: float = 0.3,
        mask_threshold: float = 0.5,
    ):
        if build_sam3_image_model is None or LocalSam3Processor is None:
            # Fall back to HF transformers if local sam3 not available
            from transformers import Sam3Model, Sam3Processor

            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            self.model = Sam3Model.from_pretrained(model_name).to(self.device).eval()
            self.processor = Sam3Processor.from_pretrained(model_name)
            self.use_local_sam3 = False
        else:
            self.model = build_sam3_image_model()
            self.processor = LocalSam3Processor(self.model)
            self.use_local_sam3 = True

        self.threshold = threshold
        self.mask_threshold = mask_threshold

    def mask_image(self, image, input_boxes: np.ndarray) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image).astype(np.uint8))

        if self.use_local_sam3:
            # Local SAM3 API: one positive box prompt per input box (normalised
            # cxcywh), keeping the highest-scoring mask so masks[i] pairs with
            # input_boxes[i]. A box SAM3 returns nothing for falls back to its
            # rectangle.
            image = image.convert("RGB")
            img_w, img_h = image.size
            self.processor.confidence_threshold = self.threshold
            masks = []
            for x1, y1, x2, y2 in np.asarray(input_boxes, dtype=np.float32).reshape(-1, 4):
                box = [
                    (x1 + x2) / 2 / img_w,
                    (y1 + y2) / 2 / img_h,
                    max(x2 - x1, 0.0) / img_w,
                    max(y2 - y1, 0.0) / img_h,
                ]
                state = self.processor.set_image(image)
                output = self.processor.add_geometric_prompt(
                    box=[float(v) for v in box], label=True, state=state
                )
                raw_masks = output.get("masks", [])
                if len(raw_masks) == 0:
                    m = torch.zeros((img_h, img_w), dtype=torch.bool)
                    iy1, iy2 = (max(0, min(img_h, int(round(v)))) for v in (y1, y2))
                    ix1, ix2 = (max(0, min(img_w, int(round(v)))) for v in (x1, x2))
                    m[iy1:iy2, ix1:ix2] = True
                else:
                    scores = torch.as_tensor(output.get("scores", [0.0])).reshape(-1)
                    m = torch.as_tensor(raw_masks)[int(torch.argmax(scores))]
                    m = m.reshape(m.shape[-2:]).cpu().bool()
                masks.append(m)
            if not masks:
                return torch.zeros((0, img_h, img_w), dtype=torch.bool)
            return torch.stack(masks)
        else:
            # HF transformers API
            if not isinstance(input_boxes, torch.Tensor):
                boxes_tensor = torch.tensor(input_boxes, dtype=torch.float32)
            else:
                boxes_tensor = input_boxes.float()

            inputs = self.processor(
                images=image,
                input_boxes=[boxes_tensor],
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model(**inputs)

            processed = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            return processed.get("masks")


def _expand_bbox(bbox, expansion_ratio, min_margin):
    """
    Expands a bounding box by a given ratio and minimum margin.

    Args:
        bbox (list): Bounding box in [x0, y0, x1, y1] format.
        expansion_ratio (float): Ratio to expand the bbox by.
        min_margin (int): Minimum margin to add to each side of the bbox.

    Returns:
        list: Expanded bounding box.
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    margin_width = max(
        int(width * expansion_ratio / 2), min_margin
    )
    margin_height = max(
        int(height * expansion_ratio / 2), min_margin
    )
    return [
        bbox[0] - margin_width,
        bbox[1] - margin_height,
        bbox[2] + margin_width,
        bbox[3] + margin_height,
    ]


def _apply_mask_outside(image, masks_or_bboxes, use_mask, expansion_ratio, min_margin):
    """
    Applies a mask to the image to mask out the regions *outside* the provided masks or bounding boxes.

    Args:
        image (PIL.Image.Image): Input PIL image.
        masks_or_bboxes (torch.Tensor or numpy.ndarray or list): Masks (torch.Tensor/numpy.ndarray) or bboxes (list of lists) defining object regions.
        use_mask (bool): If True, use masks; if False, use bboxes.
        expansion_ratio (float): Ratio to expand bboxes before masking.
        min_margin (int): Minimum margin for bbox expansion.

    Returns:
        PIL.Image.Image: Image with background masked out (filled black).
    """
    mask_img = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask_img)

    if use_mask:
        combined_mask = Image.new("L", image.size, 0)
        for mask in masks_or_bboxes:
            mask_np = _normalize_mask(mask)
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).convert(
                "L"
            )
            combined_mask = Image.composite(
                Image.new("L", image.size, 255), combined_mask, mask_pil
            )
        mask_img = combined_mask
    else:
        expanded_bboxes = [
            _expand_bbox(bbox, expansion_ratio, min_margin) for bbox in masks_or_bboxes
        ]
        for expanded_bbox in expanded_bboxes:
            mask_draw.rectangle(
                expanded_bbox, fill=255
            )

    if image.mode != "RGBA":
        image = image.convert("RGBA")
    background = Image.new(
        "RGBA", image.size, (0, 0, 0, 255)
    )
    return Image.composite(
        image, background, mask_img
    )


def _brighten_color(rgb, boost=0.6):
    boost = max(0.0, min(1.0, float(boost)))
    return tuple(int(c + (255 - c) * boost) for c in rgb)


def _normalize_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.array(mask)

    mask_np = np.squeeze(mask_np)
    if mask_np.ndim == 2:
        return mask_np
    if mask_np.ndim >= 3:
        return mask_np[0]
    return mask_np


def _draw_masks_overlay(
    image,
    masks,
    colors,
    alpha=0.1,
    draw_edge=False,
    edge_thickness=3,
    edge_color=None,
    edge_boost=0.6,
    edge_alpha=0.35,
):
    """
    Overlays semi-transparent colored masks on the image.

    Args:
        image (PIL.Image.Image): Input PIL image.
        masks (torch.Tensor or numpy.ndarray or list): Masks to overlay.
        colors (list): List of color specs, either:
                       - CSS-style strings ("red", "#FF00FF", "rgb(255,0,0)") or
                       - 3-tuples of ints (R, G, B)
        alpha (float): Transparency between 0 (invisible) and 1 (opaque).

    Returns:
        PIL.Image.Image: Image with masks overlaid.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    for i, mask in enumerate(masks):
        rgb = (
            ImageColor.getrgb(colors[i % len(colors)])
            if isinstance(colors[i % len(colors)], str)
            else tuple(colors[i % len(colors)])
        )
        m = _normalize_mask(mask)

        if alpha > 0:
            overlay = Image.new("RGBA", image.size, rgb + (0,))
            mask_pil = Image.fromarray((m * alpha * 255).astype(np.uint8)).convert("L")
            overlay.putalpha(mask_pil)
            image = Image.alpha_composite(image, overlay)

        if draw_edge:
            thickness = max(1, int(edge_thickness))
            mask_full = Image.fromarray((m * 255).astype(np.uint8)).convert("L")
            dil = mask_full.filter(ImageFilter.MaxFilter(size=2 * thickness + 1))
            ero = mask_full.filter(ImageFilter.MinFilter(size=2 * thickness + 1))
            outline = ImageChops.difference(dil, ero)

            if edge_color is not None:
                edge_rgb = ImageColor.getrgb(edge_color)
            else:
                edge_rgb = _brighten_color(rgb, boost=edge_boost)
            edge_alpha = max(0.0, min(1.0, float(edge_alpha)))
            if edge_alpha < 1.0:
                outline = outline.point(lambda p: int(p * edge_alpha))
            edge_ovl = Image.new("RGBA", image.size, edge_rgb + (255,))
            edge_ovl.putalpha(outline)
            image = Image.alpha_composite(image, edge_ovl)

    return image.convert("RGB")


def _draw_black_masks_overlay(image, masks):
    """
    Overlays black masks on the image.

    Args:
        image (PIL.Image.Image): Input PIL image.
        masks (torch.Tensor or numpy.ndarray or list): Masks to overlay.

    Returns:
        PIL.Image.Image: Image with masks overlaid.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    for i, mask in enumerate(masks):
        mask_image_np = _normalize_mask(mask)
        mask_image_pil = Image.fromarray(
            (mask_image_np * 255).astype(np.uint8)
        ).convert("L")
        color_overlay = Image.new(
            "RGBA", image.size, (0, 0, 0, 255)
        )
        image = Image.composite(
            color_overlay, image, mask_image_pil
        )
    return image


def _draw_bboxes_and_text(
    image,
    bboxes,
    draw_labels,
    font,
    box_width_ratio,
    chosen_colors,
    expansion_ratio,
    min_margin,
    small_area_threshold_ratio=0.05,
    small_box_line_width_coeff=0.5,
    extra_thick=False,
):
    """
    Draws bounding boxes and optional labels on the image.

    Args:
        image (PIL.Image.Image): Input PIL image.
        bboxes (list): List of bounding boxes in [x0, y0, x1, y1] format.
        draw_labels (bool): Whether to draw index labels inside the bboxes.
        font (PIL.ImageFont.FreeTypeFont): Font for labels.
        box_width_ratio (float): Ratio of bbox line width to image width.
        chosen_colors (list): List of colors to use for bboxes and labels.
        expansion_ratio (float): Ratio to expand bboxes before drawing.
        min_margin (int): Minimum margin for bbox expansion.
    """
    img_width, img_height = image.size
    scaled_font = _get_image_scaled_font(image.size)
    image_area = img_width * img_height
    small_area_threshold = (
        image_area * small_area_threshold_ratio
    )
    draw = ImageDraw.Draw(image)
    expanded_bboxes = [
        _expand_bbox(bbox, expansion_ratio, min_margin) for bbox in bboxes
    ]

    for i, expanded_bbox in enumerate(
        expanded_bboxes
    ):
        color = chosen_colors[i % len(chosen_colors)]
        base_line_width = int(max(image.size) * box_width_ratio)
        original_bbox = bboxes[i]
        bbox_width = original_bbox[2] - original_bbox[0]
        bbox_height = original_bbox[3] - original_bbox[1]
        bbox_area = bbox_width * bbox_height

        # Adjust line width if bbox area is small
        line_width = base_line_width
        if bbox_area > 0 and bbox_area < small_area_threshold:
            line_width = int(base_line_width * small_box_line_width_coeff)
        if extra_thick:
            line_width = max(2, int(line_width * 1.5))
        else:
            line_width = max(1, line_width)

        draw.rectangle(expanded_bbox, outline=color, width=line_width)

        if (
            draw_labels
        ):
            text = str(i)
            text_bbox = draw.textbbox(
                (expanded_bbox[0], expanded_bbox[1]), text, font=scaled_font
            )
            draw.rectangle([*text_bbox], fill=color)
            draw.text(
                (text_bbox[0], text_bbox[1]), text, fill="white", font=scaled_font
            )


def _get_image_scaled_font(
    image_size,
    base_font_size=30,
    reference_image_min_side=1080,
    min_font_size=12,
    max_font_size=96,
):
    """Returns a font with size scaled to the image dimensions."""
    img_width, img_height = image_size
    min_side = max(1, min(img_width, img_height))
    scaled_size = int(base_font_size * (min_side / reference_image_min_side))
    scaled_size = max(min_font_size, min(max_font_size, scaled_size))

    try:
        return ImageFont.truetype("DejaVuSans.ttf", scaled_size)
    except OSError:
        return ImageFont.load_default()


def _draw_black_bbox(image, bboxes):
    """
    Draws black bounding boxes on the image.

    Args:
        image (PIL.Image.Image): Input PIL image.
        bboxes (list): List of bounding boxes in [x0, y0, x1, y1] format.

    Returns:
        PIL.Image.Image: Image with black bounding boxes drawn.
    """
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        # Convert tensor or numpy array to list of ints for PIL
        if isinstance(bbox, torch.Tensor):
            coords = bbox.cpu().tolist()
        elif isinstance(bbox, np.ndarray):
            coords = bbox.tolist()
        else:
            coords = list(bbox)
        draw.rectangle(
            coords, outline="black", width=3, fill="black"
        )
    return image


class MarkerV2:
    def __init__(self):
        self.font = ImageFont.load_default()
        self.colors = ["red", "green", "blue", "#4B0082", "orange", "pink"]

        # Lazy-loaded SAM3-based masker (heavy GPU model — only load when needed)
        self._masker = None

        self.box_width_ratio = 0.01

    @property
    def masker(self):
        """Lazy-loaded MaskerSAM3 — only instantiated on first use."""
        if self._masker is None:
            self._masker = MaskerSAM3()
        return self._masker

    def mark_objects(
        self,
        image,
        bboxes=None,
        masks=None,
        mask_background=False,
        draw_bbox=False,
        overlay_masks=False,
        draw_text=False,
        mask_regions=False,
        mask_bbox=False,
        extra_thick=False,
        crop=False,
        draw_edge=False,
        edge_thickness=1.5,
        edge_color=None,
        edge_alpha=0.35,
    ):
        image = copy(image)  # Create a copy to avoid modifying original image

        min_margin_val = 0
        expansion_ratio_val = 0.0
        if masks is True:  # Generate masks from bboxes
            masks = self.masker.mask_image(image, bboxes)
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().detach()
            if isinstance(masks, np.ndarray):
                masks = torch.tensor(masks).cpu().detach()
            if hasattr(masks, "shape") and len(masks.shape) == 3:
                masks = masks.unsqueeze(0)

            use_mask = True
        elif masks is not None and masks is not False:
            use_mask = True
        else:
            use_mask = False

        if use_mask:
            if mask_background:
                image = _apply_mask_outside(
                    image,
                    masks,
                    use_mask=use_mask,
                    expansion_ratio=expansion_ratio_val,
                    min_margin=min_margin_val,
                )
            if overlay_masks or draw_edge:
                image = _draw_masks_overlay(
                    image,
                    masks,
                    self.colors,
                    alpha=0.1 if overlay_masks else 0.0,
                    draw_edge=draw_edge,
                    edge_thickness=edge_thickness,
                    edge_color=edge_color,
                    edge_alpha=edge_alpha,
                )
            if mask_regions:
                image = _draw_black_masks_overlay(image, masks)
            if mask_bbox:
                image = _draw_black_bbox(image, bboxes)
        elif (
            bboxes is not None
        ):  # No masks, but bboxes are provided, use bbox-based operations
            use_mask = False
            if mask_background:
                image = _apply_mask_outside(
                    image,
                    bboxes,
                    use_mask=use_mask,
                    expansion_ratio=expansion_ratio_val,
                    min_margin=min_margin_val,
                )

        if bboxes is not None and draw_bbox:
            _draw_bboxes_and_text(
                image,
                bboxes,
                draw_labels=draw_text,
                font=self.font,
                box_width_ratio=self.box_width_ratio,
                chosen_colors=self.colors,
                expansion_ratio=expansion_ratio_val,
                min_margin=min_margin_val,
                extra_thick=extra_thick,
            )

        return image.convert("RGB")
