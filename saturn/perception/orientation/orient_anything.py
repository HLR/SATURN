"""
Orient Anything V2 Integration for Object Orientation Estimation (VGGT backbone).
"""

from saturn.settings import env
import os
import sys
import random
import types
from collections import Counter
from dataclasses import dataclass
from functools import partial
import torch
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Optional
from saturn.log import get_logger, progress

log = get_logger(__name__)


# Random TTA crops per object. Each object gets this + 1 forward passes (the
# original is always included). Changing it changes orientation estimates, not
# just throughput.
TTA_CROPS = 3


def _tta_crops(explicit: Optional[int] = None) -> int:
    """Random TTA crops per object: ``explicit`` if given, else ``TTA_CROPS``."""
    return int(explicit) if explicit is not None else TTA_CROPS


try:
    from scipy.optimize import curve_fit
    from scipy.integrate import trapezoid
    _HAS_SCIPY = True
except Exception:
    curve_fit = None
    trapezoid = None
    _HAS_SCIPY = False


# ============================================================================
# Test-Time Augmentation Helper Functions
# ============================================================================

def random_crop(image: Image.Image, crop_scale: Tuple[float, float] = (0.8, 0.95), rng: "random.Random" = None) -> Image.Image:
    """
    Randomly crop an image to a percentage of its original size.
    
    Args:
        image: PIL Image to crop.
        crop_scale: Tuple of (min_scale, max_scale) for crop size relative to original.
    
    Returns:
        Cropped PIL Image.
    """
    assert isinstance(image, Image.Image), "Input must be PIL.Image.Image"
    assert len(crop_scale) == 2 and 0 < crop_scale[0] <= crop_scale[1] <= 1
    
    width, height = image.size
    
    _r = rng or random
    crop_width = _r.randint(int(width * crop_scale[0]), int(width * crop_scale[1]))
    crop_height = _r.randint(int(height * crop_scale[0]), int(height * crop_scale[1]))
    
    left = _r.randint(0, width - crop_width)
    top = _r.randint(0, height - crop_height)
    
    cropped_image = image.crop((left, top, left + crop_width, top + crop_height))
    
    return cropped_image


def get_crop_images(image: Image.Image, num: int = 3, crop_scale: Tuple[float, float] = (0.8, 0.95)) -> List[Image.Image]:
    """
    Generate multiple random crops of an image for test-time augmentation.
    
    Args:
        image: PIL Image to crop.
        num: Number of random crops to generate.
        crop_scale: Tuple of (min_scale, max_scale) for crop size.
    
    Returns:
        List of cropped PIL Images.
    """
    # Deterministic TTA: seed the crop RNG from the crop's pixel content, so the
    # same object crop always yields the same augmentations.
    import zlib
    rng = random.Random(zlib.crc32(image.tobytes()) ^ (image.size[0] << 16) ^ image.size[1])
    cropped_images = []
    for _ in range(num):
        cropped_images.append(random_crop(image, crop_scale, rng=rng))
    return cropped_images


def remove_outliers_and_average(tensor: torch.Tensor, threshold: float = 1.5) -> float:
    """
    Remove outliers using IQR method and return the average of remaining values.
    Used for polar and rotation angles (linear values).
    
    Args:
        tensor: 1D tensor of angle values.
        threshold: IQR multiplier for outlier detection.
    
    Returns:
        Average value after outlier removal.
    """
    assert tensor.dim() == 1, "Input tensor must be 1-dimensional"
    
    if len(tensor) == 0:
        return 0.0
    if len(tensor) == 1:
        return float(tensor[0])
    
    # Calculate IQR
    q1 = torch.quantile(tensor, 0.25)
    q3 = torch.quantile(tensor, 0.75)
    iqr = q3 - q1
    
    # Define bounds
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    # Filter outliers
    non_outliers = tensor[(tensor >= lower_bound) & (tensor <= upper_bound)]
    
    if len(non_outliers) == 0:
        return float(torch.mean(tensor))
    
    return float(torch.mean(non_outliers))


def remove_outliers_and_average_circular(tensor: torch.Tensor, threshold: float = 1.5) -> float:
    """
    Remove outliers and average for circular (angle) data.
    Used for azimuth which wraps around at 360 degrees.
    
    This method converts angles to 2D unit vectors, removes outliers based on
    distance from the mean vector, and computes the circular mean of remaining values.
    
    Args:
        tensor: 1D tensor of angle values in degrees (0-360).
        threshold: IQR multiplier for outlier detection.
    
    Returns:
        Average angle in degrees after outlier removal.
    """
    assert tensor.dim() == 1, "Input tensor must be 1-dimensional"
    
    if len(tensor) == 0:
        return 0.0
    if len(tensor) == 1:
        return float(tensor[0])
    
    # Convert angles to 2D points on unit circle
    radians = tensor * torch.pi / 180.0
    x_coords = torch.cos(radians)
    y_coords = torch.sin(radians)
    
    # Compute mean vector
    mean_x = torch.mean(x_coords)
    mean_y = torch.mean(y_coords)
    
    # Compute distances from mean
    differences = torch.sqrt((x_coords - mean_x) ** 2 + (y_coords - mean_y) ** 2)
    
    # Calculate IQR of distances
    q1 = torch.quantile(differences, 0.25)
    q3 = torch.quantile(differences, 0.75)
    iqr = q3 - q1
    
    # Define bounds
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    # Filter outliers
    non_outliers = tensor[(differences >= lower_bound) & (differences <= upper_bound)]
    
    if len(non_outliers) == 0:
        # Fallback: use all values
        mean_angle = torch.atan2(mean_y, mean_x) * 180.0 / torch.pi
        mean_angle = (mean_angle + 360) % 360
        return float(mean_angle)
    
    # Recompute circular mean for non-outliers
    radians = non_outliers * torch.pi / 180.0
    x_coords = torch.cos(radians)
    y_coords = torch.sin(radians)
    
    mean_x = torch.mean(x_coords)
    mean_y = torch.mean(y_coords)
    
    mean_angle = torch.atan2(mean_y, mean_x) * 180.0 / torch.pi
    mean_angle = (mean_angle + 360) % 360
    
    return float(mean_angle)


# Orient-Anything V2 paths
HF_CKPT_PATH_V2 = "demo_ckpts/rotmod_realrotaug_best.pt"
HF_REPO_V2 = "Viglong/OriAnyV2_ckpt"

def _alpha_confidence(dir_num: int) -> float:
    """Map the OA-V2 symmetry alpha to a front-reliability score in [0, 1].

    Alpha is a symmetry class, not a certainty: 1 = unique front (fully
    reliable), 2/4 = n-fold symmetric (the argmax is one of n equal peaks),
    0 = no front at all.
    """
    return 1.0 / dir_num if dir_num > 0 else 0.0


def _mode_dir_num(results: List["OrientationResult"]) -> int:
    """Most common symmetry alpha over TTA crops (ties: first seen)."""
    return Counter(r.dir_num for r in results).most_common(1)[0][0]


@dataclass
class OrientationResult:
    """Data class to hold orientation estimation results."""
    azimuth: float      # 0-360 degrees, horizontal rotation
    polar: float        # -90 to 90 degrees, vertical angle
    rotation: float     # -180 to 180 degrees, roll
    confidence: float   # 0-1, OA-V2 alpha-derived confidence (normalized)
    dir_num: int = 1    # OA-V2 symmetry alpha class in {0,1,2,4}
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'azimuth': self.azimuth,
            'polar': self.polar,
            'rotation': self.rotation,
            'confidence': self.confidence,
            'dir_num': self.dir_num,
        }
    
    def get_projected_axes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the 2D projection of the object's 3D axes (Front, Right, Top).
        Based on Orient-Anything-V2 visualization logic.
        
        Returns:
            Tuple of (front, right, top) 2D vectors.
        """
        phi = np.radians(self.azimuth)
        theta = np.radians(self.polar)
        gamma = np.radians(-1 * self.rotation)
        
        # Front (X)
        x = np.array([
            -1 * np.sin(phi) * np.cos(gamma) - np.cos(phi) * np.sin(theta) * np.sin(gamma),
            np.sin(phi) * np.sin(gamma) - np.cos(phi) * np.sin(theta) * np.cos(gamma)
        ])
        
        # Right (Y)
        y = np.array([
            -1 * np.cos(phi) * np.cos(gamma) + np.sin(phi) * np.sin(theta) * np.sin(gamma),
            np.cos(phi) * np.sin(gamma) + np.sin(phi) * np.sin(theta) * np.cos(gamma)
        ])
        
        # Top (Z)
        z = np.array([
            np.cos(theta) * np.sin(gamma),
            np.cos(theta) * np.cos(gamma)
        ])
        
        return x, y, z

    def get_front_direction_2d(self) -> np.ndarray:
        """
        Get the 2D projection of the object's front-facing direction.
        Returns a normalized 2D vector [dx, dy] representing where the object is facing.
        """
        front, _, _ = self.get_projected_axes()
        norm = np.linalg.norm(front)
        return front / norm if norm > 1e-6 else front


class OrientAnythingEstimator:
    """
    Orient Anything V2 model for estimating object orientation from images.
    
    This estimator uses the VGGT backbone with an MLP head to predict:
    - Azimuth (0-360°): Horizontal rotation angle
    - Polar/Elevation (-90 to 90°): Vertical angle 
    - Rotation (-180 to 180°): Roll angle
    - Confidence (0-1): Model confidence in the prediction
    
    The model is designed to work on single-object images, so for multi-object
    scenes, objects should be isolated (e.g., via segmentation masks) before
    orientation estimation.
    """
    
    def __init__(
        self, 
        device: str = None,
        cache_dir: str = './',
        auto_download: bool = True,
        orient_anything_v2_path: str = None
    ):
        """
        Initialize the Orient Anything V2 estimator.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
            cache_dir: Directory to cache downloaded models.
            auto_download: Whether to automatically download model weights.
            orient_anything_v2_path: Path to Orient-Anything-V2 repo. If None, auto-detect.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        self.auto_download = auto_download
        self.initialized = False
        self.orient_anything_v2_path = orient_anything_v2_path
        
        # Determine dtype based on GPU capability
        if torch.cuda.is_available():
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        
        self.model = None
    
    def _init_model(self):
        """Lazily initialize the Orient-Anything V2 model."""
        if self.initialized:
            return

        # Add Orient-Anything-V2 to path
        if self.orient_anything_v2_path is None:
            # Auto-detect common locations
            possible_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'tools', 'Orient-Anything-V2'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'Orient-Anything-V2'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 'Orient-Anything-V2'),
                os.path.join(env("SATURN_TOOLS_DIR"), 'Orient-Anything-V2'),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    self.orient_anything_v2_path = p
                    break
        
        if self.orient_anything_v2_path and self.orient_anything_v2_path not in sys.path:
            sys.path.insert(0, self.orient_anything_v2_path)

        oa_utils_path = None
        if self.orient_anything_v2_path:
            oa_utils_path = os.path.join(self.orient_anything_v2_path, 'utils')
        if oa_utils_path and os.path.isdir(oa_utils_path):
            existing_utils = sys.modules.get('utils')
            if existing_utils is None or not hasattr(existing_utils, '__path__'):
                utils_pkg = types.ModuleType('utils')
                utils_pkg.__path__ = [oa_utils_path]
                sys.modules['utils'] = utils_pkg
            elif oa_utils_path not in existing_utils.__path__:
                existing_utils.__path__.insert(0, oa_utils_path)

        try:
            from vision_tower import VGGT_OriAny_Ref
            from utils.app_utils import preprocess_images, background_preprocess
        except ImportError as e:
            raise ImportError(
                f"Could not import Orient-Anything-V2. Make sure the path is correct: {e}"
            )
        
        # Store preprocessing functions
        self._preprocess_images = preprocess_images
        self._background_preprocess = background_preprocess
        
        # Load checkpoint
        if self.auto_download:
            try:
                from huggingface_hub import hf_hub_download
                ckpt_path = hf_hub_download(
                    repo_id=HF_REPO_V2,
                    filename=HF_CKPT_PATH_V2,
                    repo_type="model",
                    cache_dir=self.cache_dir,
                    resume_download=True
                )
            except Exception as e:
                # Try local path
                local_path = os.path.join(self.orient_anything_v2_path, 
                    'models--Viglong--OriAnyV2_ckpt/snapshots/9920497694862d84faef12004e31753d335ae187/demo_ckpts/rotmod_realrotaug_best.pt')
                if os.path.exists(local_path):
                    ckpt_path = local_path
                else:
                    raise RuntimeError(f"Could not download or find checkpoint: {e}")
        
        # Initialize model
        # Output dim: 360 (azimuth) + 180 (elevation) + 360 (rotation) = 900
        self.model = VGGT_OriAny_Ref(out_dim=900, dtype=self.dtype, nopretrain=True)
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.model.eval()
        self.model = self.model.to(self.device)
        
        self.initialized = True
        log.info(f"OrientAnything V2 model initialized on {self.device}")
    
    def _preprocess_single(self, image: Image.Image, remove_background: bool = False) -> torch.Tensor:
        """Preprocess a single image for V2 model."""
        if remove_background:
            image = self._background_preprocess(image, True)
        return self._preprocess_images([image], mode="pad").to(self.device)
    
    def _preprocess_batch(self, images: List[Image.Image], remove_background: bool = False) -> torch.Tensor:
        """Preprocess a batch of images for V2 model."""
        if remove_background:
            images = [self._background_preprocess(img, True) for img in images]
        return self._preprocess_images(images, mode="pad").to(self.device)
    
    def _parse_predictions(self, pred: torch.Tensor, batch_idx: int = 0) -> OrientationResult:
        """Parse V2 model predictions into OrientationResult.
        
        V2 output structure: 360 (azimuth) + 180 (elevation) + 360 (rotation)
        """
        # Handle different pred shapes
        if pred.dim() == 3:
            # Shape: (B, S, D) where S=1 for single image
            pred_flat = pred.view(-1, pred.shape[-1])
            p = pred_flat[batch_idx]
        elif pred.dim() == 2:
            p = pred[batch_idx]
        else:
            p = pred
        
        azimuth = torch.argmax(p[0:360]).float()
        polar = torch.argmax(p[360:360+180]).float() - 90  # elevation: -90 to 90
        rotation = torch.argmax(p[360+180:360+180+360]).float() - 180  # rotation: -180 to 180

        # OA-V2-style alpha estimation from azimuth distribution (utils/app_utils.py)
        az_logits = p[0:360]
        az_distribution = torch.sigmoid(az_logits).detach().float().cpu().numpy()[None, :]
        dir_num = int(self._val_fit_alpha(az_distribution)[0])

        # Normalized confidence for downstream code expecting [0,1].
        # Raw OA-V2 alpha is exposed as `dir_num` in {0,1,2,4}.
        confidence = _alpha_confidence(dir_num)
        
        return OrientationResult(
            azimuth=float(azimuth),
            polar=float(polar),
            rotation=float(rotation),
            confidence=confidence,
            dir_num=dir_num,
        )

    @staticmethod
    def _von_mises_pdf_alpha_numpy(alpha: float, x: np.ndarray, mu: float, kappa: float) -> np.ndarray:
        normalization = 2 * np.pi
        return np.exp(kappa * np.cos(alpha * (x - mu))) / normalization

    def _val_fit_alpha(self, distribute: np.ndarray) -> np.ndarray:
        """OA-V2 alpha fitting from utils/app_utils.py with a safe fallback when SciPy is unavailable."""
        distribute = np.asarray(distribute, dtype=np.float64)
        if distribute.ndim == 1:
            distribute = distribute[None, :]

        if distribute.shape[-1] != 360:
            return np.ones((distribute.shape[0],), dtype=np.int64)

        # Fallback heuristic when scipy is unavailable.
        if not _HAS_SCIPY:
            out = []
            for y_noise in distribute:
                y = np.asarray(y_noise, dtype=np.float64)
                y = np.maximum(y, 0.0)
                s = float(y.sum()) + 1e-8
                y = y / s
                peak = float(np.max(y))
                if peak < 0.005:
                    out.append(0)
                elif peak < 0.012:
                    out.append(4)
                elif peak < 0.025:
                    out.append(2)
                else:
                    out.append(1)
            return np.asarray(out, dtype=np.int64)

        fit_alphas: List[int] = []
        x = np.linspace(0, 2 * np.pi, 360)
        alphas = [1.0, 2.0, 4.0]

        for y_noise in distribute:
            y_noise = np.asarray(y_noise, dtype=np.float64)
            y_noise = np.maximum(y_noise, 0.0)
            y_noise /= float(trapezoid(y_noise, x) + 1e-8)

            initial_guess = [x[int(np.argmax(y_noise))], 1.0]
            saved_params = []
            saved_r_squared = []

            for alpha in alphas:
                try:
                    vm_partial = partial(self._von_mises_pdf_alpha_numpy, alpha)
                    params, _ = curve_fit(vm_partial, x, y_noise, p0=initial_guess)

                    residuals = y_noise - vm_partial(x, *params)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_noise - np.mean(y_noise)) ** 2)
                    r_squared = 1 - (ss_res / (ss_tot + 1e-8))

                    saved_params.append(params)
                    saved_r_squared.append(r_squared)
                    if r_squared > 0.8:
                        break
                except Exception:
                    saved_params.append((0.0, 0.0))
                    saved_r_squared.append(0.0)

            max_index = int(np.argmax(saved_r_squared))
            alpha = float(alphas[max_index])
            _, kappa_fit = saved_params[max_index]
            r_squared = float(saved_r_squared[max_index])

            if alpha == 1.0 and kappa_fit >= 0.6 and r_squared >= 0.45:
                pass
            elif alpha == 2.0 and kappa_fit >= 0.5 and r_squared >= 0.45:
                pass
            elif alpha == 4.0 and kappa_fit >= 0.25 and r_squared >= 0.45:
                pass
            else:
                alpha = 0.0

            fit_alphas.append(int(alpha))

        return np.asarray(fit_alphas, dtype=np.int64)
    
    def estimate_orientation(self, image: Image.Image, remove_background: bool = False) -> OrientationResult:
        """
        Estimate orientation of an object in an image.
        
        Args:
            image: PIL Image containing a single object (ideally with background removed).
            remove_background: Whether to remove background before inference.
        
        Returns:
            OrientationResult with azimuth, polar, rotation angles and confidence.
        """
        self._init_model()

        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image")

        # Convert to RGB
        if image.mode == 'RGBA':
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        image = image.convert('RGB')

        # Preprocess using V2 pipeline
        image_tensor = self._preprocess_single(image, remove_background)
        # Add sequence dimension: (B, C, H, W) -> (B, S, C, H, W) where S=1
        image_tensor = image_tensor.unsqueeze(1)

        # Inference
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
                pred = self.model(image_tensor)
        
        return self._parse_predictions(pred, 0)
    
    def estimate_orientations_batch(self, images: List[Image.Image], remove_background: bool = False) -> List[OrientationResult]:
        """
        Estimate orientations for a batch of images with a single batched forward.

        Args:
            images: List of PIL Images, each containing a single object.
            remove_background: Whether to remove background before inference.

        Returns:
            List of OrientationResult objects.
        """
        self._init_model()

        if len(images) == 0:
            return []

        rgb_images: List[Image.Image] = []
        for img in images:
            if not isinstance(img, Image.Image):
                raise TypeError("All inputs must be PIL Images")
            if img.mode == 'RGBA':
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)
            rgb_images.append(img.convert('RGB'))

        # V2 (VGGT) supports a true batch dim: (B, S=1, C, H, W) → (B, 900).
        # The forward is chunked to bound peak VRAM: server-side batching ×
        # per-request boxes × TTA crops can reach hundreds of crops.
        chunk_size = int(env("SAPY_ORIANY_INTERNAL_CHUNK"))
        batch = self._preprocess_batch(rgb_images, remove_background).unsqueeze(1)
        preds: List[torch.Tensor] = []
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
                for start in range(0, batch.shape[0], chunk_size):
                    sub = batch[start:start + chunk_size]
                    preds.append(self.model(sub))
        pred = torch.cat(preds, dim=0) if len(preds) > 1 else preds[0]

        return [self._parse_predictions(pred, i) for i in range(len(rgb_images))]
    
    def estimate_orientation_with_tta(
        self,
        image: Image.Image,
        num_crops: int = 3,
        crop_scale: Tuple[float, float] = (0.8, 0.95),
        include_original: bool = True,
        outlier_threshold: float = 1.5,
        remove_background: bool = False
    ) -> OrientationResult:
        """
        Estimate orientation with test-time augmentation for improved robustness.

        This method creates multiple random crops of the input image, runs inference
        on each crop, removes outliers using IQR method, and averages the results.

        Args:
            image: PIL Image containing a single object.
            num_crops: Number of random crops to generate. Default 3.
            crop_scale: Tuple of (min_scale, max_scale) for crop size. Default (0.8, 0.95).
            include_original: Whether to include the original image in the ensemble. Default True.
            outlier_threshold: IQR multiplier for outlier detection. Default 1.5.
            remove_background: Whether to remove background before inference.

        Returns:
            OrientationResult with averaged azimuth, polar, rotation and mean confidence.
        """
        self._init_model()
        
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image")
        
        # Convert to RGB if necessary
        if image.mode == 'RGBA':
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        image = image.convert('RGB')
        
        # Generate augmented images
        augmented_images = get_crop_images(image, num=num_crops, crop_scale=crop_scale)
        
        # Optionally include the original image
        if include_original:
            augmented_images.append(image)
        
        # Get predictions for all augmented images
        all_results = self.estimate_orientations_batch(augmented_images, remove_background)
        
        # Extract predictions as tensors for aggregation
        azimuth_preds = torch.tensor([r.azimuth for r in all_results])
        polar_preds = torch.tensor([r.polar + 90 for r in all_results])  # Convert back to 0-180 range for averaging
        rotation_preds = torch.tensor([r.rotation + 180 for r in all_results])  # Convert back to 0-360 range
        confidence_preds = torch.tensor([r.confidence for r in all_results])
        
        # Aggregate with outlier removal
        # Azimuth uses circular averaging (handles 0/360 wraparound)
        final_azimuth = remove_outliers_and_average_circular(azimuth_preds, threshold=outlier_threshold)
        
        # Polar and rotation use standard averaging (linear values)
        final_polar = remove_outliers_and_average(polar_preds, threshold=outlier_threshold) - 90
        final_rotation = remove_outliers_and_average(rotation_preds, threshold=outlier_threshold) - 180
        
        # Average confidence
        final_confidence = float(torch.mean(confidence_preds))
        
        return OrientationResult(
            azimuth=final_azimuth,
            polar=final_polar,
            rotation=final_rotation,
            confidence=final_confidence,
            dir_num=_mode_dir_num(all_results),
        )
    
    def estimate_orientations_batch_with_tta(
        self,
        images: List[Image.Image],
        num_crops: int = 3,
        crop_scale: Tuple[float, float] = (0.8, 0.95),
        include_original: bool = True,
        outlier_threshold: float = 1.5,
        remove_background: bool = False
    ) -> List[OrientationResult]:
        """
        Batched TTA: all augmentations of all images go through one forward pass.

        For B input images and K=num_crops (+1 if include_original), this issues a
        single B*(K+1) forward instead of B*(K+1) sequential forwards.

        Returns one OrientationResult per input image, aggregated with the same
        outlier-removal + circular/linear averaging as estimate_orientation_with_tta.
        """
        self._init_model()
        if len(images) == 0:
            return []

        # Build per-image augmentation groups, then flatten for one big forward.
        flat_images: List[Image.Image] = []
        group_sizes: List[int] = []
        for img in images:
            if not isinstance(img, Image.Image):
                raise TypeError("All inputs must be PIL Images")
            if img.mode == 'RGBA':
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)
            rgb = img.convert('RGB')
            aug = get_crop_images(rgb, num=num_crops, crop_scale=crop_scale)
            if include_original:
                aug.append(rgb)
            group_sizes.append(len(aug))
            flat_images.extend(aug)

        flat_results = self.estimate_orientations_batch(flat_images, remove_background)

        # Aggregate per source image.
        out: List[OrientationResult] = []
        cursor = 0
        for size in group_sizes:
            grp = flat_results[cursor:cursor + size]
            cursor += size

            azimuth_preds = torch.tensor([r.azimuth for r in grp])
            polar_preds = torch.tensor([r.polar + 90 for r in grp])
            rotation_preds = torch.tensor([r.rotation + 180 for r in grp])
            confidence_preds = torch.tensor([r.confidence for r in grp])

            final_az = remove_outliers_and_average_circular(
                azimuth_preds, threshold=outlier_threshold
            )
            final_pol = remove_outliers_and_average(
                polar_preds, threshold=outlier_threshold
            ) - 90
            final_rot = remove_outliers_and_average(
                rotation_preds, threshold=outlier_threshold
            ) - 180
            final_conf = float(torch.mean(confidence_preds))

            out.append(OrientationResult(
                azimuth=final_az,
                polar=final_pol,
                rotation=final_rot,
                confidence=final_conf,
                dir_num=_mode_dir_num(grp),
            ))
        return out


class ObjectPerspectiveSpatialRelations:
    """
    Extract per-object crops (mask-isolated when masks are given) from a scene image
    for Orient-Anything V2 orientation estimation.
    """
    
    def __init__(
        self, 
        orientation_estimator: OrientAnythingEstimator = None,
        device: str = None,
        orient_anything_v2_path: str = None
    ):
        """
        Initialize the per-object crop extractor.
        
        Args:
            orientation_estimator: Pre-initialized OrientAnythingEstimator (optional).
            device: Device to use.
            orient_anything_v2_path: Path to Orient-Anything-V2 repo.
        """
        if orientation_estimator is not None:
            self.orientation_estimator = orientation_estimator
        else:
            self.orientation_estimator = OrientAnythingEstimator(
                device=device,
                orient_anything_v2_path=orient_anything_v2_path
            )
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def extract_object_images(
        self,
        image: Image.Image,
        bboxes: np.ndarray,
        masks: Optional[List[np.ndarray]] = None,
        padding: int = 10
    ) -> List[Image.Image]:
        """
        Extract cropped images of individual objects for orientation estimation.
        
        Args:
            image: Full scene image.
            bboxes: Array of bounding boxes [N, 4] in (x1, y1, x2, y2) format.
            masks: Optional list of segmentation masks for cleaner extraction.
            padding: Padding around bounding boxes.
        
        Returns:
            List of cropped PIL Images, one per object.
        """
        extracted = []
        img_w, img_h = image.size
        
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img_w, x2 + padding)
            y2 = min(img_h, y2 + padding)
            
            if masks is not None and i < len(masks):
                # Use mask for cleaner extraction
                mask = masks[i]
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                if mask.ndim == 3:
                    mask = mask.squeeze()
                
                # Create RGBA image with transparent background
                rgba = image.convert('RGBA')
                rgba_arr = np.array(rgba)
                
                # Apply mask to alpha channel
                alpha = (mask > 0).astype(np.uint8) * 255
                if alpha.shape != (img_h, img_w):
                    from PIL import Image as PILImage
                    alpha_img = PILImage.fromarray(alpha)
                    alpha_img = alpha_img.resize((img_w, img_h), PILImage.NEAREST)
                    alpha = np.array(alpha_img)
                
                rgba_arr[:, :, 3] = alpha
                masked_img = Image.fromarray(rgba_arr, 'RGBA')
                cropped = masked_img.crop((x1, y1, x2, y2))
                
                # Convert back to RGB on a black background (note: OA-V2's own
                # RGBA preprocessing composites onto white)
                rgb_crop = Image.new('RGB', cropped.size, (0, 0, 0))
                rgb_crop.paste(cropped, mask=cropped.split()[3])
                extracted.append(rgb_crop)
            else:
                # Simple crop
                cropped = image.crop((x1, y1, x2, y2))
                extracted.append(cropped.convert('RGB'))
        
        return extracted
    

def test_orient_anything():
    """Simple test function."""
    progress("Testing OrientAnythingEstimator...")
    
    # Create a simple test image (red square)
    test_img = Image.new('RGB', (256, 256), color='white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_img)
    draw.rectangle([80, 80, 180, 180], fill='red')
    
    # Initialize estimator
    estimator = OrientAnythingEstimator()
    
    # Test 1: Standard orientation estimation
    progress("\n--- Standard Orientation Estimation ---")
    result = estimator.estimate_orientation(test_img)
    progress(f"Orientation result: {result}")
    progress(f"Front direction 2D: {result.get_front_direction_2d()}")
    
    # Test 2: Orientation estimation with Test-Time Augmentation
    progress("\n--- Orientation Estimation with TTA ---")
    result_tta = estimator.estimate_orientation_with_tta(
        test_img,
        num_crops=6,
        crop_scale=(0.8, 0.95),
        include_original=True,
        outlier_threshold=1.5
    )
    progress(f"TTA Orientation result: {result_tta}")
    progress(f"TTA Front direction 2D: {result_tta.get_front_direction_2d()}")
    
    # Compare results
    progress("\n--- Comparison ---")
    progress(f"Azimuth diff: {abs(result.azimuth - result_tta.azimuth):.2f}°")
    progress(f"Polar diff: {abs(result.polar - result_tta.polar):.2f}°")
    progress(f"Rotation diff: {abs(result.rotation - result_tta.rotation):.2f}°")
    progress(f"Confidence (standard): {result.confidence:.3f}")
    progress(f"Confidence (TTA): {result_tta.confidence:.3f}")
    
    return result, result_tta


if __name__ == "__main__":
    test_orient_anything()
