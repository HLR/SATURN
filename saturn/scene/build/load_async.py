"""Async scene loading for the benchmark runner.

Reuses the helpers in ``load.py`` and gathers SAM3 / OriAny / VGGT calls
against Ray Serve and vLLM.
"""

from __future__ import annotations

from saturn.settings import env
import asyncio
import inspect
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union


import numpy as np
from PIL import Image

from saturn.perception.codec import encode_image

from saturn.scene.adapters import from_camera_rotation_matrix
from saturn.scene.fusion import fit_ground_plane_xz
from .load import (
    _bbox_to_mask,
    _world_geometry_from_depth,
    _extract_mask_world_points,
    _extrinsic_to_4x4,
    _level_merged_dicts,
    _make_static_depth_provider,
    _merge_new_observations,
)
from saturn.scene.scene import Scene
from saturn.scene.types import Camera, MergedObject
from saturn.log import get_logger

log = get_logger(__name__)


async def _call_maybe_async(fn, *args, **kwargs):
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _run_sync_on_loop(loop: asyncio.AbstractEventLoop, coro):
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        raise RuntimeError(
            "Sync scene.detect()/ground() cannot run on the event-loop thread; "
            "use scene.detect_async()/ground_async() instead."
        )
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()


def _normalize_keywords(
    keywords: Optional[Union[List[str], Dict[int, List[str]]]], num_views: int
) -> Optional[Dict[int, List[str]]]:
    if isinstance(keywords, list):
        return {v: list(keywords) for v in range(num_views)}
    return keywords


def _make_static_orientation_provider(estimates):
    class _StaticOrientationProvider:
        def __init__(self, vals):
            self._vals = list(vals)

        def predict(self, image, bboxes, masks=None):
            return self._vals

    return _StaticOrientationProvider(estimates)


def _rehydrate_vggt_result(payload: Dict[str, Any]):
    """Rebuild a ``VGGTResult`` from the serialized Ray Serve payload."""
    transform_info = payload.get("transform_info", []) or []

    from saturn.perception.reconstruction.vggt import VGGTResult

    world_points = np.asarray(payload["world_points"])
    world_points_conf = np.asarray(payload["world_points_conf"])
    depth = np.asarray(payload["depth"])
    depth_conf = np.asarray(payload["depth_conf"])
    extrinsics = np.asarray(payload["extrinsics"])
    intrinsics = np.asarray(payload["intrinsics"])
    num_views = int(payload.get("num_views", world_points.shape[0]))
    return VGGTResult(
        world_points=world_points,
        world_points_conf=world_points_conf,
        depth=depth,
        depth_conf=depth_conf,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        transform_info=transform_info,
        num_views=num_views,
    )


async def _run_vggt_async(vggt_reconstructor, images: List[Image.Image]):
    if hasattr(vggt_reconstructor, "reconstruct") and hasattr(
        vggt_reconstructor.reconstruct, "remote"
    ):
        payload = await vggt_reconstructor.reconstruct.remote(
            [encode_image(img) for img in images]
        )
        return _rehydrate_vggt_result(payload)
    return await _call_maybe_async(vggt_reconstructor.reconstruct, images)


# Relative detection rule: a box must score at least _DET_RHO x the best box of
# its (image, phrase) and at least _DET_FLOOR (also the SAM3 request threshold).
_DET_RHO = 0.5
_DET_FLOOR = 0.05


def keep_relative(dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Relative detection rule: keep a box scoring at least RHO x the best box
    for this (image, phrase) and at least the floor. SAM3 scales all scores of an
    image by a per-image presence factor, so a fixed cutoff can silently drop
    every object of a scene."""
    if not dets:
        return dets
    best = max(float(d.get("score", 0.0)) for d in dets)
    cut = max(_DET_FLOOR, _DET_RHO * best)
    return [d for d in dets if float(d.get("score", 0.0)) >= cut]


def _convert_sam3_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    masks = result.get("masks", [])
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    if len(boxes) == 0:
        return []

    score_values = np.asarray(scores if len(scores) > 0 else np.ones(len(boxes)), dtype=float).reshape(-1)
    order = np.argsort(score_values)[::-1]
    detections = []
    for det_idx in order:
        box = boxes[det_idx]
        if hasattr(box, "cpu"):
            box = box.detach().cpu().tolist()
        elif hasattr(box, "tolist"):
            box = box.tolist()

        mask = masks[det_idx] if det_idx < len(masks) else None
        if mask is not None:
            if hasattr(mask, "cpu"):
                mask = mask.detach().cpu().numpy().astype(bool)
            else:
                mask = np.asarray(mask, dtype=bool)
            while mask.ndim > 2:
                mask = mask[0]

        detections.append(
            {
                "box": list(box),
                "mask": mask,
                "score": float(score_values[det_idx]) if det_idx < len(score_values) else 1.0,
            }
        )
    return detections


async def _run_sam3_per_view_async(
    sam3,
    images: List[Image.Image],
    keywords: Optional[Union[List[str], Dict[int, List[str]]]] = None,
) -> List[Dict[str, List[Dict[str, Any]]]]:
    keywords = _normalize_keywords(keywords, len(images))

    async def _one_view(view_idx: int, image: Image.Image):
        view_det: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        if keywords is not None and view_idx in keywords:
            results = await asyncio.gather(
                *[
                    _call_maybe_async(sam3.predict_with_masks, image, kw, _DET_FLOOR)
                    for kw in keywords[view_idx]
                ]
            )
            for kw, result in zip(keywords[view_idx], results):
                view_det[kw].extend(keep_relative(_convert_sam3_result(result)))

        return dict(view_det)

    return list(await asyncio.gather(*[_one_view(i, img) for i, img in enumerate(images)]))


async def _extract_per_view_observations_async(
    images: List[Image.Image],
    depth_predictions: list,
    world_geometries: List[Dict[str, Any]],
    per_view_detections: List[Dict[str, List[Dict[str, Any]]]],
    orientation_provider=None,
    conf_percentile: float = 40.0,
) -> List[Dict[str, List[Dict[str, Any]]]]:
    """Per-view 3D observations: for each view, keyword -> observation dicts
    holding the world points under the detection mask and the detection's
    world-frame orientation."""
    return list(
        await asyncio.gather(
            *[
                _view_observations_async(
                    i, image, pred, geom, det_map, orientation_provider, conf_percentile
                )
                for i, (image, pred, geom, det_map) in enumerate(
                    zip(images, depth_predictions, world_geometries, per_view_detections)
                )
            ]
        )
    )


async def _view_observations_async(
    view_idx: int,
    image: Image.Image,
    pred,
    geom: Dict[str, Any],
    det_map: Dict[str, List[Dict[str, Any]]],
    orientation_provider,
    conf_percentile: float,
) -> Dict[str, List[Dict[str, Any]]]:
    """Observations of one view, keyword -> list of observation dicts."""
    view_obs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    detections, keywords_for_detections = _view_detections(image, det_map)
    if not detections:
        return view_obs

    objects = await _extract_view_objects_async(image, pred, detections, orientation_provider)
    if env("SAPY_FUSION_DEBUG"):
        log.debug(f"[detcount] view={view_idx} detections={len(detections)} "
                  f"extracted_objects={len(objects)} keywords={len(keywords_for_detections)}")
    for obj, keyword in zip(objects, keywords_for_detections):
        view_obs[keyword].append(
            _observation_from_object(obj, geom, pred, view_idx, conf_percentile)
        )
    return view_obs


def _view_detections(image: Image.Image, det_map: Dict[str, List[Dict[str, Any]]]):
    """The detections of one view as Detection2D (a box-filled mask stands in
    for a missing mask), with the keyword of each."""
    from saturn.perception.types import Detection2D

    detections = []
    keywords_for_detections = []
    for keyword, det_infos in det_map.items():
        for det_info in det_infos:
            bbox = tuple(map(float, det_info["box"]))
            mask = det_info.get("mask")
            if mask is None:
                mask = _bbox_to_mask(image.size, bbox)
            else:
                mask = np.asarray(mask, dtype=bool)
                while mask.ndim > 2:
                    mask = mask[0]
            detections.append(
                Detection2D(
                    bbox_xyxy=bbox,
                    mask=mask,
                    label=keyword,
                    score=float(det_info.get("score", 1.0)),
                    source="sam3_text",
                )
            )
            keywords_for_detections.append(keyword)
    return detections, keywords_for_detections


async def _extract_view_objects_async(image: Image.Image, pred, detections: list, orientation_provider):
    """Lift the detections of one view to 3D objects with the view's cached
    depth and, when a provider is given, its orientation estimates."""
    from saturn.perception.geometry.object_extraction import (
        extract_objects_with_providers,
    )

    depth_provider = _make_static_depth_provider(pred)
    static_orientation_provider = None
    if orientation_provider is not None:
        ests = await _call_maybe_async(
            orientation_provider.predict,
            image,
            [det.bbox_xyxy for det in detections],
            masks=[det.mask for det in detections],
        )
        static_orientation_provider = _make_static_orientation_provider(ests)

    config = SimpleNamespace(
        keep_point_clouds=True,
        trust_model_elevation=0.85,
        trust_pca_horizontal=0.0,
    )
    objects, _camera, _depth_est = extract_objects_with_providers(
        image,
        detections,
        depth_provider=depth_provider,
        orientation_provider=static_orientation_provider,
        config=config,
    )
    return objects


def _observation_from_object(obj, geom: Dict[str, Any], pred, view_idx: int, conf_percentile: float) -> Dict[str, Any]:
    """Observation dict of one extracted object: world points under its mask,
    world-frame orientation, and the 2D detection."""
    world_points, point_conf, mask_processed = _extract_mask_world_points(
        geom, obj.detection.mask, conf_percentile=conf_percentile
    )
    return {
        "world_points": world_points,
        **_world_orientation(obj.pose.orientation, pred),
        "bbox": list(obj.detection.bbox_xyxy),
        "mask": obj.detection.mask,
        "score": float(obj.detection.score or 0.0),
        "view_idx": view_idx,
    }


def _world_orientation(orientation, pred) -> Dict[str, Any]:
    """World-frame front/up/right of an orientation estimate, its confidence,
    and the camera-frame front. An object without an estimate faces -Z with
    confidence 0."""
    if orientation is not None and orientation.rotation_matrix is not None:
        triad = from_camera_rotation_matrix(
            orientation.rotation_matrix,
            pred.extrinsics,
            confidence=float(orientation.confidence or 0.0),
            source=orientation.source or "orientation_estimate",
        )
        front_world = triad.forward
        up_world = triad.up
        right_world = triad.right
        orientation_confidence = triad.confidence
        front_camera = (
            np.asarray(orientation.front_direction_3d, dtype=float)
            if getattr(orientation, "front_direction_3d", None) is not None
            else None
        )
    else:
        front_world = np.array([0.0, 0.0, -1.0], dtype=float)
        up_world = np.array([0.0, 1.0, 0.0], dtype=float)
        right_world = np.array([1.0, 0.0, 0.0], dtype=float)
        orientation_confidence = 0.0
        front_camera = None
    return {
        "front_world": front_world,
        "up_world": up_world,
        "right_world": right_world,
        "front_camera": front_camera,
        "orientation_confidence": orientation_confidence,
    }


def _append_merged_objects(scene, merged, source: str = "detect") -> List[int]:
    new_indices = []
    start_idx = len(scene.objects)
    for i, obj_dict in enumerate(merged):
        idx = start_idx + i
        new_obj = MergedObject(
            id=idx,
            label=obj_dict["label"],
            views=obj_dict["views"],
            center_world=obj_dict["center_world"],
            rotation_world=obj_dict["rotation_world"],
            front_world=obj_dict["front_world"],
            up_world=obj_dict["up_world"],
            right_world=obj_dict["right_world"],
            euler_world_deg=obj_dict.get("euler_world_deg", np.zeros(3)),
            dims=obj_dict["dims"],
            corners_world=obj_dict["corners_world"],
            height=float(obj_dict["dims"][1]),
            support_y=obj_dict["support_y"],
            world_points=obj_dict.get("world_points"),
            per_view_bboxes=obj_dict.get("per_view_bboxes", {}),
            per_view_masks=obj_dict.get("per_view_masks", {}),
            per_view_scores=obj_dict.get("per_view_scores", {}),
            per_view_centers=obj_dict.get("per_view_centers", {}),
            per_view_fronts=obj_dict.get("per_view_fronts", {}),
            per_view_fronts_camera=obj_dict.get("per_view_fronts_camera", {}),
            per_view_orientation_confidence=obj_dict.get(
                "per_view_orientation_confidence", {}
            ),
            metadata={**obj_dict.get("metadata", {}), "source": source},
        )
        scene.objects.append(new_obj)
        new_indices.append(idx)
    scene._invalidate_caches()
    return new_indices


async def _ground_new_objects_async(scene, description, vlm=None, sam3=None, orientation_provider=None, camera: Optional[int] = None,
                                    unique: bool = False):
    if vlm is None:
        raise RuntimeError("scene.ground() requires a VLM with ground() method.")
    if not hasattr(scene, "_depth_predictions") or scene._depth_predictions is None:
        raise RuntimeError("scene.ground() requires cached depth predictions.")
    if not hasattr(scene, "_world_geometries") or scene._world_geometries is None:
        raise RuntimeError("scene.ground() requires cached world geometries.")

    per_view_detections = []
    mask_errors = []
    for view_idx, image in enumerate(scene.images):
        if camera is not None and view_idx != camera:
            per_view_detections.append({})
            continue
        bbox = await _call_maybe_async(
            getattr(vlm, "ground_async", None) or getattr(vlm, "ground"),
            image,
            description,
        )
        if bbox is None:
            per_view_detections.append({})
            continue
        img_w, img_h = image.size
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            per_view_detections.append({})
            continue
        if (x2 - x1) >= 0.98 * img_w and (y2 - y1) >= 0.98 * img_h:
            per_view_detections.append({})
            continue
        mask = None
        if sam3 is not None:
            try:
                masks = await _call_maybe_async(sam3.predict_masks, image, [bbox])
                if masks:
                    mask = masks[0]
            except Exception as e:
                mask = None
                mask_errors.append((view_idx, e))
        per_view_detections.append(
            {description: [{"box": bbox, "mask": mask, "score": 1.0}]}
        )
    if mask_errors:
        first_view, first_error = mask_errors[0]
        log.warning(
            f"[ground] SAM3 predict_masks failed for {description!r} on {len(mask_errors)} view(s) "
            f"(first: view {first_view}, {first_error!r}); the box stands in for the mask"
        )

    if not any(per_view_detections):
        return []

    observations = await _extract_per_view_observations_async(
        scene.images,
        scene._depth_predictions,
        scene._world_geometries,
        per_view_detections,
        orientation_provider=orientation_provider,
    )
    merged = _merge_new_observations(observations, scene, unique_keyword=description if unique else None)
    return _append_merged_objects(scene, merged, source="vlm_ground")


async def _detect_new_objects_async(scene, description, camera=None, sam3=None, orientation_provider=None, unique=False):
    from saturn.perception.reconstruction.vggt import _PseudoDepthPrediction

    if sam3 is None:
        raise RuntimeError("SAM3 instance not available for scene.detect().")
    keywords = {i: [description] for i in range(len(scene.images))} if camera is None else {camera: [description]}
    per_view_detections = await _run_sam3_per_view_async(sam3, scene.images, keywords=keywords)

    if getattr(scene, "_vggt_result", None) is not None:
        if getattr(scene, "_world_geometries", None) is None:
            scene._world_geometries = [scene._vggt_result.get_world_geometry(i) for i in range(scene._vggt_result.num_views)]
        if getattr(scene, "_depth_predictions", None) is None:
            scene._depth_predictions = [_PseudoDepthPrediction(scene._vggt_result, i) for i in range(scene._vggt_result.num_views)]
    else:
        if getattr(scene, "_depth_predictions", None) is None:
            raise RuntimeError("scene.detect() needs a VGGT reconstruction or cached depth predictions.")
        if getattr(scene, "_world_geometries", None) is None:
            scene._world_geometries = [_world_geometry_from_depth(p) for p in scene._depth_predictions]

    observations = await _extract_per_view_observations_async(
        scene.images,
        scene._depth_predictions,
        scene._world_geometries,
        per_view_detections,
        orientation_provider=orientation_provider,
    )
    merged = _merge_new_observations(observations, scene, unique_keyword=description if unique else None)
    return _append_merged_objects(scene, merged, source="detect")


async def load_scene_async(
    images: List[Image.Image],
    keywords: Optional[Union[List[str], Dict[int, List[str]]]] = None,
    segmentor: str = "sam3",
    level_ground: bool = True,
    *,
    sam3=None,
    orientation_provider=None,
    vggt_reconstructor=None,
    conf_percentile: float = 40.0,
    unique_keywords: Optional[set] = None,
):
    """Build a multi-view Scene asynchronously.

    Per-view detection and the multi-view reconstruction give per-view 3D
    observations; the multi-view assignment fuses them into objects; the
    ground plane levels objects and cameras.
    """
    per_view_detections = await _detect_per_view_async(sam3, images, keywords)
    vggt_result, depth_predictions, world_geometries = await _reconstruct_async(
        images, vggt_reconstructor
    )
    observations = await _extract_per_view_observations_async(
        images,
        depth_predictions,
        world_geometries,
        per_view_detections,
        orientation_provider=orientation_provider,
        conf_percentile=conf_percentile,
    )
    merged_dicts = _fuse_observations(observations, depth_predictions, unique_keywords)

    ground_info = None
    if level_ground and merged_dicts:
        ground_info = _level_ground(merged_dicts, world_geometries)

    cameras = _build_cameras(depth_predictions, world_geometries, ground_info, len(merged_dicts))
    objects = [_merged_object(idx, obj_dict) for idx, obj_dict in enumerate(merged_dicts)]

    loop = asyncio.get_running_loop()
    scene = Scene(objects=objects, cameras=cameras, images=images, ground_info=ground_info)
    scene._depth_predictions = depth_predictions
    scene._world_geometries = world_geometries
    scene._vggt_result = vggt_result
    attach_runtime_hooks(
        scene,
        loop=loop,
        sam3=sam3,
        orientation_provider=orientation_provider,
    )
    return scene


async def _detect_per_view_async(sam3, images: List[Image.Image], keywords):
    """SAM3 detections per view (keyword -> boxes); none when SAM3 is absent."""
    if sam3 is not None:
        return await _run_sam3_per_view_async(sam3, images, keywords=keywords)
    if keywords is not None:
        raise ValueError("sam3 instance must be provided when keywords are specified.")
    return [{} for _ in images]


async def _reconstruct_async(images: List[Image.Image], vggt_reconstructor):
    """VGGT multi-view reconstruction. Returns (VGGT result, per-view depth
    predictions, per-view world geometries)."""
    from saturn.perception.reconstruction.vggt import _PseudoDepthPrediction

    if vggt_reconstructor is None:
        raise ValueError("vggt_reconstructor must be provided.")
    vggt_result = await _run_vggt_async(vggt_reconstructor, images)
    if int(vggt_result.num_views) != len(images):
        raise ValueError(
            f"Reconstruction returned {vggt_result.num_views} view(s) for {len(images)} image(s); "
            "depth, geometry and cameras need exactly one view per image."
        )
    depth_predictions = [_PseudoDepthPrediction(vggt_result, i) for i in range(vggt_result.num_views)]
    world_geometries = [vggt_result.get_world_geometry(i) for i in range(len(images))]
    return vggt_result, depth_predictions, world_geometries


def _fuse_observations(observations, depth_predictions: list, unique_keywords: Optional[set]) -> List[Dict[str, Any]]:
    """Fuse the per-view observations into object dicts (multi-view assignment).

    Cameras are not built yet on this path (they come after the objects), so
    the camera centres come straight from the extrinsics,
    camera_center_world = -R_w2c^T @ t, and the scale-invariant reference for
    fusion is their median pairwise distance.
    """
    from saturn.scene.fusion import merge_objects_by_keyword

    cam_positions = []
    for pred in depth_predictions:
        ext = np.asarray(pred.extrinsics, dtype=float)
        R_w2c = ext[:3, :3]
        t = ext[:3, 3]
        cam_positions.append(-R_w2c.T @ t)
    if len(cam_positions) >= 2:
        scene_scale = float(np.median([
            float(np.linalg.norm(cam_positions[i] - cam_positions[j]))
            for i in range(len(cam_positions))
            for j in range(i + 1, len(cam_positions))
        ]))
    else:
        scene_scale = 0.0
    return merge_objects_by_keyword(
        observations,
        scene_scale=scene_scale,
        unique_keywords=unique_keywords,
        cam_positions={i: p for i, p in enumerate(cam_positions)},
    )


def _level_ground(merged_dicts: List[Dict[str, Any]], world_geometries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fit the ground plane to the dense reconstruction and move the fused
    object dicts into the leveled world, in place. Returns the leveling
    transform."""
    from saturn.perception.orientation.convention import (
        user_euler_from_rotation_matrix,
    )

    ground_info = fit_ground_plane_xz(_dense_scene_points(world_geometries))
    _level_merged_dicts(merged_dicts, ground_info)
    for obj_dict in merged_dicts:
        if obj_dict.get("world_points") is not None and len(obj_dict["world_points"]) > 0:
            obj_dict["support_y"] = float(np.percentile(obj_dict["world_points"][:, 1], 2))
        obj_dict["euler_world_deg"] = user_euler_from_rotation_matrix(obj_dict["rotation_world"])
    return ground_info


def _dense_scene_points(world_geometries: List[Dict[str, Any]]) -> np.ndarray:
    """Finite reconstruction points of all views, at most 50000 per view
    (random subset, global numpy RNG)."""
    all_dense_points = []
    for geom in world_geometries:
        wp = geom["world_points"]
        valid = np.isfinite(wp).all(axis=-1)
        valid_pts = wp[valid]
        if len(valid_pts) > 50000:
            idx = np.random.choice(len(valid_pts), 50000, replace=False)
            valid_pts = valid_pts[idx]
        all_dense_points.append(valid_pts)
    return np.concatenate(all_dense_points, axis=0) if all_dense_points else np.zeros((0, 3), dtype=float)


def _build_cameras(
    depth_predictions: list,
    world_geometries: List[Dict[str, Any]],
    ground_info: Optional[Dict[str, Any]],
    num_objects: int,
) -> List[Camera]:
    """One Camera per view, in the leveled world when ``ground_info`` is set.
    Camera entity ids follow the object ids."""
    cameras = []
    for view_idx, (pred, _geom) in enumerate(zip(depth_predictions, world_geometries)):
        ext_4x4 = _extrinsic_to_4x4(pred.extrinsics)
        if ground_info is not None:
            ext_4x4 = _level_extrinsics(ext_4x4, ground_info)
        entity_id = num_objects + view_idx
        cameras.append(
            Camera(
                id=view_idx,
                entity_id=entity_id,
                intrinsics=np.asarray(pred.intrinsics, dtype=float),
                extrinsics=ext_4x4,
                image_size=(pred.depth.shape[0], pred.depth.shape[1]),
            )
        )
    return cameras


def _level_extrinsics(ext_4x4: np.ndarray, ground_info: Dict[str, Any]) -> np.ndarray:
    """World-to-camera extrinsics in the leveled world (x' = R_g x + [0, y_shift, 0])."""
    g_rot = ground_info["rotation"]
    g_y = ground_info["y_shift"]
    R_cam = ext_4x4[:3, :3]
    t_cam = ext_4x4[:3, 3]
    R_new = R_cam @ g_rot.T
    t_new = t_cam - R_new @ np.array([0, g_y, 0])
    leveled = np.eye(4, dtype=float)
    leveled[:3, :3] = R_new
    leveled[:3, 3] = t_new
    return leveled


def _merged_object(idx: int, obj_dict: Dict[str, Any]) -> MergedObject:
    """MergedObject with id ``idx`` from a fused object dict."""
    return MergedObject(
        id=idx,
        label=obj_dict["label"],
        views=obj_dict["views"],
        center_world=np.asarray(obj_dict["center_world"], dtype=float),
        rotation_world=np.asarray(obj_dict["rotation_world"], dtype=float),
        front_world=np.asarray(obj_dict["front_world"], dtype=float),
        up_world=np.asarray(obj_dict["up_world"], dtype=float),
        right_world=np.asarray(obj_dict["right_world"], dtype=float),
        euler_world_deg=np.asarray(obj_dict.get("euler_world_deg", np.zeros(3)), dtype=float),
        dims=np.asarray(obj_dict["dims"], dtype=float),
        corners_world=np.asarray(obj_dict["corners_world"], dtype=float),
        height=float(obj_dict["dims"][1]),
        support_y=float(obj_dict["support_y"]),
        world_points=obj_dict.get("world_points"),
        per_view_bboxes=obj_dict.get("per_view_bboxes", {}),
        per_view_masks=obj_dict.get("per_view_masks", {}),
        per_view_scores=obj_dict.get("per_view_scores", {}),
        per_view_centers=obj_dict.get("per_view_centers", {}),
        per_view_fronts=obj_dict.get("per_view_fronts", {}),
        per_view_fronts_camera=obj_dict.get("per_view_fronts_camera", {}),
        per_view_orientation_confidence=obj_dict.get(
            "per_view_orientation_confidence", {}
        ),
        metadata=obj_dict.get("metadata", {}),
    )


def attach_runtime_hooks(
    scene,
    *,
    loop,
    sam3=None,
    orientation_provider=None,
    vlm_grounder=None,
):
    """(Re)bind the non-serializable runtime callbacks on *scene*.

    ``Scene.to_dict()``/``from_dict()`` round-trips (the in-process scene
    cache, dumped scenes) drop these hooks: a from_dict scene raises
    "the scene loader must provide a detect_fn callback" on ``scene.detect()``,
    so planner-driven grounding after scene build finds no objects. Call this
    after every from_dict when live model handles are available.
    """
    scene._vlm = vlm_grounder

    async def _scene_detect_async(description: str, camera: Optional[int] = None, unique: bool = False):
        return await _detect_new_objects_async(
            scene,
            description,
            camera=camera,
            sam3=sam3,
            orientation_provider=orientation_provider,
            unique=unique,
        )

    async def _scene_ground_async(
        description: str, vlm=None, camera: Optional[int] = None, unique: bool = False,
    ):
        return await _ground_new_objects_async(
            scene,
            description,
            vlm=vlm or getattr(scene, "_vlm", None),
            sam3=sam3,
            orientation_provider=orientation_provider,
            camera=camera,
            unique=unique,
        )

    scene.detect_async = _scene_detect_async
    scene.ground_async = _scene_ground_async
    scene._detect_fn = lambda scene_ref, description, camera=None, unique=False: _run_sync_on_loop(
        loop, scene_ref.detect_async(description, camera=camera, unique=unique)
    )
    scene.ground_fn = lambda scene_ref, description, vlm, camera=None, unique=False: _run_sync_on_loop(
        loop, scene_ref.ground_async(description, vlm=vlm, camera=camera, unique=unique)
    )
    return scene
