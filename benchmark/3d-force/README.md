# 3D-FORCE

Two tasks over multi-view renders of Spatial457 scenes (Blender; exact 3D ground truth):

| file | task | items | answer |
|---|---|---|---|
| `3DForceRef.json` | **REF** — referring expression: find the object satisfying a chain/star/hybrid of frame-of-reference relations | 2088 | a 2D box in the query view (scored by IoU ≥ 0.5) |
| `3DForcePuzzle.json` | **SAG** — verify a stated spatial configuration | 1150 | yes / no |

Each item carries: the question, the view image paths (`image_filename`), the relation graph (`topology`, `depth`, per-relation `perspective` ∈ camera/object/mixed), the GT answer, and for REF the per-view GT boxes of every object (`bboxes`, `masks`). Relations use **object-centric** and **camera-centric** frames; "in front of X (X's view)" means from X's own facing direction.

## Data

The question files and the multi-view images are on Hugging Face: [`iamdanialkamali/3D-FORCE-Zip`](https://huggingface.co/datasets/iamdanialkamali/3D-FORCE-Zip). `setup/30_datasets.sh` downloads them to `data/3d-force/` and unpacks the image pack (`multiview_json_jpg_only.zip`, 9.1 GB) to `data/3d-force/multiview/`.

`image_filename` entries are absolute paths of the original renders (`.../scene_NNNNNN/HASH/image.png`); the loader rebases them onto `data/3d-force/multiview/` with the pack's `.jpg` extension (`FORCE3D_DATA_ROOT`, `FORCE3D_IMAGE_ROOT` and `FORCE3D_IMAGE_EXT` override the defaults). Every view directory also contains `bboxes.json` (per-object boxes, masks, visibility, a `slot` id stable across views) and `camera.json`.

## Evaluation

`python -m saturn.cli` scores REF by IoU of the predicted object's box in the query view and SAG by exact match. Per-stratum breakdowns (topology × depth × views) are what the paper reports; aggregate accuracy alone hides the depth effect.
