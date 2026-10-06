#!/usr/bin/env bash
# Evaluation data from Hugging Face -> data/: 3D-FORCE (question files + 9.1 GB image pack),
# MMSI-Bench and MindCube.
set -euo pipefail; cd "$(dirname "$0")/.."; source .venv/bin/activate; mkdir -p data
hf download iamdanialkamali/3D-FORCE-Zip --repo-type dataset --local-dir data/3d-force
[ -d data/3d-force/multiview ] || unzip -q data/3d-force/multiview_json_jpg_only.zip -d data/3d-force/
hf download RunsenXu/MMSI-Bench --repo-type dataset --local-dir data/mmsi
hf download MLL-Lab/MindCube    --repo-type dataset --local-dir data/mindcube
echo "datasets ready under data/"
