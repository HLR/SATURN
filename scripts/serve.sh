#!/usr/bin/env bash
# Start the perception services (VGGT, SAM3, Orient-Anything) on Ray Serve, port 8011.
# GPUS: cards for the services (2 recommended). Orient-Anything is the highest-call-rate
# service, so it runs several replicas (SAPY_ORIANY_MIN_REPLICAS / SAPY_ORIANY_MAX_REPLICAS).
set -uo pipefail; cd "$(dirname "$0")/.."; source .venv/bin/activate
GPUS="${GPUS:-0,1}"; export SAPY_SERVE_HTTP_PORT="${SAPY_SERVE_HTTP_PORT:-8011}"
ray stop --force >/dev/null 2>&1; sleep 3
setsid env CUDA_VISIBLE_DEVICES="$GPUS" ray start --head --disable-usage-stats >/dev/null 2>&1; sleep 8
CUDA_VISIBLE_DEVICES="$GPUS" SAPY_ENABLE_VGGT=1 SAPY_ENABLE_SAM3=1 \
SAPY_ORIANY_MIN_REPLICAS="${SAPY_ORIANY_MIN_REPLICAS:-4}" SAPY_ORIANY_MAX_REPLICAS="${SAPY_ORIANY_MAX_REPLICAS:-4}" \
SAPY_SAM3_MIN_REPLICAS=1 SAPY_SAM3_MAX_REPLICAS=1 \
SAPY_VGGT_NUM_GPUS=0.5 SAPY_SAM3_NUM_GPUS=0.5 SAPY_ORIANY_NUM_GPUS=0.25 \
  python -m saturn.serving.deploy
