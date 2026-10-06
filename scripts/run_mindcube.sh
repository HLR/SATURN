#!/usr/bin/env bash
# MindCube, one slice per call:   scripts/run_mindcube.sh {among|around|rotation} [seed 0|1] [extra saturn.cli args]
# Plans every question and writes its program through the code LLM (the seed is the code LLM's).
# Configuration: configs/benchmarks.json.
set -euo pipefail; cd "$(dirname "$0")/.."; source .venv/bin/activate; set -a; [ -f .env ] && source .env; set +a
SLICE="${1:?among|around|rotation}"; shift
SEED=0; if [[ "${1:-}" =~ ^[0-9]+$ ]]; then SEED=$1; shift; fi
export SAPY_CODEGEN_SEED="$SEED"; MODE=unified   # planner prompt tag in cache and result names
python -m saturn.cli --dataset "mindcube-${SLICE}" \
  --planner_cache "cache/planner_${MODE}_mindcube-${SLICE}_s${SEED}.json" \
  --vlm_base_url "http://127.0.0.1:${SAPY_VLM_PORT:-8100}/v1" --vlm_max_concurrency 32 \
  --results_save_file "mindcube_${SLICE}_s${SEED}_${MODE}_$(date +%Y%m%d_%H%M%S).json" \
  --num_samples 1000000 --max_concurrent_samples 4 --no-write_program_cache "$@"
