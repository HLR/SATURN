#!/usr/bin/env bash
# MMSI-Bench, all 1000 questions:   scripts/run_mmsi.sh [seed 0|1] [extra saturn.cli args]
# Plans every question and writes its program through the code LLM (the seed is the code LLM's).
# Configuration: configs/benchmarks.json.
set -euo pipefail; cd "$(dirname "$0")/.."; source .venv/bin/activate; set -a; [ -f .env ] && source .env; set +a
SEED=0; if [[ "${1:-}" =~ ^[0-9]+$ ]]; then SEED=$1; shift; fi
export SAPY_CODEGEN_SEED="$SEED"; MODE=unified   # planner prompt tag in cache and result names
python -m saturn.cli --dataset mmsi --planner_cache "cache/planner_${MODE}_mmsi_s${SEED}.json" \
  --vlm_base_url "http://127.0.0.1:${SAPY_VLM_PORT:-8100}/v1" --vlm_max_concurrency 32 \
  --results_save_file "mmsi_s${SEED}_${MODE}_$(date +%Y%m%d_%H%M%S).json" \
  --num_samples 1000000 --max_concurrent_samples 4 --no-write_program_cache "$@"
