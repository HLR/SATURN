#!/usr/bin/env bash
# 3D-FORCE, non-GT (SAM3 detection + multi-view fusion):   scripts/run_force3d.sh {ref|puzzle} [seed 0|1] [extra saturn.cli args]
#   ref = REF task, puzzle = SAG task. Plans every question with the unified prompt and writes its program
# through the code LLM (the seed is the code LLM's). Configuration: configs/benchmarks.json.
set -euo pipefail; cd "$(dirname "$0")/.."; source .venv/bin/activate; set -a; [ -f .env ] && source .env; set +a
TASK="${1:?ref|puzzle}"; shift
case "$TASK" in ref|puzzle) ;; *) echo "usage: $0 {ref|puzzle} [seed]" >&2; exit 2 ;; esac
SEED=0; if [[ "${1:-}" =~ ^[0-9]+$ ]]; then SEED=$1; shift; fi
export SAPY_CODEGEN_SEED="$SEED"; MODE=unified   # planner prompt tag in cache and result names
python -m saturn.cli --dataset "force3d-${TASK}" \
  --planner_cache "cache/planner_${MODE}_force3d-${TASK}_s${SEED}.json" \
  --vlm_base_url "http://127.0.0.1:${SAPY_VLM_PORT:-8100}/v1" --vlm_max_concurrency 16 \
  --results_save_file "force3d_${TASK}_nogt_s${SEED}_${MODE}_$(date +%Y%m%d_%H%M%S).json" \
  --num_workers 1 --max_concurrent_samples 4 --exec_concurrency 4 --no-write_program_cache "$@"
