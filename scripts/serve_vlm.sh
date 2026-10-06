#!/usr/bin/env bash
# Serve the scoring VLM with vLLM on port 8100 (default: Qwen3-VL-8B-Instruct).
set -euo pipefail; cd "$(dirname "$0")/.."; source .vllm/bin/activate
MODEL="${SAPY_VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"; PORT="${SAPY_VLM_PORT:-8100}"; TP="${TP:-1}"
CUDA_VISIBLE_DEVICES="${GPUS:-2}" vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size "$TP" \
  --max-model-len 32768 --limit-mm-per-prompt '{"image":16}' --gpu-memory-utilization 0.9
