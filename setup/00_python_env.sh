#!/usr/bin/env bash
# Create .venv (SATURN + evaluation) and .vllm (the VLM server). Python 3.10, CUDA 12.4.
set -euo pipefail; cd "$(dirname "$0")/.."
command -v uv >/dev/null || pip install uv
uv venv --python 3.10 .venv;  uv pip install --python .venv/bin/python -r requirements.txt
uv venv --python 3.10 .vllm;  uv pip install --python .vllm/bin/python -r requirements.vllm.txt
echo "envs ready: .venv (run benchmarks) and .vllm (serve the VLM)"
