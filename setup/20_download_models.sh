#!/usr/bin/env bash
# Model weights. SAM3 is gated: accept the license at https://huggingface.co/facebook/sam3 and `hf auth login` first.
set -euo pipefail; cd "$(dirname "$0")/.."; source .venv/bin/activate
hf download Viglong/OriAnyV2_ckpt   --local-dir tools/models--Viglong--OriAnyV2_ckpt
hf download facebook/dinov2-large   --local-dir tools/models--facebook--dinov2-large
hf download facebook/VGGT-1B        --local-dir tools/models--facebook--VGGT-1B
hf download facebook/sam3           --local-dir tools/sam3/checkpoints || echo "SAM3 download failed: accept the license + hf auth login"
hf download Qwen/Qwen3-VL-8B-Instruct   # served by scripts/serve_vlm.sh
echo "models ready"
