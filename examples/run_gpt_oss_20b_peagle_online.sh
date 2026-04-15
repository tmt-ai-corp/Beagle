#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python -m sglang.launch_server \
  --model-path openai/gpt-oss-20b \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path cache/peagle/GPT-OSS-20B-P-EAGLE \
  --speculative-num-steps 7 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 8 \
  --disable-overlap-schedule \
  --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 1 \
  --tp-size 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16
