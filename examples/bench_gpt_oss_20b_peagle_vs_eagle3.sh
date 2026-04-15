#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python benchmarks/compare_peagle_vs_eagle3.py \
  --model-path openai/gpt-oss-20b \
  --eagle3-draft-model-path zhuyksir/EAGLE3-gpt-oss-20b-bf16 \
  --peagle-draft-model-path cache/peagle/GPT-OSS-20B-P-EAGLE \
  --eagle3-config-list 1,3,1,4 1,5,1,6 1,7,1,8 \
  --peagle-config-list 1,3,1,4 1,5,1,6 1,7,1,8 \
  --benchmark-list mtbench:80 humaneval:164 \
  --dtype bfloat16 \
  --mem-fraction-static 0.8 \
  --tp-size 1
