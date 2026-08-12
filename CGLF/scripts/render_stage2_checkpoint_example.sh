#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT=${EVAL_ROOT:-/path/to/scene_hrgt}
STAGE1_DIR=${STAGE1_DIR:-/path/to/output/stage1_epi}
STAGE2_CKPT=${STAGE2_CKPT:-/path/to/output/stage2_liif/stage2_5000.pth}

python render_stage2_checkpoint.py \
  -s "${EVAL_ROOT}" \
  -m "${STAGE1_DIR}" \
  --stage2_checkpoint "${STAGE2_CKPT}" \
  --stage2_mode liif \
  --stage1_iteration 30000 \
  --out_method_name ours_5000 \
  --max_views 144
