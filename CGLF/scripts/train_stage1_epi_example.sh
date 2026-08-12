#!/usr/bin/env bash
set -euo pipefail

SCENE_ROOT=${SCENE_ROOT:-/path/to/scene_x2}
INIT_PLY=${INIT_PLY:-$SCENE_ROOT/points3d.ply}
OUT_DIR=${OUT_DIR:-/path/to/output/stage1_epi}

python train.py \
  -s "${SCENE_ROOT}" \
  -m "${OUT_DIR}" \
  --eval \
  --init_ply_path "${INIT_PLY}" \
  --iterations 30000 \
  --save_iterations 30000 \
  --test_iterations 30000 \
  --epi_loss_weight 0.2 \
  --epi_loss_interval 10 \
  --epi_num_views 3 \
  --epi_num_lines 4 \
  --epi_spatial_stride 1
