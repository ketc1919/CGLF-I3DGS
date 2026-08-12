#!/usr/bin/env bash
set -euo pipefail

PSEUDO_ROOT=${PSEUDO_ROOT:-/path/to/scene_x2_pseudo}
HR_EVAL_ROOT=${HR_EVAL_ROOT:-/path/to/scene_hrgt}
STAGE1_DIR=${STAGE1_DIR:-/path/to/output/stage1_epi}
STAGE2_OUT=${STAGE2_OUT:-/path/to/output/stage2_liif}

python train_stage2_supergs_liif.py \
  -s "${PSEUDO_ROOT}" \
  -m "${STAGE1_DIR}" \
  --stage2_output "${STAGE2_OUT}" \
  --stage1_iteration 30000 \
  --iterations 5000 \
  --eval_interval 1000 \
  --save_interval 1000 \
  --stage2_feature_lr 1e-3 \
  --decoder_lr 3e-5 \
  --liif_lr 1e-3 \
  --liif_hidden_dim 64 \
  --liif_k 4 \
  --liif_k_render 1 \
  --liif_temperature 0.05 \
  --liif_knn_chunk_size 8192 \
  --depth_dir_name depth \
  --growth_interval 100 \
  --densify_until 2000 \
  --refine_iters 1000 \
  --uncertainty_threshold 0.02 \
  --refine_max_new 1024 \
  --lambda_ssim 0.2 \
  --lambda_vol 0.01 \
  --lambda_lpips 0.05 \
  --vote_voxel_size 0.03 \
  --max_candidates_per_view 1500 \
  --error_threshold 0.1 \
  --vote_threshold 3 \
  --max_new_anchors 1000 \
  --eval_source "${HR_EVAL_ROOT}"
