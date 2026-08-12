# CGLF

Continuous Gaussian Light Field (CGLF) is a two-stage anchor-based Gaussian
super-resolution pipeline built on top of Scaffold-GS.

This repository contains:

- a Stage-1 anchor-based Gaussian backbone trainer with optional EPI
  consistency loss;
- a Stage-2 LIIF-style continuous feature field trainer for light-field /
  multi-view super-resolution;
- rendering and evaluation utilities for both Stage-1 and Stage-2 checkpoints;
- helper tools for depth-based point-cloud initialization and LIIF field
  distillation.

The codebase is based on Scaffold-GS and keeps the upstream rendering /
representation modules, while adding the CGLF Stage-2 training pipeline and
light-field-specific utilities.

## Repository layout

```text
CGLF/
|- arguments/                       # shared CLI parameter definitions
|- assets/                          # paper / repo figures
|- gaussian_renderer/               # rendering backend
|- lpipsPyTorch/                    # LPIPS fallback implementation
|- scene/                           # scene loading, Gaussian model, LIIF field
|- submodules/                      # diff-gaussian-rasterization, simple-knn
|- tools/
|  |- create_depth_init_ply.py      # build init point cloud from per-view depth
|  |- distill_stage1_liif_field.py  # optional stage-1 field distillation
|  `- ...
|- scripts/
|  |- train_stage1_epi_example.sh
|  |- train_stage2_liif_example.sh
|  `- render_stage2_checkpoint_example.sh
|- train.py                         # Stage-1 training
|- train_stage2_supergs_liif.py     # Main Stage-2 CGLF training
|- train_stage2_supergs_strict.py   # Hash / strict Stage-2 baseline
|- train_stage2_supergs_liif_gradgrow.py
|- render.py                        # Stage-1 rendering
|- render_stage2_checkpoint.py      # Stage-2 rendering + metric recomputation
|- metrics.py                       # image metric computation
|- environment.yml                  # conda environment
`- build_scaffold_extensions.cmd    # Windows extension build helper
```

## Requirements

Tested configuration:

- Python `3.7.13`
- PyTorch `1.12.1`
- CUDA `11.6`

The default conda environment is defined in `environment.yml`.

## Setup

### 1. Clone with submodules

```bash
git clone --recursive <your-github-url> CGLF
cd CGLF
```

If you already cloned the repository without submodules:

```bash
git submodule update --init --recursive
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate scaffold_gs
```

The environment file installs the CUDA extensions from:

- `submodules/diff-gaussian-rasterization`
- `submodules/simple-knn`

### 3. Windows fallback for extension build

If editable installation of the submodules fails on Windows, run:

```bat
build_scaffold_extensions.cmd
```

This script is repo-relative and rebuilds both CUDA extensions inside the
currently activated `scaffold_gs` environment.

## Data format

### Stage-1 root

Stage-1 expects a NeRF-style scene root:

```text
scene_root/
|- images/
|- transforms_train.json
|- transforms_test.json
`- points3d.ply                  # optional if provided via --init_ply_path
```

Optional depth supervision inputs can also exist under directories such as:

```text
scene_root/
`- depth/
```

### Stage-2 root

Stage-2 typically uses a pseudo-HR root with the same split format:

```text
scene_pseudo_root/
|- images/
|- transforms_train.json
|- transforms_test.json
`- depth/                        # optional, used for some ablations / growth
```

You can also provide an HR evaluation root through `--eval_source`.

## Stage-1 training

Main entry point:

```bash
python train.py \
  -s <stage1_scene_root> \
  -m <stage1_output_dir> \
  --eval \
  --init_ply_path <init_points3d.ply> \
  --iterations 30000 \
  --save_iterations 30000 \
  --test_iterations 30000
```

### Stage-1 with EPI consistency

This is the CGLF light-field version used in our experiments:

```bash
python train.py \
  -s <stage1_scene_root> \
  -m <stage1_output_dir> \
  --eval \
  --init_ply_path <init_points3d.ply> \
  --iterations 30000 \
  --save_iterations 30000 \
  --test_iterations 30000 \
  --epi_loss_weight 0.2 \
  --epi_loss_interval 10 \
  --epi_num_views 3 \
  --epi_num_lines 4 \
  --epi_spatial_stride 1
```

Important notes:

- `--init_ply_path` lets you override the default `points3d.ply`.
- The Stage-1 trainer automatically runs final rendering and metric evaluation
  after training.

## Stage-2 training

Main CGLF Stage-2 entry point:

```bash
python train_stage2_supergs_liif.py \
  -s <stage2_pseudo_root> \
  -m <stage1_output_dir> \
  --stage2_output <stage2_output_dir> \
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
  --eval_source <hr_eval_root>
```

Outputs written to `--stage2_output`:

- `stage2_args.json`
- `stage2_<iter>.pth`
- `metrics_history.jsonl`
- `summary.json`

### Optional Stage-2 variants

- `train_stage2_supergs_strict.py`: hash / strict Stage-2 baseline
- `train_stage2_supergs_liif_gradgrow.py`: LIIF + gradient-growth ablation

## Rendering and evaluation

### Stage-1 rendering / metrics

```bash
python render.py -m <stage1_output_dir>
python metrics.py -m <stage1_output_dir>
```

### Stage-2 checkpoint rendering

This utility restores a Stage-2 checkpoint and writes rendered images plus
recomputed metrics:

```bash
python render_stage2_checkpoint.py \
  -s <eval_scene_root> \
  -m <stage1_output_dir> \
  --stage2_checkpoint <stage2_output_dir/stage2_5000.pth> \
  --stage2_mode liif \
  --stage1_iteration 30000 \
  --out_method_name ours_5000 \
  --max_views 144
```

The recomputed metrics are written to:

```text
<stage1_output_dir>/test/<out_method_name>/recomputed_metrics.json
```

## Useful tools

### Depth-based point-cloud initialization

```bash
python tools/create_depth_init_ply.py \
  --source_path <scene_root> \
  --output_ply <output_points3d.ply> \
  --depth_dir_name depth \
  --stride 4 \
  --max_points_per_view 10000
```

### Distill a LIIF field from a frozen Stage-1 model

```bash
python tools/distill_stage1_liif_field.py \
  -s <stage1_scene_root> \
  -m <stage1_output_dir> \
  --stage1_iteration 30000 \
  --output <distill_output_dir>
```

## Suggested public release checklist

Before pushing this repository to GitHub:

1. Make sure no local experiment outputs are inside the repo.
2. Keep submodules either committed as git submodules or vendor them
   intentionally.
3. Verify your license choice for any newly added paper-specific assets.
4. If you publish trained weights, store them outside the repo and link them
   from the README.

## License

This repository keeps the upstream `LICENSE.md`. If you intend to publish a
different project-level license, review compatibility with Scaffold-GS and the
included submodules before changing it.
