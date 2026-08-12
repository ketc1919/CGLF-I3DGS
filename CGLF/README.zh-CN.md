# CGLF

Continuous Gaussian Light Field（CGLF）是一个基于 Scaffold-GS 构建的两阶段、基于 anchor 的 Gaussian 超分辨率流程。

本仓库包含：

- Stage-1：基于 anchor 的 Gaussian 主干训练器，可选 EPI 一致性损失；
- Stage-2：用于光场 / 多视角超分辨率的 LIIF 风格连续特征场训练器；
- Stage-1 和 Stage-2 checkpoint 的渲染与评估工具；
- 基于深度的点云初始化和 LIIF field 蒸馏辅助工具。

代码基于 Scaffold-GS，保留了上游的渲染 / 表示模块，同时加入了 CGLF 的 Stage-2 训练流程和光场相关工具。

## 仓库结构

```text
CGLF/
|- arguments/                       # 共享 CLI 参数定义
|- assets/                          # 论文 / 仓库图示
|- gaussian_renderer/               # 渲染后端
|- lpipsPyTorch/                    # LPIPS 备用实现
|- scene/                           # 场景加载、Gaussian 模型、LIIF field
|- submodules/                      # diff-gaussian-rasterization, simple-knn
|- tools/
|  |- create_depth_init_ply.py      # 从逐视角深度构建初始化点云
|  |- distill_stage1_liif_field.py  # 可选的 Stage-1 field 蒸馏
|  `- ...
|- scripts/
|  |- train_stage1_epi_example.sh
|  |- train_stage2_liif_example.sh
|  `- render_stage2_checkpoint_example.sh
|- train.py                         # Stage-1 训练
|- train_stage2_supergs_liif.py     # 主要的 Stage-2 CGLF 训练
|- train_stage2_supergs_strict.py   # Hash / strict Stage-2 baseline
|- train_stage2_supergs_liif_gradgrow.py
|- render.py                        # Stage-1 渲染
|- render_stage2_checkpoint.py      # Stage-2 渲染 + 指标重新计算
|- metrics.py                       # 图像指标计算
|- environment.yml                  # conda 环境
`- build_scaffold_extensions.cmd    # Windows 扩展构建辅助脚本
```

## 环境要求

测试过的配置：

- Python `3.7.13`
- PyTorch `1.12.1`
- CUDA `11.6`

默认 conda 环境定义在 `environment.yml` 中。

## 安装

### 1. 克隆仓库和子模块

```bash
git clone --recursive <your-github-url> CGLF
cd CGLF
```

如果你已经克隆了仓库但没有拉取子模块：

```bash
git submodule update --init --recursive
```

### 2. 创建 conda 环境

```bash
conda env create -f environment.yml
conda activate scaffold_gs
```

环境文件会从以下路径安装 CUDA 扩展：

- `submodules/diff-gaussian-rasterization`
- `submodules/simple-knn`

### 3. Windows 下扩展构建的备用方式

如果在 Windows 上子模块的 editable 安装失败，运行：

```bat
build_scaffold_extensions.cmd
```

这个脚本使用仓库相对路径，并会在当前激活的 `scaffold_gs` 环境中重新构建两个 CUDA 扩展。

## 数据格式

### Stage-1 根目录

Stage-1 需要 NeRF 风格的场景根目录：

```text
scene_root/
|- images/
|- transforms_train.json
|- transforms_test.json
`- points3d.ply                  # 如果通过 --init_ply_path 提供，则此项可选
```

可选的深度监督输入也可以放在如下目录：

```text
scene_root/
`- depth/
```

### Stage-2 根目录

Stage-2 通常使用 pseudo-HR 根目录，split 格式相同：

```text
scene_pseudo_root/
|- images/
|- transforms_train.json
|- transforms_test.json
`- depth/                        # 可选，用于部分消融 / anchor growth
```

也可以通过 `--eval_source` 提供 HR 评估根目录。

## Stage-1 训练

主入口：

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

### 带 EPI 一致性的 Stage-1

这是我们实验中使用的 CGLF 光场版本：

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

重要说明：

- `--init_ply_path` 可以覆盖默认的 `points3d.ply`。
- Stage-1 训练器会在训练结束后自动执行最终渲染和指标评估。

## Stage-2 训练

主要的 CGLF Stage-2 入口：

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

写入 `--stage2_output` 的输出包括：

- `stage2_args.json`
- `stage2_<iter>.pth`
- `metrics_history.jsonl`
- `summary.json`

### 可选 Stage-2 变体

- `train_stage2_supergs_strict.py`：hash / strict Stage-2 baseline
- `train_stage2_supergs_liif_gradgrow.py`：LIIF + gradient-growth 消融

## 渲染和评估

### Stage-1 渲染 / 指标

```bash
python render.py -m <stage1_output_dir>
python metrics.py -m <stage1_output_dir>
```

### Stage-2 checkpoint 渲染

该工具会恢复 Stage-2 checkpoint，并写出渲染图像和重新计算的指标：

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

重新计算的指标会写入：

```text
<stage1_output_dir>/test/<out_method_name>/recomputed_metrics.json
```

## 实用工具

### 基于深度的点云初始化

```bash
python tools/create_depth_init_ply.py \
  --source_path <scene_root> \
  --output_ply <output_points3d.ply> \
  --depth_dir_name depth \
  --stride 4 \
  --max_points_per_view 10000
```

### 从冻结的 Stage-1 模型蒸馏 LIIF field

```bash
python tools/distill_stage1_liif_field.py \
  -s <stage1_scene_root> \
  -m <stage1_output_dir> \
  --stage1_iteration 30000 \
  --output <distill_output_dir>
```

## 建议的公开发布检查清单

在把仓库推送到 GitHub 前：

1. 确认仓库中没有本地实验输出。
2. 子模块要么作为 git submodule 提交，要么明确地以内置代码形式保留。
3. 确认所有新增的论文相关资源所使用的许可证。
4. 如果发布训练好的权重，请将其存放在仓库外部，并从 README 中链接。

## 许可证

本仓库保留上游的 `LICENSE.md`。如果你打算发布不同的项目级许可证，请在修改前检查它与 Scaffold-GS 以及所包含子模块的兼容性。
