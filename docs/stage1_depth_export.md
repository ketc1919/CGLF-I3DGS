# Stage-1 最终深度导出技术文档

本文记录当前工作区中“Stage-1 最终深度导出”改动的实际行为。文档对应的仓库是
`C:\Users\New\Documents\CGLF-I3DGS-stage1-refactor`，目标分支为
`refactor/i3dgs-scaffold-stage1`。本文只描述 I3DGS、适配器和复合 Stage-1
编排入口的代码；没有修改 CGLF 的训练逻辑，也没有把深度导出描述成已经完成的
Stage-2 深度投票验证。

## 1. 背景、目标和边界

现有 Stage-1 先由 I3DGS 从无序图像估计最终相机状态和 BA landmarks，再通过
`--cglf_scene_path` 导出 CGLF 可读取的场景。此次改动在这个“最终导出”边界上增加
可选的深度输出：Depth Anything V2 给出密集的相对逆深度，最终 BA landmarks 的
二维观测和相机坐标系 Z 给出尺度/偏移约束，最后把对齐后的逆深度重采样到导出 RGB
图像尺寸并转换为相机 Z 深度。

本改动明确不做以下事情：

- 不重新估计相机位姿，不修改 BA 表，不把深度反向写回 I3DGS 优化；
- 不使用高斯渲染深度替代 BA 对齐深度；
- 不修改 `CGLF/train.py` 或 CGLF Stage-2 训练逻辑；
- 不在 `--dry_run` 中加载 DA V2，不创建训练输出；
- 不宣称深度单位为米。深度使用 BA/I3DGS 的任意场景单位。

## 2. 当前改动清单

以下是当前工作区相对 `HEAD` 的实际改动。`README.md` 和 `SNAPSHOT.md` 没有被本次
任务覆盖。行号以当前工作区文件为准，查看时可用编辑器跳转。

| 文件 | 主要变化 |
| --- | --- |
| `i3dgs/depth_export.py` | 新增深度导出模块：读取最终 BA 有效观测，计算相机 Z，调用已初始化的 DA V2，完成逆深度对齐、像素重采样、`.npy` 和 `depth_stats.json` 写入。 |
| `i3dgs/args.py:263-266` | 新增 `--export_depth`，`store_true`，默认 `False`；要求同时提供 `--cglf_scene_path`。 |
| `i3dgs/cglf_export.py:349-450` | `export_cglf_scene()` 新增 `export_depth=False`；深度写入适配器 staging 目录，深度成功后才进入现有原子发布。 |
| `i3dgs/train.py:49-55,721-729` | 在任务初始化时检查参数组合；最终保存重建后把 `export_depth` 传给适配器。 |
| `i3dgs/scene/mono_depth.py:187-194` | 现有分块对齐的鲁棒筛选使用有限值和 `<=`，允许零残差块保留一致样本。 |
| `pipeline/run_stage1.py:431-516` | `validate_exported_scene()` 增加深度目录、统计文件、逐注册图像 `.npy` 的结构、dtype、维度和数值检查。 |
| `pipeline/run_stage1.py:564-621,704-755,797-906` | 增加顶层 `--export_depth`，只传给 I3DGS；manifest 记录该开关；普通模式和 skip 模式分别验证深度。 |
| `tests/test_depth_export.py` | 新增合成 BA、关键帧和 mock 深度估计器测试，不加载真实模型或启动 GPU。 |
| `tests/test_stage1_pipeline.py:183-274` | 增加 skip 深度门禁和 `--export_depth` 只进入 I3DGS 命令的测试。 |

当前分支的 `HEAD` 是 `8dc91e2 Disable xFormers FA3 for SM120 mono depth initialization`；
该 SM120 兼容性改动已经在提交中，不属于本次未提交补丁。本文将它视为当前代码基线，
不把它列为本次文档对应的工作区改动。

## 3. 数据流

正常流程的实际顺序如下：

```text
输入图像目录
    │
    ▼
I3DGS：VPR/匹配、位姿估计、BA、最终 SceneModel 保存
    │  SceneModel 仍持有 ba_problem、最终 keyframes、depth_estimator
    ▼
适配器 preflight + staging 目录
    ├─ 复制已注册原图
    ├─ 复制 cameras.bin / images.bin
    ├─ 过滤并写 points3D.ply
    └─ --export_depth 开启时：
         BA 对应关系 → 相机 Z → DA V2 相对逆深度 → 尺度/偏移对齐
         → 映射到导出 RGB 尺寸 → depth/*.npy + depth_stats.json
    │
    ▼
所有必需文件成功后，staging 目录原子 rename 为 exported_scene
    │
    ▼
复合 pipeline 验证 exported_scene
    │
    ▼
Scaffold-GS 读取 exported_scene；`--eval` 仍只传给 Scaffold-GS
```

`SceneModel` 的深度估计器只在 I3DGS 初始化阶段建立；`depth_export.py` 通过
`scene_model.depth_estimator` 复用它，不重复加载 DA V2。深度导出发生在最终
`scene_model.save()` 之后，因此使用的是最终 BA/keyframe 状态，而不是中间 checkpoint。

## 4. BA 对应关系与几何定义

`collect_ba_correspondences()` 从 `scene_model.ba_problem` 读取预分配数组的活动范围：

```text
landmarks[:size]
n_obs[:size]
obs_lm_ids[:obs_size]
obs_kf_ids[:obs_size]
obs_pt2d_ids[:obs_size]
obs_uvs[:obs_size]
```

它显式按最终关键帧 `keyframe.index` 建立对应关系，并过滤：

- landmark、关键帧和 2D 点索引越界；
- `n_obs < 2`；
- XYZ 或 UV 非有限；
- UV 超出当前处理图像范围；
- 相机坐标 Z 非正；
- 使用最终相机重投影后误差大于当前默认阈值 `3.0` 像素。

同一 landmark 在同一关键帧出现多次时，当前实现采用按 observation 表顺序保留第一条
可用观测的确定性规则。`n_obs` 只作为观测次数门槛，不被解释成视角数量或最终内点标志。

对世界坐标点 `X_world`，代码使用：

```text
X_camera = X_world @ R.T + t
Z = X_camera[:, 2]
target_inverse_depth = 1 / Z
```

这里的 `Z` 是相机坐标系前向深度，单位继承 I3DGS/BA 场景的任意尺度；它不是米，也不是
用于显示的 0～1 归一化深度。

## 5. DA V2 对齐和像素映射

### 5.1 预测、对齐、输出三种数值

代码区分以下三个量：

1. **原始预测**：DA V2 返回的 2D 相对逆深度-like 数组，形状由估计器决定；
2. **对齐逆深度**：在 BA 观测位置采样原始预测，用 `1/Z` 作为目标，拟合
   `aligned = scale * raw + offset`；
3. **最终 Z 深度**：把目标 RGB 网格上的对齐逆深度取倒数，正值写出，其他位置写 `0`。

当前全局拟合的默认条件是至少 `8` 个有效对应点，离群筛选乘数默认 `5.0`。常量预测、
点数不足、拟合参数非有限或对齐后没有正值时抛出 `DepthExportError`，不静默填充常数
深度。若 `scene_model.args.depth_grid_size > 0`，实现会继续调用现有
`scene.mono_depth.align_depth_grid()` 做分块对齐；合成测试没有启用 CUDA 分块路径。

### 5.2 处理图像到导出 RGB 的映射

当前首版按居中针孔、纯缩放关系处理像素映射。目标图像像素中心映射到处理图像为：

```text
u_processed = (u_target + 0.5) * W_processed / W_target - 0.5
v_processed = (v_target + 0.5) * H_processed / H_target - 0.5
```

随后按 `align_corners` 对应的端点关系映射到 DA V2 预测网格并双线性采样。改变输出
分辨率只改变数组网格，不对 Z 数值乘缩放比例。当前代码读取注册原图的真实宽高，输出
二维 `float32 [H,W]`，无效值固定为 `0`。

不支持的裁剪、旋转或其他相机模型没有被扩展；真实数据若不满足纯缩放假设，需要单独
验证，而不能仅凭一次 resize 认为坐标已经正确。

## 6. 输出接口和目录结构

启用 `--export_depth` 后，导出的场景结构为：

```text
exported_scene/
├── export_stats.json
├── depth_stats.json
├── images/
│   └── 仅成功注册且被复制的原图
├── depth/
│   └── <Stage-2 实际 camera.image_name 按第一个句点截断>.npy
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.ply
```

例如 `scene.001.jpg` 当前映射为 `depth/scene.npy`，这是代码中的
`name.split('.', 1)[0]` 约定，不是 `Path.stem`。导出器会拒绝映射后的文件名冲突。

每个 `.npy` 必须满足：

- dtype 为 `float32`；
- shape 为二维 `[H,W]`，与对应导出 RGB 相同；
- 有效值有限且为正；
- 无效值为 `0`；
- 数值含义是相机坐标系 Z，不做显示归一化。

`depth_stats.json` 记录来源、Z 定义、场景单位、图像/深度文件映射、处理/预测/目标尺寸、
观测过滤统计、对齐模式和参数、有效像素比例、深度分位数以及失败/回退信息。

## 7. 安装、参数和调用方式

### 7.1 安装前提

本次改动没有新增独立的 Conda 环境文件；它复用仓库已有的 I3DGS 和 Scaffold-GS
环境。下面是仓库 README 中的最小安装骨架，版本和 CUDA wheel 应按目标 GPU/驱动调整。
安装命令只准备环境，不运行训练，也不证明真实深度导出已经成功。

```bash
# 工作目录：/root/autodl-tmp/CGLF-I3DGS/i3dgs
# 创建 I3DGS 环境；Python、PyTorch、CUDA wheel 必须与实际 GPU/驱动匹配。
cd /root/autodl-tmp/CGLF-I3DGS/i3dgs
conda create -n i3dgs python=3.12 -y
conda activate i3dgs
pip install hatchling
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt --no-build-isolation
pip install cupy-cuda12x
```

```bash
# 工作目录：/root/autodl-tmp/CGLF-I3DGS/CGLF 或 Scaffold-GS 子模块目录。
# 使用 CGLF README 提供的环境文件安装 Scaffold-GS 依赖和 CUDA 扩展。
cd /root/autodl-tmp/CGLF-I3DGS/CGLF
conda env create -f environment.yml
conda activate scaffold_gs
```

当前 AutoDL 实验使用的是独立的 `scaffold-cu128` 环境，而不是上面示例中的默认
`scaffold_gs` 名称；复合 pipeline 通过 `--scaffold_python` 明确指定解释器。若扩展
已经在目标环境中编译完成，不要仅凭安装成功推断深度导出路径可用，仍需执行合成测试
和一次真实小场景验证。

### 7.2 I3DGS 直接调用

`i3dgs/args.py` 中新增参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--export_depth` | `False` | 开启最终 BA 对齐相机 Z 深度导出。必须同时设置 `--cglf_scene_path`。 |
| `--cglf_scene_path` | 空字符串 | 现有适配器输出目录；深度写到其 `depth/`。 |

在 AutoDL 的 I3DGS 环境中，从仓库的 `i3dgs` 工作目录运行：

```bash
# 工作目录：/root/autodl-tmp/CGLF-I3DGS/i3dgs
# 环境：i3dgs；--source_path 是无序输入图像场景，--model_path 保存 I3DGS 结果，
# --cglf_scene_path 是新建的 CGLF 场景目录，--export_depth 开启深度导出。
# 预期输出：model_path 下的 I3DGS 重建，以及 cglf_scene_path/depth/*.npy 和 depth_stats.json。
cd /root/autodl-tmp/CGLF-I3DGS/i3dgs
conda activate /root/autodl-tmp/conda-envs/i3dgs
python train.py \
  --source_path /root/autodl-tmp/datasets/MipNeRF360_without_pose/garden \
  --model_path /root/autodl-tmp/i3dgs-runs/garden_depth_v1 \
  --cglf_scene_path /root/autodl-tmp/exported-scenes/garden_depth_v1 \
  --export_depth
```

`cglf_scene_path` 必须是尚不存在的输出目录，且不能位于输入图像目录内部；适配器的
现有 preflight、符号链接保护和原子发布仍然生效。

### 7.3 复合 Stage-1 调用

顶层 `pipeline/run_stage1.py` 新增同名 `--export_depth`，默认关闭，并且**只**追加到
I3DGS 参数列表，不传给 Scaffold-GS。顶层 `--iterations` 仍只控制 Scaffold-GS；
若需要单独控制 I3DGS，使用已有的 `--i3dgs_num_iterations`。

```bash
# 工作目录：/root/autodl-tmp/CGLF-I3DGS
# 环境：i3dgs；pipeline 会分别用 i3dgs_python 和 scaffold_python 启动两个子进程。
# --eval 只传给 Scaffold-GS；--export_depth 只传给 I3DGS。
# 预期输出：I3DGS/适配器日志、exported_scene（含 depth/），以及 Scaffold-GS checkpoint。
cd /root/autodl-tmp/CGLF-I3DGS
conda activate /root/autodl-tmp/conda-envs/i3dgs
python pipeline/run_stage1.py \
  --input_scene /root/autodl-tmp/datasets/MipNeRF360_without_pose/garden \
  --i3dgs_output /root/autodl-tmp/i3dgs-runs/garden_depth_pipeline_v1 \
  --exported_scene /root/autodl-tmp/exported-scenes/garden_depth_pipeline_v1 \
  --scaffold_output /root/autodl-tmp/scaffold-runs/garden_depth_pipeline_v1 \
  --i3dgs_python /root/autodl-tmp/conda-envs/i3dgs/bin/python \
  --scaffold_python /root/autodl-tmp/conda-envs/scaffold-cu128/bin/python \
  --iterations 30000 \
  --voxel_size 0.001 \
  --appearance_dim 0 \
  --eval \
  --export_depth
```

日志默认分别写到：

```text
/root/autodl-tmp/i3dgs-runs/garden_depth_pipeline_v1.i3dgs.log
/root/autodl-tmp/scaffold-runs/garden_depth_pipeline_v1.scaffold-gs.log
```

完成后，manifest 固定写入：

```text
/root/autodl-tmp/scaffold-runs/garden_depth_pipeline_v1/stage1_manifest.json
```

其中包含输入/输出路径、实际参数数组、迭代数、voxel size、appearance dim、`eval`、
`export_depth`、I3DGS Git SHA 和 Scaffold-GS Git SHA。

### 7.4 已有场景的 skip 检查

`--skip_i3dgs --export_depth` 不会生成深度，只会要求已有 `exported_scene` 的每个
注册图像都已经存在合法 `.npy` 和 `depth_stats.json`；否则在启动 Scaffold-GS 前失败。

```bash
# 工作目录：/root/autodl-tmp/CGLF-I3DGS
# 环境：i3dgs；复用已经完成并验证的 exported_scene，不重新运行 I3DGS。
# 预期输出：只运行 Scaffold-GS；若任何深度文件缺失，命令会在训练前明确报错。
cd /root/autodl-tmp/CGLF-I3DGS
conda activate /root/autodl-tmp/conda-envs/i3dgs
python pipeline/run_stage1.py \
  --input_scene /root/autodl-tmp/datasets/MipNeRF360_without_pose/garden \
  --i3dgs_output /root/autodl-tmp/i3dgs-runs/unused_for_skip \
  --exported_scene /root/autodl-tmp/exported-scenes/garden_depth_pipeline_v1 \
  --scaffold_output /root/autodl-tmp/scaffold-runs/garden_depth_pipeline_v1_skip \
  --i3dgs_python /root/autodl-tmp/conda-envs/i3dgs/bin/python \
  --scaffold_python /root/autodl-tmp/conda-envs/scaffold-cu128/bin/python \
  --iterations 30000 \
  --voxel_size 0.001 \
  --appearance_dim 0 \
  --eval \
  --export_depth \
  --skip_i3dgs
```

### 7.5 dry-run

`--dry_run` 只打印完整命令，不调用 subprocess，不读取 Git SHA，不创建日志、输出目录或
manifest。普通模式打印 I3DGS 和 Scaffold-GS 两条命令；skip 模式只打印 Scaffold-GS
命令。它仍执行只读的路径门禁；因此 skip+export_depth 仍会验证已有深度文件。

## 8. 最小可运行示例和安全验证

下面的命令不运行真实训练：

```powershell
# 工作目录：C:\Users\New\Documents\CGLF-I3DGS-stage1-refactor
# 查看复合入口是否暴露 --export_depth；只读取 argparse 帮助文本。
python pipeline/run_stage1.py -h | Select-String -Pattern 'export_depth'

# 运行全部当前单元测试；测试只使用临时目录、合成 PLY、合成 BA 和 mock 子进程。
python -m unittest discover -s tests -v

# 编译深度导出、适配器、I3DGS 入口、pipeline 和测试文件；不会启动模型。
python -m py_compile i3dgs/depth_export.py i3dgs/cglf_export.py i3dgs/scene/mono_depth.py i3dgs/train.py i3dgs/args.py pipeline/run_stage1.py tests/test_depth_export.py tests/test_stage1_pipeline.py

# 检查现有补丁的尾随空格等格式问题。
git diff --check
```

本次实际验证结果：

- `python -m unittest discover -s tests -v`：17 个测试全部通过；
- `python -m py_compile ...`：通过；
- `git diff --check`：通过；
- 未运行 I3DGS、Scaffold-GS 或真实 GPU 深度导出。

Windows 本地直接执行 `python i3dgs/train.py -h` 曾在导入阶段因环境缺少 `cupy` 而无法
到达 argparse；这属于环境依赖问题，不等价于 AutoDL 的 I3DGS 环境已经验证通过。应在
实际 I3DGS 环境中执行上面的 AutoDL 命令。

## 9. 测试覆盖范围

`tests/test_depth_export.py` 使用内存中的 BA 数组、关键帧和 mock estimator，覆盖：

- 活动数组范围、关键帧索引、越界/负 Z/非有限值过滤；
- `X_world → X_camera → Z` 和已知尺度/偏移的逆深度拟合；
- 同尺寸/不同尺寸重采样、半像素映射、Z 单位不随分辨率缩放；
- `.npy` 文件命名、二维形状、`float32`、正值/零无效值及统计文件；
- 对齐失败时 staging 清理且最终输出目录不发布半成品。

`tests/test_stage1_pipeline.py` 覆盖：

- binary little-endian PLY 和有限 XYZ；
- 输出目录嵌套/非空门禁；
- `skip_i3dgs` 的 exported scene 门禁和已有深度门禁；
- 普通 dry-run 的零副作用；
- `--eval` 只进入 Scaffold-GS；`--export_depth` 只进入 I3DGS；
- mock 的两个训练 subprocess、日志、Git SHA 和 manifest。

测试中的 mock 通过替换 `subprocess.run` 或深度估计器，阻止真实训练和模型下载；因此
测试通过只证明接口、数据结构和失败边界，不证明真实场景的几何质量。

## 10. 功能状态分类

### 已实现并在当前环境验证

- `--export_depth` 的 I3DGS 参数、pipeline 透传和 manifest 记录；
- BA 活动范围读取、索引/观测过滤和相机 Z 计算；
- 全局逆深度尺度/偏移拟合及现有分块对齐调用路径；
- 导出 `.npy`、`depth_stats.json` 和深度场景验证；
- staging 失败不发布半成品；
- 17 个合成/模拟单元测试、语法编译和 `git diff --check`。

### 已实现但尚未完整验证

- 真实 DA V2 输出形状、CUDA device、batch 和权重缓存上的端到端调用；
- 真实最终 BA 的 correspondence 数量、重投影阈值和分块对齐覆盖率；
- Garden 等真实场景上从 I3DGS 到 exported scene 再到 Scaffold-GS 的一次完整
  `--export_depth` 运行；
- 真实 Stage-2 读取这些深度图并进行带深度投票的训练/评估。

### 仍处于后续设计/验证阶段

- Stage-2 当前无效深度候选的回退逻辑可能选到无效像素；本任务没有修改它，开启深度
  投票前必须单独修复并测试；
- 深度投票的置信度、遮挡处理、阈值和最终超分辨率指标尚未由本改动定义；
- 尚未用真实 GT 深度或多场景实验确认尺度对齐的质量阈值。

## 11. 已知限制、兼容性和后续工作

1. **依赖环境**：I3DGS 运行需要与当前 CUDA/PyTorch 匹配的 `torch`、`cupy`、CUDA 扩展、
   Pillow 以及 DA V2 权重；Windows 本地缺少 `cupy` 时不能用 `train.py -h` 代替环境验证。
2. **场景单位**：输出 Z 沿用 BA 的任意尺度，不能直接用于米制误差报告。
3. **像素模型**：首版仅验证纯缩放、居中针孔和当前半像素约定；裁剪、旋转、畸变模型
   需要明确报错或新增适配，不能默认为同一个映射。
4. **有效覆盖**：每个注册图像若不足 `8` 个有效 BA 对应，当前导出会失败，适配器不会
   发布不完整场景。
5. **Stage-2 安全性**：深度文件导出成功不代表 Stage-2 深度投票已经安全；应先处理
   无效深度候选回退，再做独立的训练、渲染和锁定测试集评估。
6. **工程验证**：下一步应在一个小型真实场景上运行一次完整 pipeline，检查
   `depth_stats.json` 的有效像素比例、分位数、对齐残差和命名映射，再决定是否扩大实验。

## 12. 当前无法仅凭代码确认的问题

- 真实输入中每个最终注册关键帧是否都有至少 8 个满足重投影阈值的 BA 对应；
- `depth_grid_size` 的真实 CUDA 分块路径在目标 AutoDL 环境中的数值稳定性和耗时；
- 实际 DA V2 权重输出与当前 `_predict_depth()` 形状规范化之间是否完全匹配；
- 所有真实图像名在“第一个句点截断”规则下是否无冲突，并且与 Stage-2 的读取命名完全一致；
- 当前场景的有效深度覆盖率、对齐残差和深度投票后 PSNR/SSIM/LPIPS 是否达到研究目标；
- 运行带深度的 Stage-2 是否需要额外的显存、batch 或候选点限制。

这些问题需要真实 GPU/场景实验或明确的 Stage-2 设计决策，不能由本次合成测试替代。
