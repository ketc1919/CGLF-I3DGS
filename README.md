# CGLF-I3DGS

本仓库是一个研究快照，用于把 I3DGS 的无序图像位姿估计与 BA landmarks 导出到 CGLF 可读取的 COLMAP 场景目录，为后续 CGLF Stage-1 和 Stage-2 超分辨率实验提供输入。

当前冻结版本完成的是 **I3DGS → CGLF 数据导出适配器**。它不修改 CGLF 的训练逻辑，也不改写原始图像。

## 目录结构

```text
CGLF-I3DGS/
├── i3dgs/                 # I3DGS 与导出适配器
├── CGLF/                  # CGLF Stage-1 / Stage-2 代码
└── SNAPSHOT.md            # 本次冻结版本的提交与验证记录
```

两个子项目的安装与原始用法见：

- [I3DGS README](i3dgs/README.md)
- [CGLF 中文 README](CGLF/README.zh-CN.md)
- [CGLF English README](CGLF/README.md)

## 克隆

I3DGS 的部分依赖保留为 Git 子模块，因此需要递归克隆：

```bash
git clone --recursive https://github.com/ketc1919/CGLF-I3DGS.git
cd CGLF-I3DGS
```

如果已经普通克隆：

```bash
git submodule update --init --recursive
```

## 导出 CGLF 场景

先按照 `i3dgs/README.md` 配置 I3DGS 环境，然后运行：

```bash
cd i3dgs
python train.py \
  -s /path/to/input_scene \
  -m /path/to/i3dgs_output \
  --cglf_scene_path /path/to/new_cglf_scene
```

`new_cglf_scene` 必须尚不存在，且不能位于原始图像目录内部。导出只在最终位姿估计与重建保存完成后触发，输出：

```text
new_cglf_scene/
├── export_stats.json
├── images/                       # 仅成功注册的原始图像
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.ply
```

Landmarks 会过滤为有限 XYZ 且 `n_obs >= 2`；PLY 字段为 `x,y,z,nx,ny,nz,red,green,blue`。导出过程先写入目标同级临时目录，再原子发布，并拒绝覆盖已有目标。

## 当前状态

- 已通过合成数据 smoke test、Python 语法检查和 Git whitespace 检查。
- Windows 上需要管理员或开发者模式才能执行符号链接专项测试；本次该测试因权限不足跳过。
- 尚未完成真实 I3DGS GPU 重建到 CGLF Stage-1/Stage-2 的完整端到端验证。

完整冻结信息见 [SNAPSHOT.md](SNAPSHOT.md)。

## 许可证与来源

I3DGS 与 CGLF 各自保留原有许可证及第三方声明。使用或再发布前，请分别阅读子目录中的许可证文件。本仓库中的组合与适配代码不改变上游项目的许可条件。
