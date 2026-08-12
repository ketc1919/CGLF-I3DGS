# Snapshot 2026-08-13

本文件记录 `snapshot-2026-08-13` 冻结版本的来源与验证边界。

## 来源版本

- 顶层仓库基线：`ketc1919/CGLF-I3DGS@f3c422f2984a`
- I3DGS 上游基线：`graphdeco-inria/i3dgs@cf4d5b9762359a1d6de76fb9abf7b3dc764c1a42`
- I3DGS 适配器提交：`f736118eedd81e2bbb70526187d76328ccc0504e`
- Depth-Anything-V2 子模块：`a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`
- XFeat 子模块：`e92685f57f8318b18725c5c8c0bd28c7fe188d9a`
- GLM 子模块：`6f14f4792a0cde5d0cf2c910506724d61cb95834`
- CGLF：本地源码没有 Git 元数据，因此按 2026-08-13 工作区文件内容冻结，无法给出可靠的上游提交 SHA。

## 本次适配内容

- 新增 `--cglf_scene_path` 参数。
- 在 I3DGS 最终重建保存后导出 BA landmarks、相机二进制文件和成功注册的原始图像。
- 仅保留有限 XYZ 且 `n_obs >= 2` 的 landmarks。
- 输出 CGLF 可读取的 `points3D.ply` 九字段格式。
- 增加提前路径检查、源图像目录与符号链接目标目录保护。
- 使用同文件系统临时目录写入并在完成后发布；异常或中断时清理 staging，拒绝覆盖已有输出。
- 未修改 CGLF 训练逻辑。

## 已执行验证

```text
python -m unittest discover -s tests -p "test_cglf_export.py" -v
python -m py_compile args.py train.py cglf_export.py tests/test_cglf_export.py
git diff --check
```

结果：核心 smoke test 通过，语法检查通过，whitespace 检查通过。Windows 符号链接测试因系统权限不足跳过。

## 尚未验证

- 尚未在真实数据上运行完整 I3DGS GPU 重建并实际导出。
- 尚未在已编译 `simple_knn` 的 CGLF 环境中完成 Stage-1 短迭代训练。
- 尚未完成关闭投票后的 Stage-2 超分辨率端到端实验。

因此，这个标签表示“代码与适配器实现已冻结”，不表示论文实验流程已经完整复现。
