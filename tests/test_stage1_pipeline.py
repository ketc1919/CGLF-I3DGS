"""复合 Stage-1 编排器的学习型合成测试。

本文件不使用真实数据集，也不启动 I3DGS 或 Scaffold-GS 训练。测试在每个
用例的临时目录中构造最小输入场景、binary little-endian points3D.ply 和
Scaffold-GS checkpoint，再直接调用编排器的验证函数或 mock 掉所有
subprocess.run。这样可以阅读和检查路径门禁、场景接口、checkpoint 文件
布局、dry-run 分支以及 manifest 写入流程，而不会改变原始数据或训练结果。
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from pipeline import run_stage1


PLY_FIELDS = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "red",
    "green",
    "blue",
)

# 这个字段集合与编排器对 exported_scene/points3D.ply 的接口约定一致。
# 测试故意使用完整字段，避免把“文件存在”误认为“PLY 结构可读”。

def write_binary_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    """在临时路径写入最小 binary little-endian PLY。

    ``points`` 只提供 XYZ；测试为每个顶点补充零法线和灰色 RGB，并写入与
    I3DGS 适配器相同的九个属性。该辅助函数只创建临时测试文件。
    """

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
            "",
        ]
    ).encode("ascii")
    body = b"".join(
        struct.pack("<ffffffBBB", *point, 0.0, 0.0, 0.0, 128, 128, 128)
        for point in points
    )
    path.write_bytes(header + body)


def make_exported_scene(path: Path, *, points: list[tuple[float, float, float]] | None = None) -> None:
    """构造一个包含 COLMAP 文件和图像占位文件的临时 exported_scene。

    目录模拟适配器交给 Scaffold-GS 的接口：``images/``、``cameras.bin``、
    ``images.bin`` 和 ``points3D.ply``。二进制 COLMAP 文件在这里仅使用非空
    占位字节，因为本测试关注编排器的文件门禁而不是 COLMAP 解码器。
    """

    (path / "images").mkdir(parents=True)
    sparse = path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (path / "images" / "registered.png").write_bytes(b"image")
    (sparse / "cameras.bin").write_bytes(b"camera")
    (sparse / "images.bin").write_bytes(b"image-record")
    write_binary_ply(sparse / "points3D.ply", points or [(0.0, 1.0, 2.0)])


def make_checkpoint(path: Path, iterations: int, appearance_dim: int = 0) -> None:
    """构造指定迭代数的临时 Scaffold-GS checkpoint 文件集合。

    ``appearance_dim`` 大于零时额外生成 appearance embedding，用来覆盖
    编排器的条件文件检查；所有文件只写入临时目录。
    """

    iteration_dir = path / "point_cloud" / f"iteration_{iterations}"
    iteration_dir.mkdir(parents=True)
    (path / "cfg_args").write_text("Namespace()\n", encoding="utf-8")
    for name in ("point_cloud.ply", "opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt"):
        (iteration_dir / name).write_bytes(b"checkpoint")
    if appearance_dim > 0:
        (iteration_dir / "embedding_appearance.pt").write_bytes(b"checkpoint")


class Stage1PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        """为每个测试准备独立临时输入场景和一张占位图像。"""

        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_scene = self.root / "input"
        (self.input_scene / "images").mkdir(parents=True)
        (self.input_scene / "images" / "frame.png").write_bytes(b"input")

    def tearDown(self) -> None:
        """清理当前测试创建的临时目录。"""

        self.temp_dir.cleanup()

    def namespace(self, **overrides: object):
        """构造传给 run_pipeline 的最小参数对象，并允许单项覆盖。"""

        values = {
            "input_scene": str(self.input_scene),
            "i3dgs_output": str(self.root / "i3dgs-output"),
            "exported_scene": str(self.root / "exported"),
            "scaffold_output": str(self.root / "scaffold-output"),
            "i3dgs_python": sys.executable,
            "scaffold_python": sys.executable,
            "iterations": 10,
            "voxel_size": 0.001,
            "appearance_dim": 0,
            "eval": False,
            "i3dgs_num_iterations": None,
            "skip_i3dgs": False,
            "dry_run": False,
        }
        values.update(overrides)
        return run_stage1.argparse.Namespace(**values)

    def test_binary_ply_and_exported_scene_validation(self) -> None:
        """验证 binary little-endian PLY 的顶点数量和九个字段能够被读取。"""

        exported = self.root / "exported"
        make_exported_scene(exported, points=[(1.0, 2.0, 3.0), (-1.0, 0.5, 4.0)])
        info = run_stage1.validate_exported_scene(exported)
        # 这里确认场景验证不仅看文件是否存在，还解析了 vertex 数量和字段名。
        self.assertEqual(info["vertex_count"], 2)
        self.assertEqual(set(info["properties"]), set(PLY_FIELDS))

    def test_exported_scene_rejects_nonfinite_xyz(self) -> None:
        """验证 PLY 含 NaN 坐标时，exported_scene 校验预期抛出异常。"""

        exported = self.root / "bad-exported"
        make_exported_scene(exported, points=[(float("nan"), 0.0, 0.0)])
        # NaN 不代表有限三维位置；此处测试预期的失败路径，而非训练流程。
        with self.assertRaises(ValueError):
            run_stage1.validate_exported_scene(exported)

    def test_checkpoint_validation_checks_appearance_embedding(self) -> None:
        """验证 appearance_dim 大于零时必须存在 embedding checkpoint。"""

        checkpoint = self.root / "checkpoint"
        make_checkpoint(checkpoint, 10, appearance_dim=4)
        info = run_stage1.validate_scaffold_checkpoint(checkpoint, 10, 4)
        # 第一条断言确认验证定位到了请求的 iteration_10 目录。
        self.assertEqual(info["iteration"], 10)
        (checkpoint / "point_cloud" / "iteration_10" / "embedding_appearance.pt").unlink()
        # 删除条件文件后，测试预期验证抛出 FileNotFoundError。
        with self.assertRaises(FileNotFoundError):
            run_stage1.validate_scaffold_checkpoint(checkpoint, 10, 4)

    def test_skip_requires_valid_exported_scene(self) -> None:
        """验证 skip_i3dgs 不能绕过 exported_scene 的存在性门禁。"""

        args = self.namespace(skip_i3dgs=True, dry_run=True)
        # 这里没有创建 exported scene，因此预期在任何 subprocess 之前失败。
        with self.assertRaises(FileNotFoundError):
            run_stage1.run_pipeline(args)

    def test_nested_paths_and_nonempty_outputs_are_rejected(self) -> None:
        """验证输入内部路径、输出嵌套和非空输出目录都会被拒绝。"""

        args = self.namespace(i3dgs_output=str(self.input_scene / "i3dgs"))
        # 第一个场景把 I3DGS 输出放进原始输入目录内部。
        with self.assertRaises(ValueError):
            run_stage1.run_pipeline(args)

        exported = self.root / "exported"
        make_exported_scene(exported)
        args = self.namespace(skip_i3dgs=True, exported_scene=str(exported))
        args.scaffold_output = str(exported / "nested-output")
        # 第二个场景把 Scaffold-GS 输出放进 exported scene 内部。
        with self.assertRaises(ValueError):
            run_stage1.run_pipeline(args)

        occupied = self.root / "occupied"
        occupied.mkdir()
        (occupied / "existing.txt").write_text("keep", encoding="utf-8")
        args = self.namespace(i3dgs_output=str(occupied))
        # 第三个场景模拟已有非空模型目录，预期不覆盖它。
        with self.assertRaises(FileExistsError):
            run_stage1.run_pipeline(args)

    def test_normal_dry_run_has_zero_side_effects_and_prints_two_commands(self) -> None:
        """验证普通 dry-run 只打印两条命令，不调用 subprocess 或创建输出。"""

        args = self.namespace(dry_run=True)
        before = set(self.root.rglob("*"))
        # mock 任何 subprocess.run；如果 dry-run 越过边界调用它，测试会立即失败。
        with mock.patch.object(
            run_stage1.subprocess,
            "run",
            side_effect=AssertionError("dry-run must not invoke subprocess"),
        ), redirect_stdout(io.StringIO()) as output:
            self.assertIsNone(run_stage1.run_pipeline(args))
        after = set(self.root.rglob("*"))
        # 比较临时目录前后目录项，检查 dry-run 的零文件副作用。
        self.assertEqual(before, after)
        text = output.getvalue()
        # 普通模式应同时显示 I3DGS 和 Scaffold-GS 两个完整命令。
        self.assertIn("I3DGS command:", text)
        self.assertIn("Scaffold-GS command:", text)
        self.assertFalse(Path(args.exported_scene).exists())

    def test_eval_is_only_forwarded_to_scaffold_command(self) -> None:
        """验证顶层 --eval 只进入 Scaffold-GS 命令，不影响 I3DGS 命令。"""

        args = self.namespace(eval=True, dry_run=True)
        with mock.patch.object(
            run_stage1.subprocess,
            "run",
            side_effect=AssertionError("dry-run must not invoke subprocess"),
        ), redirect_stdout(io.StringIO()) as output:
            self.assertIsNone(run_stage1.run_pipeline(args))

        lines = output.getvalue().splitlines()
        i3dgs_line = next(line for line in lines if line.startswith("I3DGS command:"))
        scaffold_line = next(line for line in lines if line.startswith("Scaffold-GS command:"))
        self.assertNotIn("--eval", i3dgs_line)
        self.assertIn("--eval", scaffold_line)

    def test_skip_dry_run_only_prints_scaffold_command(self) -> None:
        """验证 skip dry-run 复用有效场景且只打印 Scaffold-GS 命令。"""

        exported = self.root / "exported"
        make_exported_scene(exported)
        args = self.namespace(exported_scene=str(exported), skip_i3dgs=True, dry_run=True)
        before = set(self.root.rglob("*"))
        # 与普通 dry-run 一样，mock 用来阻止任何真实训练子进程。
        with mock.patch.object(
            run_stage1.subprocess,
            "run",
            side_effect=AssertionError("dry-run must not invoke subprocess"),
        ), redirect_stdout(io.StringIO()) as output:
            self.assertIsNone(run_stage1.run_pipeline(args))
        # 已有 exported scene 应保持不变，只允许产生终端输出。
        self.assertEqual(before, set(self.root.rglob("*")))
        text = output.getvalue()
        # skip 模式不应出现 I3DGS 命令，但仍应出现 Scaffold-GS 命令。
        self.assertNotIn("I3DGS command:", text)
        self.assertIn("Scaffold-GS command:", text)

    def test_normal_pipeline_mocks_both_training_subprocesses_and_writes_manifest(self) -> None:
        """用 mock 模拟两次训练，检查完整流程、Git SHA、日志和 manifest。"""

        args = self.namespace(appearance_dim=0)

        def fake_run(command, **kwargs):
            """按命令类型创建合成输出，避免启动真实训练或 Git。"""

            # 训练命令必须明确关闭 shell，并把 stderr 合并到 stdout 日志；Git
            # 查询则使用 PIPE 读取 SHA，因此两类调用的 stderr 目标不同。
            self.assertFalse(kwargs.get("shell"))
            self.assertEqual(kwargs["stderr"], subprocess.STDOUT if command[0] != "git" else subprocess.PIPE)
            if command[0] == "git":
                return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")
            command_script = str(command[1]).replace("\\", "/")
            if command_script.endswith("i3dgs/train.py"):
                # 这里模拟 I3DGS 完成位姿/BA 后由 adapter 写出 exported scene。
                make_exported_scene(Path(args.exported_scene))
            elif command_script.endswith("Scaffold-GS/train.py"):
                # 这里模拟 Scaffold-GS 写出目标迭代的 checkpoint 文件。
                make_checkpoint(Path(args.scaffold_output), args.iterations, args.appearance_dim)
            else:
                self.fail(f"unexpected command: {command}")
            return subprocess.CompletedProcess(command, 0)

        with mock.patch.object(run_stage1.subprocess, "run", side_effect=fake_run) as mocked:
            result = run_stage1.run_pipeline(args)
        # 两次训练加两次只读 Git SHA 查询，全部来自 fake_run。
        self.assertEqual(mocked.call_count, 4)  # two training calls and two git SHA calls
        manifest_path = Path(result["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # manifest 应记录 I3DGS 已执行，且命令以参数数组保存。
        self.assertEqual(manifest["commands"]["i3dgs"]["status"], "executed")
        self.assertIsInstance(manifest["commands"]["i3dgs"]["argv"], list)
        self.assertFalse(manifest["eval"])
        # fake Git 返回的 SHA 应被写入 manifest；两个独立日志也应存在。
        self.assertEqual(manifest["i3dgs_git_sha"], "abc123")
        self.assertTrue(Path(result["i3dgs_log"]).is_file())
        self.assertTrue(Path(result["scaffold_log"]).is_file())


if __name__ == "__main__":
    unittest.main()
