"""Stage-1 深度导出模块的合成测试。

测试只构造内存中的 BA 数组、关键帧和 mock DA V2 估计器，不加载模型、不使用
GPU、不启动训练。重点验证最终 BA 索引对应、相机 Z 定义、逆深度尺度/偏移、
像素重采样和 ``float32 [H,W]`` 文件契约。
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
import json

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "i3dgs"))

from depth_export import (  # noqa: E402
    DepthExportError,
    camera_z_from_world,
    collect_ba_correspondences,
    export_depth_maps,
    fit_inverse_depth_affine,
    resample_inverse_depth_to_target,
)


class FakeKeyframe:
    """提供深度导出器需要的最小最终关键帧接口。"""

    def __init__(self, index: int, name: str, width: int = 4, height: int = 4):
        self.index = index
        self.info = {"name": name}
        self.width = width
        self.height = height
        self.f = np.array(1.0, dtype=np.float32)
        self.centre = np.array([(width - 1) / 2, (height - 1) / 2], dtype=np.float32)
        self.image = np.zeros((3, height, width), dtype=np.float32)
        self._R = np.eye(3, dtype=np.float64)
        self._t = np.zeros(3, dtype=np.float64)

    def get_R(self):
        return self._R

    def get_t(self):
        return self._t


class FakeBA:
    """模拟 BA 的有效切片和预分配容量。"""

    def __init__(self):
        self.size = 4
        self.landmarks = np.array(
            [[0.0, 0.0, 2.0], [1.0, 0.0, 3.0], [0.0, 1.0, 4.0], [0.0, 0.0, -1.0], [99.0, 99.0, 99.0]],
            dtype=np.float64,
        )
        self.n_obs = np.array([2, 2, 2, 2, 0], dtype=np.int32)
        self.obs_size = 5
        self.obs_lm_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        self.obs_kf_ids = np.array([7, 7, 7, 7, 7], dtype=np.int32)
        self.obs_pt2d_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        self.obs_uvs = np.array(
            [[1.5, 1.5], [2.0, 1.5], [1.5, 2.0], [1.5, 1.5], [1.0, 1.0]],
            dtype=np.float64,
        )


class FakeScene:
    def __init__(self):
        self.ba_problem = FakeBA()
        self.keyframes = [FakeKeyframe(7, "scene.001.jpg")]
        self.depth_estimator = lambda image: np.array(
            [[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]]],
            dtype=np.float32,
        )


class DepthExportTests(unittest.TestCase):
    def test_ba_correspondences_use_live_ranges_and_keyframe_id(self) -> None:
        """预分配尾部、后方点和越界 landmark 不得进入最终对应关系。"""

        scene = FakeScene()
        uv, xyz, stats = collect_ba_correspondences(scene, scene.keyframes[0], min_observations=2)
        self.assertEqual(stats["candidates"], 4)
        self.assertEqual(stats["kept"], 3)
        self.assertEqual(uv.shape, (3, 2))
        np.testing.assert_array_equal(xyz[:, 2], np.array([2.0, 3.0, 4.0]))

    def test_camera_z_and_affine_alignment(self) -> None:
        """验证 Z 是相机光轴深度，并恢复已知逆深度尺度/偏移。"""

        xyz = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]])
        np.testing.assert_allclose(camera_z_from_world(xyz, np.eye(3), np.zeros(3)), [2.0, 4.0])
        mono = np.arange(1.0, 9.0)
        target = 2.5 * mono + 0.4
        scale, offset, used, stats = fit_inverse_depth_affine(mono, target, min_correspondences=4)
        self.assertAlmostEqual(scale, 2.5, places=6)
        self.assertAlmostEqual(offset, 0.4, places=6)
        self.assertEqual(int(used.sum()), 8)
        self.assertEqual(stats["mode"], "global_median_mean_abs_deviation")

    def test_resampling_preserves_z_units_and_shape(self) -> None:
        """改变分辨率只改变像素网格，不把 Z 数值乘以缩放比例。"""

        inverse = np.full((2, 2), 0.5, dtype=np.float32)
        sampled, valid = resample_inverse_depth_to_target(
            inverse, processed_size=(2, 2), target_size=(4, 4)
        )
        self.assertEqual(sampled.shape, (4, 4))
        self.assertTrue(valid.all())
        np.testing.assert_allclose(1.0 / sampled[valid], 2.0)

    def test_export_writes_float32_z_and_stats(self) -> None:
        """验证真实导出形状、命名、Z 数值约定和统计文件。"""

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "scene.001.jpg"
            Image.new("RGB", (4, 4), color=(20, 30, 40)).save(source)
            output = root / "depth"
            scene = FakeScene()
            stats = export_depth_maps(
                scene,
                ["scene.001.jpg"],
                {"scene.001.jpg": source},
                output,
                min_correspondences=3,
                reprojection_threshold_px=3.0,
            )
            path = output / "scene.npy"
            self.assertTrue(path.is_file())
            depth = np.load(path, allow_pickle=False)
            self.assertEqual(depth.dtype, np.float32)
            self.assertEqual(depth.shape, (4, 4))
            self.assertTrue(np.isfinite(depth).all())
            self.assertTrue((depth >= 0).all())
            self.assertTrue((depth > 0).any())
            self.assertEqual(stats["files"][0]["depth_definition"], "camera_coordinate_z")
            self.assertTrue((output.parent / "depth_stats.json").is_file())

    def test_degenerate_or_insufficient_alignment_fails(self) -> None:
        """常量 DA 预测和对应点不足时必须显式失败，不能静默填常数。"""

        with self.assertRaises(DepthExportError):
            fit_inverse_depth_affine(np.ones(8), np.arange(8.0), min_correspondences=4)
        with self.assertRaises(DepthExportError):
            fit_inverse_depth_affine(np.ones(2), np.ones(2), min_correspondences=4)

    def test_adapter_does_not_publish_when_depth_alignment_fails(self) -> None:
        """深度失败时适配器的 staging 目录必须被清理且最终目录不能出现。"""

        from PIL import Image
        from cglf_export import export_cglf_scene

        class OnePointScene:
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input" / "frame.png"
            source.parent.mkdir(parents=True)
            Image.new("RGB", (4, 4), color=(10, 20, 30)).save(source)
            reconstruction = root / "reconstruction"
            sparse = reconstruction / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "cameras.bin").write_bytes(b"camera")
            (sparse / "images.bin").write_bytes(b"image")
            (reconstruction / "metadata.json").write_text(
                json.dumps({"keyframes": [{"info": {"name": "frame.png"}}]}),
                encoding="utf-8",
            )
            keyframe = FakeKeyframe(0, "frame.png")
            scene = OnePointScene()
            scene.keyframes = [keyframe]
            scene.ba_problem = SimpleNamespace(
                size=1,
                landmarks=np.array([[0.0, 0.0, 2.0]]),
                n_obs=np.array([2]),
                obs_size=1,
                obs_lm_ids=np.array([0]),
                obs_kf_ids=np.array([0]),
                obs_pt2d_ids=np.array([0]),
                obs_uvs=np.array([[1.5, 1.5]]),
            )
            scene.depth_estimator = lambda image: np.ones((1, 4, 4), dtype=np.float32)
            output = root / "exported"
            fake_image = SimpleNamespace(name="frame.png")
            with mock.patch("cglf_export.read_images_binary", return_value={1: fake_image}):
                with self.assertRaises(DepthExportError):
                    export_cglf_scene(
                        scene,
                        reconstruction,
                        [source],
                        output,
                        export_depth=True,
                    )
            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".exported.tmp-*")))


if __name__ == "__main__":
    unittest.main()
