import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image as PILImage

try:
    from plyfile import PlyData
except ImportError:
    PlyData = None

from cglf_export import (
    PLY_VERTEX_PROPERTIES,
    _source_image_map,
    export_cglf_scene,
    preflight_cglf_export,
)
from dataloaders.read_write_model import (
    BaseImage,
    Camera,
    write_cameras_binary,
    write_images_binary,
)


PLY_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def read_test_ply(path):
    header_lines = []
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise AssertionError("PLY header has no end_header")
            decoded = line.decode("ascii").rstrip("\r\n")
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        payload = handle.read()

    vertex_line = next(line for line in header_lines if line.startswith("element vertex "))
    vertex_count = int(vertex_line.rsplit(" ", 1)[1])
    properties = tuple(
        line.rsplit(" ", 1)[1]
        for line in header_lines
        if line.startswith("property ")
    )
    vertices = np.frombuffer(payload, dtype=PLY_VERTEX_DTYPE)
    if len(vertices) != vertex_count:
        raise AssertionError(
            f"PLY payload has {len(vertices)} vertices, header says {vertex_count}"
        )
    return properties, vertices


class CGLFExportSmokeTest(unittest.TestCase):
    def test_filters_landmarks_and_builds_non_overwriting_scene(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_images = root / "source" / "images"
            reconstruction = root / "reconstruction"
            sparse = reconstruction / "sparse" / "0"
            output = root / "cglf_scene"
            source_images.mkdir(parents=True)
            sparse.mkdir(parents=True)

            image_specs = {
                "registered_a.png": (255, 0, 0),
                "registered_b.jpg": (0, 0, 255),
                "not_registered.png": (0, 255, 0),
            }
            input_paths = []
            for name, color in image_specs.items():
                path = source_images / name
                PILImage.new("RGB", (4, 3), color).save(path)
                input_paths.append(path)

            cameras = {
                image_id: Camera(
                    id=image_id,
                    model="SIMPLE_PINHOLE",
                    width=4,
                    height=3,
                    params=np.array([4.0, 1.5, 1.0]),
                )
                for image_id in range(2)
            }
            registered_names = ["registered_a.png", "registered_b.jpg"]
            images = {
                image_id: BaseImage(
                    id=image_id,
                    qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                    tvec=np.zeros(3),
                    camera_id=image_id,
                    name=name,
                    xys=np.empty((0, 2)),
                    point3D_ids=np.empty((0,), dtype=np.int64),
                )
                for image_id, name in enumerate(registered_names)
            }
            write_cameras_binary(cameras, str(sparse / "cameras.bin"))
            write_images_binary(images, str(sparse / "images.bin"))
            (reconstruction / "metadata.json").write_text(
                json.dumps(
                    {
                        "keyframes": [
                            {"info": {"name": name}} for name in registered_names
                        ]
                    }
                ),
                encoding="utf-8",
            )

            red = np.zeros((3, 3, 4), dtype=np.float32)
            red[0] = 1.0
            blue = np.zeros((3, 3, 4), dtype=np.float32)
            blue[2] = 1.0
            scene_model = SimpleNamespace(
                keyframes=[SimpleNamespace(image=red), SimpleNamespace(image=blue)],
                ba_problem=SimpleNamespace(
                    size=4,
                    landmarks=np.array(
                        [
                            [1.0, 2.0, 3.0],
                            [np.nan, 0.0, 0.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0],
                        ],
                        dtype=np.float32,
                    ),
                    n_obs=np.array([2, 3, 1, 2], dtype=np.int32),
                    obs_size=4,
                    obs_lm_ids=np.array([0, 0, 3, 3], dtype=np.int32),
                    obs_kf_ids=np.array([0, 1, 1, 1], dtype=np.int32),
                    obs_uvs=np.array(
                        [[0.0, 0.0], [1.0, 1.0], [2.0, 1.0], [3.0, 2.0]],
                        dtype=np.float32,
                    ),
                ),
            )

            real_copy2 = shutil.copy2

            def copy_while_final_path_is_hidden(source, destination, *args, **kwargs):
                self.assertFalse(output.exists())
                staging_paths = list(root.glob(f".{output.name}.tmp-*"))
                self.assertEqual(len(staging_paths), 1)
                self.assertTrue(Path(destination).is_relative_to(staging_paths[0]))
                return real_copy2(source, destination, *args, **kwargs)

            with patch(
                "cglf_export.shutil.copy2",
                side_effect=copy_while_final_path_is_hidden,
            ):
                stats = export_cglf_scene(
                    scene_model=scene_model,
                    reconstruction_path=reconstruction,
                    input_image_paths=input_paths,
                    output_path=output,
                )

            self.assertEqual(stats["input_images"], 3)
            self.assertEqual(stats["registered_images"], 2)
            self.assertAlmostEqual(stats["registration_rate"], 2 / 3)
            self.assertEqual(stats["landmarks_before_filter"], 4)
            self.assertEqual(stats["landmarks_after_filter"], 2)

            self.assertEqual(
                sorted(path.name for path in (output / "images").iterdir()),
                sorted(registered_names),
            )
            self.assertFalse((output / "images" / "not_registered.png").exists())
            self.assertEqual(
                (output / "sparse" / "0" / "cameras.bin").read_bytes(),
                (sparse / "cameras.bin").read_bytes(),
            )
            self.assertEqual(
                (output / "sparse" / "0" / "images.bin").read_bytes(),
                (sparse / "images.bin").read_bytes(),
            )

            properties, vertices = read_test_ply(
                output / "sparse" / "0" / "points3D.ply"
            )
            self.assertEqual(properties, PLY_VERTEX_PROPERTIES)
            self.assertEqual(len(vertices), 2)
            np.testing.assert_allclose(
                np.column_stack((vertices["x"], vertices["y"], vertices["z"])),
                [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]],
            )
            np.testing.assert_array_equal(
                np.column_stack((vertices["nx"], vertices["ny"], vertices["nz"])),
                np.zeros((2, 3), dtype=np.float32),
            )
            np.testing.assert_array_equal(
                np.column_stack((vertices["red"], vertices["green"], vertices["blue"])),
                [[128, 0, 128], [0, 0, 255]],
            )
            if PlyData is not None:
                cglf_style_vertices = PlyData.read(
                    output / "sparse" / "0" / "points3D.ply", mmap=False
                )["vertex"]
                self.assertEqual(
                    tuple(cglf_style_vertices.data.dtype.names), PLY_VERTEX_PROPERTIES
                )
                self.assertEqual(len(cglf_style_vertices), 2)

            on_disk_stats = json.loads((output / "export_stats.json").read_text())
            self.assertEqual(on_disk_stats, stats)
            self.assertEqual(list(root.glob(f".{output.name}.tmp-*")), [])

            preflight_output = root / "preflight_only"
            preflight_cglf_export(preflight_output, input_paths)
            self.assertFalse(preflight_output.exists())

            existing_file = root / "existing_file"
            existing_file.write_bytes(b"must stay unchanged")
            with self.assertRaises(FileExistsError):
                preflight_cglf_export(existing_file, input_paths)
            self.assertEqual(existing_file.read_bytes(), b"must stay unchanged")

            with self.assertRaises(FileExistsError):
                preflight_cglf_export(output, input_paths)
            with self.assertRaises(FileExistsError):
                export_cglf_scene(
                    scene_model=scene_model,
                    reconstruction_path=reconstruction,
                    input_image_paths=input_paths,
                    output_path=output,
                )

            forbidden_output = source_images / "must_not_be_created"
            with self.assertRaises(ValueError):
                export_cglf_scene(
                    scene_model=scene_model,
                    reconstruction_path=reconstruction,
                    input_image_paths=input_paths,
                    output_path=forbidden_output,
                )
            self.assertFalse(forbidden_output.exists())

            interrupted_output = root / "interrupted_scene"
            with patch(
                "cglf_export.shutil.copy2",
                side_effect=KeyboardInterrupt("simulated Ctrl+C"),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    export_cglf_scene(
                        scene_model=scene_model,
                        reconstruction_path=reconstruction,
                        input_image_paths=input_paths,
                        output_path=interrupted_output,
                    )
            self.assertFalse(interrupted_output.exists())
            self.assertEqual(
                list(root.glob(f".{interrupted_output.name}.tmp-*")), []
            )

            failed_output = root / "failed_scene"
            source_bytes = {
                path: path.read_bytes()
                for path in input_paths + [sparse / "cameras.bin", sparse / "images.bin"]
            }
            with patch(
                "cglf_export.shutil.copy2",
                side_effect=OSError("simulated copy failure"),
            ):
                with self.assertRaises(OSError):
                    export_cglf_scene(
                        scene_model=scene_model,
                        reconstruction_path=reconstruction,
                        input_image_paths=input_paths,
                        output_path=failed_output,
                    )
            self.assertFalse(failed_output.exists())
            self.assertEqual(list(root.glob(f".{failed_output.name}.tmp-*")), [])
            for path, original_bytes in source_bytes.items():
                self.assertEqual(path.read_bytes(), original_bytes)

            raced_output = root / "raced_scene"
            copy_count = 0

            def create_competing_output(source, destination, *args, **kwargs):
                nonlocal copy_count
                if copy_count == 0:
                    raced_output.mkdir()
                    (raced_output / "owner.txt").write_text("other process")
                copy_count += 1
                return real_copy2(source, destination, *args, **kwargs)

            with patch(
                "cglf_export.shutil.copy2",
                side_effect=create_competing_output,
            ):
                with self.assertRaises(FileExistsError):
                    export_cglf_scene(
                        scene_model=scene_model,
                        reconstruction_path=reconstruction,
                        input_image_paths=input_paths,
                        output_path=raced_output,
                    )
            self.assertEqual(
                (raced_output / "owner.txt").read_text(), "other process"
            )
            self.assertEqual(list(root.glob(f".{raced_output.name}.tmp-*")), [])

    def test_symbolic_link_keeps_its_input_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            real_images = root / "real" / "images"
            aliases = root / "aliases"
            real_images.mkdir(parents=True)
            aliases.mkdir()
            target = real_images / "target.png"
            alias = aliases / "registered_alias.png"
            PILImage.new("RGB", (2, 2), (1, 2, 3)).save(target)
            try:
                alias.symlink_to(target)
            except OSError as error:
                self.skipTest(f"Symbolic links are unavailable: {error}")

            image_map = _source_image_map([alias])
            self.assertEqual(set(image_map), {"registered_alias.png"})
            self.assertEqual(image_map["registered_alias.png"], target.resolve())

            allowed_output = root / "allowed_scene"
            preflight_cglf_export(allowed_output, [alias])
            self.assertFalse(allowed_output.exists())

            for forbidden_output in (
                aliases / "scene_in_alias_directory",
                real_images / "scene_in_target_directory",
            ):
                with self.assertRaises(ValueError):
                    preflight_cglf_export(forbidden_output, [alias])
                self.assertFalse(forbidden_output.exists())

            broken_link = root / "broken_output_link"
            try:
                broken_link.symlink_to(root / "missing_target", target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Broken symbolic links are unavailable: {error}")
            self.assertFalse(broken_link.exists())
            self.assertTrue(broken_link.is_symlink())
            with self.assertRaises(FileExistsError):
                preflight_cglf_export(broken_link, [alias])


if __name__ == "__main__":
    unittest.main()
