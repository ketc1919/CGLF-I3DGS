"""Export an in-memory I3DGS reconstruction as a CGLF COLMAP scene.

The BA landmarks are not part of the regular I3DGS checkpoint, so this adapter
must be called while the reconstructed ``SceneModel`` is still alive.  Camera
files are copied byte-for-byte from the reconstruction output, while registered
source images are selected by cross-checking ``images.bin`` and
``metadata.json``.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping

import numpy as np

from dataloaders.read_write_model import read_images_binary


PLY_VERTEX_PROPERTIES = (
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


def _as_numpy(value, dtype=None) -> np.ndarray:
    """Convert a NumPy/torch-like value to a detached CPU array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _metadata_image_names(metadata_path: Path) -> list[str]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    keyframes = metadata.get("keyframes")
    if not isinstance(keyframes, list):
        raise ValueError(f"Missing keyframes list in {metadata_path}")

    names = []
    for index, keyframe in enumerate(keyframes):
        name = keyframe.get("info", {}).get("name") if isinstance(keyframe, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"Missing registered image name in metadata keyframe {index}"
            )
        names.append(name)
    return names


def _validated_registered_names(images_bin: Path, metadata_path: Path) -> list[str]:
    images = read_images_binary(str(images_bin))
    binary_names = [image.name for _, image in sorted(images.items())]
    metadata_names = _metadata_image_names(metadata_path)

    duplicate_binary = [name for name, count in Counter(binary_names).items() if count > 1]
    duplicate_metadata = [
        name for name, count in Counter(metadata_names).items() if count > 1
    ]
    if duplicate_binary or duplicate_metadata:
        raise ValueError(
            "Registered image names must be unique; "
            f"images.bin duplicates={duplicate_binary}, "
            f"metadata duplicates={duplicate_metadata}"
        )

    if Counter(binary_names) != Counter(metadata_names):
        missing_from_metadata = sorted(set(binary_names) - set(metadata_names))
        missing_from_binary = sorted(set(metadata_names) - set(binary_names))
        raise ValueError(
            "images.bin and metadata.json disagree on registered images; "
            f"only in images.bin={missing_from_metadata}, "
            f"only in metadata.json={missing_from_binary}"
        )

    basenames = [name.replace("\\", "/").rsplit("/", 1)[-1] for name in binary_names]
    duplicate_basenames = [
        name for name, count in Counter(basenames).items() if count > 1
    ]
    if duplicate_basenames:
        raise ValueError(
            "Registered image paths collide after CGLF basename handling: "
            f"{duplicate_basenames}"
        )

    casefolded_basenames: dict[str, str] = {}
    for name in basenames:
        folded = name.casefold()
        if folded in casefolded_basenames and casefolded_basenames[folded] != name:
            raise ValueError(
                "Registered image names collide on a case-insensitive filesystem: "
                f"{casefolded_basenames[folded]} and {name}"
            )
        casefolded_basenames[folded] = name

    cglf_names = [name.split(".")[0] for name in basenames]
    duplicate_cglf_names = [
        name for name, count in Counter(cglf_names).items() if count > 1
    ]
    if duplicate_cglf_names:
        raise ValueError(
            "Registered images collide under CGLF image-name handling: "
            f"{duplicate_cglf_names}"
        )

    casefolded_cglf_names: dict[str, str] = {}
    for name in cglf_names:
        folded = name.casefold()
        if folded in casefolded_cglf_names and casefolded_cglf_names[folded] != name:
            raise ValueError(
                "Registered images collide under case-insensitive CGLF "
                f"image-name handling: {casefolded_cglf_names[folded]} and {name}"
            )
        casefolded_cglf_names[folded] = name

    return basenames


def _source_image_map(input_image_paths: Iterable[os.PathLike | str]) -> Mapping[str, Path]:
    image_map: dict[str, Path] = {}
    for raw_path in input_image_paths:
        raw_path = Path(raw_path)
        name = raw_path.name
        path = raw_path.resolve()
        if name in image_map and image_map[name] != path:
            raise ValueError(f"Input images have a duplicate basename: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {path}")
        image_map[name] = path

    casefolded_names: dict[str, str] = {}
    for name in image_map:
        folded = name.casefold()
        if folded in casefolded_names and casefolded_names[folded] != name:
            raise ValueError(
                "Input image names collide on a case-insensitive filesystem: "
                f"{casefolded_names[folded]} and {name}"
            )
        casefolded_names[folded] = name
    return image_map


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _validate_output_location(
    output_path: Path, input_image_paths: Iterable[os.PathLike | str]
) -> None:
    """Keep adapter output out of alias and resolved source-image directories."""
    resolved_output = output_path.resolve()
    source_directories = set()
    for path in input_image_paths:
        image_path = Path(path)
        # Protect both the directory containing a possible symlink and the
        # directory containing its resolved target.  The exporter dereferences
        # file symlinks while copying, so both locations are source data.
        source_directories.add(image_path.parent.resolve())
        source_directories.add(image_path.resolve().parent)
    for source_directory in source_directories:
        if _is_within(resolved_output, source_directory):
            raise ValueError(
                "CGLF output must be outside the original image directory: "
                f"output={resolved_output}, images={source_directory}"
            )


def preflight_cglf_export(
    output_path: os.PathLike | str,
    input_image_paths: Iterable[os.PathLike | str],
) -> None:
    """Validate a CGLF export destination without creating or changing files."""
    raw_output_path = Path(output_path)
    input_image_paths = list(input_image_paths)

    # Path.exists() follows symlinks and returns False for a broken one.  Treat
    # any directory entry at the requested destination as user-owned instead.
    if os.path.lexists(raw_output_path):
        raise FileExistsError(
            f"Refusing to overwrite existing CGLF scene directory: {raw_output_path}"
        )

    _validate_output_location(raw_output_path, input_image_paths)
    input_image_map = _source_image_map(input_image_paths)
    if not input_image_map:
        raise ValueError("No input images were provided to the CGLF exporter")


def _sample_landmark_colors(scene_model, landmark_count: int) -> np.ndarray:
    """Average RGB samples from valid BA observations; use gray as fallback."""
    colors = np.full((landmark_count, 3), 127, dtype=np.uint8)
    ba_problem = scene_model.ba_problem
    obs_size = int(ba_problem.obs_size)
    if landmark_count == 0 or obs_size == 0:
        return colors

    obs_lm_ids = _as_numpy(ba_problem.obs_lm_ids[:obs_size], np.int64)
    obs_kf_ids = _as_numpy(ba_problem.obs_kf_ids[:obs_size], np.int64)
    obs_uvs = _as_numpy(ba_problem.obs_uvs[:obs_size], np.float64)

    valid = (
        (obs_lm_ids >= 0)
        & (obs_lm_ids < landmark_count)
        & (obs_kf_ids >= 0)
        & (obs_kf_ids < len(scene_model.keyframes))
        & np.isfinite(obs_uvs).all(axis=1)
    )
    if not np.any(valid):
        return colors

    color_sums = np.zeros((landmark_count, 3), dtype=np.float64)
    color_counts = np.zeros(landmark_count, dtype=np.int64)

    for keyframe_id in np.unique(obs_kf_ids[valid]):
        observation_ids = np.flatnonzero(valid & (obs_kf_ids == keyframe_id))
        image = _as_numpy(scene_model.keyframes[int(keyframe_id)].image)
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3 or image.shape[0] < 3:
            continue

        height, width = image.shape[1:3]
        uv = obs_uvs[observation_ids]
        x = np.rint(uv[:, 0]).astype(np.int64)
        y = np.rint(uv[:, 1]).astype(np.int64)
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not np.any(inside):
            continue

        observation_ids = observation_ids[inside]
        x = x[inside]
        y = y[inside]
        samples = np.asarray(image[:3, y, x].T, dtype=np.float64)
        finite_samples = np.isfinite(samples).all(axis=1)
        if not np.any(finite_samples):
            continue

        samples = samples[finite_samples]
        observation_ids = observation_ids[finite_samples]
        if samples.size and np.nanmax(samples) <= 1.0 + 1e-6:
            samples *= 255.0
        samples = np.clip(samples, 0.0, 255.0)
        landmark_ids = obs_lm_ids[observation_ids]
        np.add.at(color_sums, landmark_ids, samples)
        np.add.at(color_counts, landmark_ids, 1)

    sampled = color_counts > 0
    colors[sampled] = np.rint(
        color_sums[sampled] / color_counts[sampled, None]
    ).astype(np.uint8)
    return colors


def extract_filtered_ba_landmarks(scene_model, min_observations: int = 2):
    """Return filtered BA xyz/rgb arrays and before/after counts."""
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1")

    ba_problem = scene_model.ba_problem
    landmark_count = int(ba_problem.size)
    xyz = _as_numpy(ba_problem.landmarks[:landmark_count], np.float32)
    n_obs = _as_numpy(ba_problem.n_obs[:landmark_count], np.int64)
    if xyz.shape != (landmark_count, 3):
        raise ValueError(f"Unexpected BA landmark shape: {xyz.shape}")
    if n_obs.shape != (landmark_count,):
        raise ValueError(f"Unexpected BA observation-count shape: {n_obs.shape}")

    keep = np.isfinite(xyz).all(axis=1) & (n_obs >= min_observations)
    rgb = _sample_landmark_colors(scene_model, landmark_count)
    return xyz[keep], rgb[keep], landmark_count, int(np.count_nonzero(keep))


def write_cglf_ply(path: os.PathLike | str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write the exact vertex schema expected by CGLF's COLMAP loader."""
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N, 3], got {xyz.shape}")
    if rgb.shape != xyz.shape:
        raise ValueError(f"rgb must have shape {xyz.shape}, got {rgb.shape}")
    if not np.isfinite(xyz).all():
        raise ValueError("PLY xyz contains non-finite values")

    vertex_dtype = np.dtype(
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
    vertices = np.zeros(len(xyz), dtype=vertex_dtype)
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, axis]
    for channel, name in enumerate(("red", "green", "blue")):
        vertices[name] = rgb[:, channel]

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment Generated from I3DGS bundle-adjustment landmarks",
            f"element vertex {len(vertices)}",
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

    with Path(path).open("wb") as handle:
        handle.write(header)
        handle.write(vertices.tobytes(order="C"))


def export_cglf_scene(
    scene_model,
    reconstruction_path: os.PathLike | str,
    input_image_paths: Iterable[os.PathLike | str],
    output_path: os.PathLike | str,
    min_observations: int = 2,
) -> dict:
    """Create a new, non-overwriting CGLF scene from an I3DGS reconstruction."""
    reconstruction_path = Path(reconstruction_path).resolve()
    raw_output_path = Path(output_path)
    input_image_paths = list(input_image_paths)
    preflight_cglf_export(raw_output_path, input_image_paths)
    output_path = raw_output_path.resolve()

    sparse_source = reconstruction_path / "sparse" / "0"
    cameras_bin = sparse_source / "cameras.bin"
    images_bin = sparse_source / "images.bin"
    metadata_path = reconstruction_path / "metadata.json"
    for required_path in (cameras_bin, images_bin, metadata_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required I3DGS output is missing: {required_path}")

    input_image_map = _source_image_map(input_image_paths)
    input_image_count = len(input_image_map)

    registered_names = _validated_registered_names(images_bin, metadata_path)
    missing_images = sorted(set(registered_names) - set(input_image_map))
    if missing_images:
        raise FileNotFoundError(
            "Registered source images are missing from this I3DGS input: "
            f"{missing_images}"
        )
    if len(registered_names) > input_image_count:
        raise ValueError(
            f"Registered image count ({len(registered_names)}) exceeds input count "
            f"({input_image_count})"
        )

    xyz, rgb, landmarks_before, landmarks_after = extract_filtered_ba_landmarks(
        scene_model, min_observations=min_observations
    )
    if landmarks_after == 0:
        raise ValueError(
            "No finite BA landmarks satisfy "
            f"n_obs >= {min_observations}; CGLF scene was not created"
        )

    stats = {
        "input_images": input_image_count,
        "registered_images": len(registered_names),
        "registration_rate": len(registered_names) / input_image_count,
        "landmarks_before_filter": landmarks_before,
        "landmarks_after_filter": landmarks_after,
        "minimum_observations": min_observations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.tmp-",
            dir=str(output_path.parent),
        )
    )
    published = False
    try:
        images_output = staging_path / "images"
        sparse_output = staging_path / "sparse" / "0"
        images_output.mkdir(parents=True)
        sparse_output.mkdir(parents=True)

        shutil.copy2(cameras_bin, sparse_output / "cameras.bin")
        shutil.copy2(images_bin, sparse_output / "images.bin")
        for name in registered_names:
            shutil.copy2(input_image_map[name], images_output / name)

        write_cglf_ply(sparse_output / "points3D.ply", xyz, rgb)
        with (staging_path / "export_stats.json").open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
            handle.write("\n")

        # Re-check immediately before publication.  The same-filesystem rename
        # makes the complete scene appear at once instead of exposing partial
        # copies to downstream readers.
        if os.path.lexists(output_path):
            raise FileExistsError(
                f"Refusing to overwrite existing CGLF scene directory: {output_path}"
            )
        staging_path.rename(output_path)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging_path, ignore_errors=True)

    print("CGLF scene export statistics:")
    print(f"  input images: {stats['input_images']}")
    print(f"  registered images: {stats['registered_images']}")
    print(f"  registration rate: {stats['registration_rate']:.2%}")
    print(f"  landmarks before filtering: {stats['landmarks_before_filter']}")
    print(f"  landmarks after filtering: {stats['landmarks_after_filter']}")
    print(f"  output: {output_path}")
    return stats
