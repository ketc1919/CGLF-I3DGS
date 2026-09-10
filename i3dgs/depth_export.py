"""Export final I3DGS BA-aligned monocular Z-depth maps.

This module deliberately operates on the live ``SceneModel``.  The regular
I3DGS checkpoint does not persist the BA observation table, while the final
``SceneModel`` still owns the landmarks, their 2-D observations, final camera
poses, and the already-initialised Depth Anything V2 estimator.

The exported arrays are camera-coordinate Z values in the reconstruction's
arbitrary scene unit.  Depth Anything V2 supplies a dense relative
inverse-depth-like map; final BA observations supply the scale and offset used
to align that map.  No pose estimation, BA update, Gaussian rendering, or
in-place mutation is performed here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path

import numpy as np


class DepthExportError(RuntimeError):
    """Raised when a frame cannot produce a geometrically defined depth map."""


def _as_numpy(value, dtype=None) -> np.ndarray:
    """Detach a NumPy/torch-like value without modifying the source object."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _first_basename(name: str) -> str:
    """Match CGLF's ``basename.split('.')[0]`` image-name convention."""

    basename = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    stem = basename.split(".", 1)[0]
    if not stem:
        raise DepthExportError(f"Image name has an empty CGLF stem: {name!r}")
    return stem


def _keyframe_name(keyframe) -> str:
    info = getattr(keyframe, "info", {})
    name = info.get("name") if isinstance(info, Mapping) else None
    if not isinstance(name, str) or not name:
        raise DepthExportError(f"Keyframe {getattr(keyframe, 'index', '?')} has no image name")
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def _camera_values(keyframe) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Read the final world-to-camera values from one keyframe."""

    R = _as_numpy(keyframe.get_R(), np.float64).reshape(3, 3)
    t = _as_numpy(keyframe.get_t(), np.float64).reshape(3)
    f = float(_as_numpy(keyframe.f).reshape(-1)[0])
    centre = _as_numpy(getattr(keyframe, "centre", None), np.float64).reshape(-1)
    if centre.size != 2:
        width = int(keyframe.width)
        height = int(keyframe.height)
        centre = np.array([(width - 1) * 0.5, (height - 1) * 0.5], dtype=np.float64)
    if not np.isfinite(R).all() or not np.isfinite(t).all() or not np.isfinite(f):
        raise DepthExportError(f"Keyframe {_keyframe_name(keyframe)} has non-finite camera values")
    if f <= 0:
        raise DepthExportError(f"Keyframe {_keyframe_name(keyframe)} has non-positive focal length")
    return R, t, f, centre


def collect_ba_correspondences(
    scene_model,
    keyframe,
    *,
    min_observations: int = 2,
    reprojection_threshold_px: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Collect final BA XYZ/UV pairs belonging to one keyframe.

    The function reads only ``[:size]`` and ``[:obs_size]`` from the BA
    storage.  It keeps UV and XYZ under one shared mask, maps observations by
    ``keyframe.index`` rather than by an accidental list position, rejects
    non-finite/behind-camera/reprojection-outlier pairs, and deterministically
    keeps the first observation for a repeated landmark in one frame.
    """

    if min_observations < 1:
        raise ValueError("min_observations must be positive")
    if reprojection_threshold_px <= 0:
        raise ValueError("reprojection_threshold_px must be positive")

    ba = scene_model.ba_problem
    landmark_count = int(ba.size)
    obs_count = int(ba.obs_size)
    xyz = _as_numpy(ba.landmarks[:landmark_count], np.float64)
    n_obs = _as_numpy(ba.n_obs[:landmark_count], np.int64)
    lm_ids = _as_numpy(ba.obs_lm_ids[:obs_count], np.int64)
    kf_ids = _as_numpy(ba.obs_kf_ids[:obs_count], np.int64)
    pt2d_ids = _as_numpy(ba.obs_pt2d_ids[:obs_count], np.int64)
    uvs = _as_numpy(ba.obs_uvs[:obs_count], np.float64)

    if xyz.shape != (landmark_count, 3) or n_obs.shape != (landmark_count,):
        raise DepthExportError("BA landmark arrays have unexpected shapes")
    if lm_ids.shape != (obs_count,) or kf_ids.shape != (obs_count,) or pt2d_ids.shape != (obs_count,):
        raise DepthExportError("BA observation index arrays have unexpected shapes")
    if uvs.shape != (obs_count, 2):
        raise DepthExportError("BA observation UV array has an unexpected shape")

    frame_id = int(keyframe.index)
    width, height = int(keyframe.width), int(keyframe.height)
    base = (
        (lm_ids >= 0)
        & (lm_ids < landmark_count)
        & (kf_ids == frame_id)
        & (pt2d_ids >= 0)
        & np.isfinite(uvs).all(axis=1)
        & (uvs[:, 0] >= 0)
        & (uvs[:, 0] <= width - 1)
        & (uvs[:, 1] >= 0)
        & (uvs[:, 1] <= height - 1)
    )
    base &= np.isfinite(n_obs[np.clip(lm_ids, 0, max(landmark_count - 1, 0))]) if landmark_count else False
    if landmark_count:
        base &= n_obs[np.clip(lm_ids, 0, landmark_count - 1)] >= min_observations
        base &= np.isfinite(xyz[np.clip(lm_ids, 0, landmark_count - 1)]).all(axis=1)

    candidate_ids = np.flatnonzero(base)
    if candidate_ids.size:
        # np.unique returns the first occurrence after sorting the landmark ID;
        # sorting those first positions restores the original deterministic BA order.
        _, first_positions = np.unique(lm_ids[candidate_ids], return_index=True)
        candidate_ids = candidate_ids[np.sort(first_positions)]

    if not candidate_ids.size:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            {"observations_total": obs_count, "candidates": 0, "kept": 0},
        )

    R, t, f, centre = _camera_values(keyframe)
    points = xyz[lm_ids[candidate_ids]]
    camera_points = points @ R.T + t
    z = camera_points[:, 2]
    projected = np.empty_like(uvs[candidate_ids])
    projected[:, 0] = f * camera_points[:, 0] / z + centre[0]
    projected[:, 1] = f * camera_points[:, 1] / z + centre[1]
    reprojection_error = np.linalg.norm(projected - uvs[candidate_ids], axis=1)
    keep = np.isfinite(z) & (z > 0) & np.isfinite(reprojection_error)
    keep &= reprojection_error <= reprojection_threshold_px

    kept_ids = candidate_ids[keep]
    stats = {
        "observations_total": obs_count,
        "candidates": int(candidate_ids.size),
        "kept": int(kept_ids.size),
        "reprojection_threshold_px": float(reprojection_threshold_px),
        "reprojection_median_px": float(np.median(reprojection_error[keep])) if np.any(keep) else None,
    }
    return uvs[kept_ids].copy(), xyz[lm_ids[kept_ids]].copy(), stats


def camera_z_from_world(xyz_world: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Transform world XYZ to camera coordinates and return the optical-axis Z."""

    xyz_world = np.asarray(xyz_world, dtype=np.float64)
    camera = xyz_world @ np.asarray(R, dtype=np.float64).reshape(3, 3).T
    camera = camera + np.asarray(t, dtype=np.float64).reshape(1, 3)
    return camera[:, 2]


def _affine_fit(target: np.ndarray, mono: np.ndarray) -> tuple[float, float]:
    """Fit ``target ~= scale * mono + offset`` using current I3DGS statistics."""

    target = np.asarray(target, dtype=np.float64)
    mono = np.asarray(mono, dtype=np.float64)
    target_median = np.median(target)
    mono_median = np.median(mono)
    target_scale = np.mean(np.abs(target - target_median))
    mono_scale = max(float(np.mean(np.abs(mono - mono_median))), 1e-12)
    scale = target_scale / mono_scale
    offset = target_median - mono_median * scale
    return float(scale), float(offset)


def fit_inverse_depth_affine(
    mono_samples: np.ndarray,
    target_samples: np.ndarray,
    *,
    min_correspondences: int = 8,
    outlier_mult: float = 5.0,
) -> tuple[float, float, np.ndarray, dict]:
    """Robustly fit the existing I3DGS median/MAD inverse-depth alignment."""

    mono = np.asarray(mono_samples, dtype=np.float64).reshape(-1)
    target = np.asarray(target_samples, dtype=np.float64).reshape(-1)
    finite = np.isfinite(mono) & np.isfinite(target)
    if int(finite.sum()) < min_correspondences:
        raise DepthExportError(
            f"only {int(finite.sum())} finite depth correspondences; "
            f"need {min_correspondences}"
        )
    mono = mono[finite]
    target = target[finite]
    if np.mean(np.abs(mono - np.median(mono))) <= 1e-12:
        raise DepthExportError("monocular inverse-depth samples are degenerate")

    scale, offset = _affine_fit(target, mono)
    residual = np.abs(scale * mono + offset - target)
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    threshold = med + outlier_mult * mad
    keep = residual <= max(threshold, 1e-12)
    if int(keep.sum()) < min_correspondences:
        keep = np.ones_like(keep, dtype=bool)
    scale, offset = _affine_fit(target[keep], mono[keep])
    aligned = scale * mono[keep] + offset
    if not np.isfinite(scale) or not np.isfinite(offset) or scale <= 0:
        raise DepthExportError("inverse-depth alignment produced invalid scale/offset")
    if not np.isfinite(aligned).all() or np.any(aligned <= 0):
        raise DepthExportError("inverse-depth alignment produced non-positive values")
    stats = {
        "correspondences_finite": int(finite.sum()),
        "correspondences_used": int(keep.sum()),
        "scale": float(scale),
        "offset": float(offset),
        "residual_median": float(np.median(np.abs(aligned - target[keep]))),
        "residual_p90": float(np.percentile(np.abs(aligned - target[keep]), 90)),
        "mode": "global_median_mean_abs_deviation",
    }
    return scale, offset, keep, stats


def _bilinear_sample(array: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample a 2-D array at floating coordinates with edge clamping."""

    h, w = array.shape
    x = np.clip(np.asarray(x, dtype=np.float64), 0, w - 1)
    y = np.clip(np.asarray(y, dtype=np.float64), 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = x - x0
    wy = y - y0
    return (
        array[y0, x0] * (1 - wx) * (1 - wy)
        + array[y0, x1] * wx * (1 - wy)
        + array[y1, x0] * (1 - wx) * wy
        + array[y1, x1] * wx * wy
    )


def resample_inverse_depth_to_target(
    aligned_inverse_depth: np.ndarray,
    *,
    processed_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map processed-image inverse depth to target RGB pixels.

    The mapping follows the current half-pixel convention:
    ``u_p=(u_t+0.5)*W_p/W_t-0.5``.  The returned mask marks finite positive
    inverse-depth values; callers convert those values to camera Z.
    """

    source = np.asarray(aligned_inverse_depth, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("aligned_inverse_depth must be a 2-D array")
    processed_w, processed_h = map(int, processed_size)
    target_w, target_h = map(int, target_size)
    if min(processed_w, processed_h, target_w, target_h) <= 0:
        raise ValueError("image dimensions must be positive")
    ys, xs = np.indices((target_h, target_w), dtype=np.float64)
    x_processed = (xs + 0.5) * processed_w / target_w - 0.5
    y_processed = (ys + 0.5) * processed_h / target_h - 0.5
    x_source = x_processed * (source.shape[1] - 1) / max(processed_w - 1, 1)
    y_source = y_processed * (source.shape[0] - 1) / max(processed_h - 1, 1)
    sampled = _bilinear_sample(source, x_source, y_source)
    valid = np.isfinite(sampled) & (sampled > 1e-8)
    return sampled.astype(np.float32), valid


def _image_size(path: Path) -> tuple[int, int]:
    """Read an RGB file's width/height without changing its pixels."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - project requirements include Pillow
        raise DepthExportError("Pillow is required to read exported image sizes") from exc
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _align_raw_depth(
    raw: np.ndarray,
    mono_samples: np.ndarray,
    target_samples: np.ndarray,
    uv: np.ndarray,
    *,
    keyframe,
    scene_model,
    min_correspondences: int,
    outlier_mult: float,
) -> tuple[np.ndarray, dict]:
    """Align a raw map globally and, when configured, with I3DGS grid blocks."""

    finite = np.isfinite(mono_samples) & np.isfinite(target_samples)
    mono_finite = np.asarray(mono_samples)[finite]
    target_finite = np.asarray(target_samples)[finite]
    uv_finite = np.asarray(uv)[finite]
    scale, offset, used, stats = fit_inverse_depth_affine(
        mono_finite,
        target_finite,
        min_correspondences=min_correspondences,
        outlier_mult=outlier_mult,
    )
    aligned_raw = scale * raw + offset
    args = getattr(scene_model, "args", None)
    grid_size = int(getattr(args, "depth_grid_size", 0) or 0)
    if grid_size <= 0:
        return aligned_raw, stats

    # The production estimator is CUDA-backed.  This path reuses the same
    # I3DGS block alignment implementation used during Gaussian initialisation;
    # synthetic tests leave depth_grid_size unset and therefore use the pure
    # global path above.
    try:
        import torch
        from scene.mono_depth import align_depth_grid

        image = keyframe.image
        device = image.device if hasattr(image, "device") else torch.device("cuda")
        mono_map = torch.as_tensor(raw, dtype=torch.float32, device=device)[None, None]
        uv_t = torch.as_tensor(uv_finite[used], dtype=torch.float32, device=device)
        sampled_t = torch.as_tensor(mono_finite[used], dtype=torch.float32, device=device)
        target_t = torch.as_tensor(target_finite[used], dtype=torch.float32, device=device)
        aligned = align_depth_grid(
            mono_map,
            uv_t,
            int(keyframe.width),
            int(keyframe.height),
            sampled_t,
            target_t,
            grid_size,
            torch.tensor(scale, dtype=torch.float32, device=device),
            torch.tensor(offset, dtype=torch.float32, device=device),
            min_pts_per_block=int(getattr(args, "depth_grid_min_pts_per_block", 10)),
            min_kpts_per_block=int(getattr(args, "depth_grid_min_kpts_per_block", 30)),
            scale_tolerance=float(getattr(args, "depth_grid_scale_tolerance", 0.5)),
        )
        aligned_raw = _as_numpy(aligned, np.float64).reshape(raw.shape)
        stats = dict(stats)
        stats["mode"] = "grid_median_mean_abs_deviation"
        stats["grid_size"] = grid_size
        stats["correspondences_used_for_grid"] = int(used.sum())
        return aligned_raw, stats
    except Exception as exc:
        raise DepthExportError(
            f"grid inverse-depth alignment failed for {_keyframe_name(keyframe)}"
        ) from exc


def _predict_depth(estimator, keyframe) -> np.ndarray:
    """Run the already-created estimator once and return a 2-D CPU array."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - I3DGS requires torch
        raise DepthExportError("PyTorch is required for DA V2 depth export") from exc
    image = keyframe.image
    if not hasattr(image, "dim"):
        image = torch.as_tensor(image)
    if image.dim() == 3:
        image = image[None]
    with torch.no_grad():
        result = estimator(image)
    raw = result[0] if isinstance(result, (tuple, list)) else result
    raw = _as_numpy(raw, np.float64)
    if raw.ndim == 4 and raw.shape[1] == 1:
        raw = raw[0, 0]
    elif raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 2 or min(raw.shape) <= 1:
        raise DepthExportError(f"DA V2 returned an unexpected depth shape: {raw.shape}")
    return raw


def export_depth_maps(
    scene_model,
    registered_names: Iterable[str],
    source_image_map: Mapping[str, Path],
    output_dir: Path,
    *,
    min_observations: int = 2,
    reprojection_threshold_px: float = 3.0,
    min_correspondences: int = 8,
    outlier_mult: float = 5.0,
) -> dict:
    """Write one Stage-2-compatible Z-depth array per registered image.

    ``output_dir`` must be inside the adapter's temporary staging directory.
    The caller publishes that staging directory only after this function
    returns successfully, so a failed frame cannot expose a partial scene.
    """

    output_dir = Path(output_dir)
    names = list(registered_names)
    if not names:
        raise DepthExportError("No registered images are available for depth export")
    estimator = getattr(scene_model, "depth_estimator", None)
    if estimator is None:
        raise DepthExportError("SceneModel has no initialized depth_estimator")

    keyframes_by_name = {_keyframe_name(kf): kf for kf in scene_model.keyframes}
    depth_names = [_first_basename(name) + ".npy" for name in names]
    if len(set(depth_names)) != len(depth_names):
        raise DepthExportError("Registered images collide under Stage-2 depth naming")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_stats = []
    for registered_name, depth_name in zip(names, depth_names):
        basename = Path(registered_name).name
        keyframe = keyframes_by_name.get(basename)
        if keyframe is None:
            raise DepthExportError(f"No live keyframe matches registered image {registered_name}")
        source_path = source_image_map.get(basename)
        if source_path is None:
            raise DepthExportError(f"No source image matches registered image {registered_name}")

        uv, xyz_world, corr_stats = collect_ba_correspondences(
            scene_model,
            keyframe,
            min_observations=min_observations,
            reprojection_threshold_px=reprojection_threshold_px,
        )
        if len(uv) < min_correspondences:
            raise DepthExportError(
                f"{registered_name}: only {len(uv)} usable final BA correspondences"
            )
        R, t, _, _ = _camera_values(keyframe)
        z = camera_z_from_world(xyz_world, R, t)
        positive = np.isfinite(z) & (z > 0)
        if int(positive.sum()) < min_correspondences:
            raise DepthExportError(f"{registered_name}: insufficient positive BA Z values")
        uv = uv[positive]
        z = z[positive]
        raw = _predict_depth(estimator, keyframe)
        mono_samples = _bilinear_sample(
            raw,
            uv[:, 0] * (raw.shape[1] - 1) / max(int(keyframe.width) - 1, 1),
            uv[:, 1] * (raw.shape[0] - 1) / max(int(keyframe.height) - 1, 1),
        )
        aligned_raw, align_stats = _align_raw_depth(
            raw,
            mono_samples,
            1.0 / z,
            uv,
            keyframe=keyframe,
            scene_model=scene_model,
            min_correspondences=min_correspondences,
            outlier_mult=outlier_mult,
        )
        processed_size = (int(keyframe.width), int(keyframe.height))
        target_size = _image_size(Path(source_path))
        target_inverse_depth, valid = resample_inverse_depth_to_target(
            aligned_raw,
            processed_size=processed_size,
            target_size=target_size,
        )
        depth_z = np.zeros(target_inverse_depth.shape, dtype=np.float32)
        valid &= np.isfinite(target_inverse_depth) & (target_inverse_depth > 1e-8)
        depth_z[valid] = 1.0 / target_inverse_depth[valid]
        if not np.any(valid):
            raise DepthExportError(f"{registered_name}: aligned depth has no valid pixels")
        np.save(output_dir / depth_name, depth_z.astype(np.float32), allow_pickle=False)
        frame_stats.append({
            "image": registered_name,
            "depth_file": depth_name,
            "processed_size": [processed_size[0], processed_size[1]],
            "prediction_size": [int(raw.shape[1]), int(raw.shape[0])],
            "target_size": [target_size[0], target_size[1]],
            "correspondences": corr_stats,
            "positive_ba_z": int(positive.sum()),
            "alignment": align_stats,
            "valid_pixels": int(valid.sum()),
            "valid_fraction": float(valid.mean()),
            "z_percentiles": [float(v) for v in np.percentile(depth_z[valid], [1, 50, 99])],
            "depth_definition": "camera_coordinate_z",
            "scene_unit": "I3DGS/BA arbitrary scene unit",
        })

    stats = {
        "depth_definition": "camera_coordinate_z",
        "source": "Depth Anything V2 relative inverse depth aligned to final BA",
        "scene_unit": "I3DGS/BA arbitrary scene unit; not meters",
        "invalid_value": 0.0,
        "files": frame_stats,
    }
    with (output_dir.parent / "depth_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return stats
