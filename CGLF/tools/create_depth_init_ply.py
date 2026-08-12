import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))


def write_ply(path, xyz, rgb):
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz, dtype=np.float32)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attrs = np.concatenate([xyz, normals, rgb], axis=1)
    elements[:] = list(map(tuple, attrs))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def voxel_downsample(xyz, rgb, voxel_size):
    if voxel_size <= 0 or xyz.shape[0] == 0:
        return xyz, rgb

    coords = np.floor(xyz / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return xyz[unique_idx], rgb[unique_idx]


def load_depth(path):
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    if path.endswith(".png") or path.endswith(".tiff") or path.endswith(".tif"):
        return np.array(Image.open(path), dtype=np.float32)
    raise RuntimeError(f"Unsupported depth format: {path}")


def load_confidence(path):
    arr = np.array(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() > 1.0:
        arr /= 255.0
    return arr.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Create Scaffold-GS init PLY from per-view depth.")
    parser.add_argument("--source_path", required=True, type=str)
    parser.add_argument("--output_ply", required=True, type=str)
    parser.add_argument("--depth_dir_name", default="depth", type=str)
    parser.add_argument("--depth_suffix", default=".npy", type=str)
    parser.add_argument("--stride", default=4, type=int)
    parser.add_argument("--max_points_per_view", default=10000, type=int)
    parser.add_argument("--resolution_divisor", default=1, type=int)
    parser.add_argument("--voxel_downsample_size", default=0.0, type=float)
    parser.add_argument("--confidence_dir_name", default="", type=str)
    parser.add_argument("--confidence_suffix", default=".png", type=str)
    parser.add_argument("--confidence_keep_quantile", default=0.0, type=float)
    parser.add_argument("--depth_clip_quantile", default=1.0, type=float)
    args = parser.parse_args()

    source_path = Path(args.source_path)
    transforms_path = source_path / "transforms_train.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing transforms file: {transforms_path}")

    data = json.loads(transforms_path.read_text())
    frames = data["frames"]
    fovx_global = data.get("camera_angle_x", None)

    all_points = []
    all_colors = []
    used_views = 0
    skipped_views = 0

    for frame in frames:
        image_name = frame["file_path"]
        image_path = source_path / f"{image_name}.png"
        if not image_path.exists():
            image_path = source_path / image_name
        depth_path = source_path / args.depth_dir_name / f"{Path(image_name).stem}{args.depth_suffix}"

        if not image_path.exists() or not depth_path.exists():
            skipped_views += 1
            continue

        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        depth = load_depth(depth_path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        confidence = None
        if args.confidence_dir_name:
            conf_path = source_path / args.confidence_dir_name / f"{Path(image_name).stem}{args.confidence_suffix}"
            if conf_path.exists():
                confidence = load_confidence(conf_path)

        h, w = image.shape[:2]
        if depth.shape[:2] != (h, w):
            depth = np.array(
                Image.fromarray(depth).resize((w, h), resample=Image.Resampling.NEAREST),
                dtype=np.float32,
            )
        if confidence is not None and confidence.shape[:2] != (h, w):
            confidence = np.array(
                Image.fromarray(confidence).resize((w, h), resample=Image.Resampling.BILINEAR),
                dtype=np.float32,
            )

        div = max(1, int(args.resolution_divisor))
        if div > 1:
            new_w = max(1, w // div)
            new_h = max(1, h // div)
            image = np.array(
                Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).resize(
                    (new_w, new_h), resample=Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
            depth = np.array(
                Image.fromarray(depth).resize((new_w, new_h), resample=Image.Resampling.NEAREST),
                dtype=np.float32,
            )
            if confidence is not None:
                confidence = np.array(
                    Image.fromarray(confidence).resize((new_w, new_h), resample=Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
            h, w = new_h, new_w

        stride = max(1, int(args.stride))
        ys = np.arange(0, h, stride, dtype=np.int32)
        xs = np.arange(0, w, stride, dtype=np.int32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        u = grid_x.reshape(-1)
        v = grid_y.reshape(-1)
        d = depth[v, u]

        valid = np.isfinite(d) & (d > 0)
        if confidence is not None:
            c = confidence[v, u]
            valid = valid & np.isfinite(c)
        if valid.sum() == 0:
            skipped_views += 1
            continue

        u = u[valid]
        v = v[valid]
        d = d[valid]
        if confidence is not None:
            c = c[valid]

        if confidence is not None and args.confidence_keep_quantile > 0.0:
            q = float(np.clip(args.confidence_keep_quantile, 0.0, 1.0))
            conf_thresh = np.quantile(c, q)
            keep = c >= conf_thresh
            u = u[keep]
            v = v[keep]
            d = d[keep]
            c = c[keep]

        if args.depth_clip_quantile < 1.0 and len(d) > 0:
            dq = float(np.clip(args.depth_clip_quantile, 0.0, 1.0))
            depth_max = np.quantile(d, dq)
            keep = d <= depth_max
            u = u[keep]
            v = v[keep]
            d = d[keep]
            if confidence is not None:
                c = c[keep]

        if len(d) > args.max_points_per_view:
            choice = np.linspace(0, len(d) - 1, args.max_points_per_view, dtype=np.int32)
            u = u[choice]
            v = v[choice]
            d = d[choice]

        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        # Match the Blender-data conversion used by Scaffold-GS dataset_readers.
        c2w[:3, 1:3] *= -1

        if fovx_global is not None:
            fovx = float(fovx_global)
        else:
            fovx = 2.0 * np.arctan(w / (2.0 * float(frame["fl_x"])))
        if "fl_y" in frame:
            fovy = 2.0 * np.arctan(h / (2.0 * float(frame["fl_y"])))
        else:
            fovy = 2.0 * np.arctan(h / (2.0 * fov2focal(fovx, w)))

        fx = fov2focal(fovx, w)
        fy = fov2focal(fovy, h)
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5

        x = (u.astype(np.float32) - cx) / fx * d
        y = (v.astype(np.float32) - cy) / fy * d
        z = d
        points_cam = np.stack([x, y, z], axis=1)

        points_world = (c2w[:3, :3] @ points_cam.T).T + c2w[:3, 3]
        colors = image[v, u]

        all_points.append(points_world.astype(np.float32))
        all_colors.append(colors.astype(np.float32))
        used_views += 1

    if not all_points:
        raise RuntimeError("No valid depth points found for init.")

    xyz = np.concatenate(all_points, axis=0)
    rgb = np.clip(np.concatenate(all_colors, axis=0) * 255.0, 0, 255).astype(np.uint8)
    xyz, rgb = voxel_downsample(xyz, rgb, float(args.voxel_downsample_size))

    output_ply = Path(args.output_ply)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    write_ply(str(output_ply), xyz, rgb)

    print(
        f"Depth init PLY written: {output_ply} | "
        f"points={xyz.shape[0]} used_views={used_views} skipped_views={skipped_views}"
    )


if __name__ == "__main__":
    main()
