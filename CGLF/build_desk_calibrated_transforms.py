from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import re

import cv2
import matio
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(".")
CALIB_DIR = ROOT / "Calibration_mat"
OUT_DIR = ROOT / "3dgs_undistorted"
IMAGE_DIR = OUT_DIR / "images"
GRID_W = 10
GRID_H = 10
ANCHOR_POSE_ID = 1


def calibration_path(view_id: int) -> Path:
    p = CALIB_DIR / f"calibration{view_id:02d}.mat"
    if p.exists():
        return p
    p = CALIB_DIR / f"calibration{view_id}.mat"
    if p.exists():
        return p
    raise FileNotFoundError(f"Missing calibration file for view {view_id}")


def read_calibration_session(view_id: int):
    data = matio.load_from_mat(calibration_path(view_id))
    session = data["calibrationSession"]
    row = session.properties["any"][0, 0]
    camera_params = row["cameraParams"].properties["any"][0, 0]
    board_set = row["boardSet"]
    labels = []
    for x in board_set.properties["BoardLabels"].reshape(-1):
        s = str(x.reshape(-1)[0]) if hasattr(x, "reshape") else str(x)
        labels.append(s)
    return camera_params, labels


def mat_to_intrinsics(camera_params, target_width=1920, target_height=1056) -> dict:
    matlab_k = camera_params["IntrinsicMatrix"].astype(float)
    fx = float(matlab_k[0, 0])
    fy = float(matlab_k[1, 1])
    cx = float(matlab_k[2, 0])
    cy = float(matlab_k[2, 1])
    k1, k2 = [float(v) for v in camera_params["RadialDistortion"].reshape(-1)]
    p1, p2 = [float(v) for v in camera_params["TangentialDistortion"].reshape(-1)]
    h, w = [int(v) for v in camera_params["ImageSize"].reshape(-1)]
    sx = target_width / float(w)
    sy = target_height / float(h)
    k = np.array([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)
    # Keep the original calibrated K after resolution scaling. OpenCV's
    # getOptimalNewCameraMatrix can return pathological focal lengths for a few
    # views in this dataset, causing an extreme crop.
    new_k = k
    return {
        "fl_x": float(new_k[0, 0]),
        "fl_y": float(new_k[1, 1]),
        "cx": float(new_k[0, 2]),
        "cy": float(new_k[1, 2]),
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
    }


def rt_to_mat(rotation_vector: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
    r, _ = cv2.Rodrigues(rotation_vector.astype(float).reshape(3, 1))
    t = translation_vector.astype(float).reshape(3)
    m = np.eye(4, dtype=float)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    if len(transforms) == 1:
        return transforms[0]
    rots = Rotation.from_matrix(np.stack([t[:3, :3] for t in transforms], axis=0))
    trans = np.stack([t[:3, 3] for t in transforms], axis=0)
    out = np.eye(4, dtype=float)
    out[:3, :3] = rots.mean().as_matrix()
    out[:3, 3] = np.median(trans, axis=0)
    return out


def build_observations():
    observations = []
    intrinsics = {}
    for view_id in range(1, GRID_W * GRID_H + 1):
        camera_params, labels = read_calibration_session(view_id)
        intrinsics[view_id] = mat_to_intrinsics(camera_params)
        rvecs = camera_params["RotationVectors"].astype(float)
        tvecs = camera_params["TranslationVectors"].astype(float)
        if len(labels) != rvecs.shape[0]:
            raise RuntimeError(f"Label/extrinsic count mismatch for camera {view_id}")
        for label, rvec, tvec in zip(labels, rvecs, tvecs):
            m = re.search(r"_(\d+)\.png$", label)
            if not m:
                continue
            pose_id = int(m.group(1))
            # MATLAB calibration extrinsic maps checkerboard coordinates to camera coordinates.
            t_cam_board = rt_to_mat(rvec, tvec)
            observations.append((view_id, pose_id, t_cam_board))
    return observations, intrinsics


def solve_camera_rig(observations):
    by_camera_pose = {}
    pose_counts = {}
    for cam_id, pose_id, t_cam_board in observations:
        by_camera_pose[(cam_id, pose_id)] = t_cam_board
        pose_counts[pose_id] = pose_counts.get(pose_id, 0) + 1

    if ANCHOR_POSE_ID not in pose_counts:
        raise RuntimeError(f"Anchor board pose _{ANCHOR_POSE_ID:02d} was not observed")

    # Use one physical checkerboard moment as the world coordinate system.
    # For cameras that saw that moment, c2w is exactly inv(board_to_camera).
    board_to_world = {ANCHOR_POSE_ID: np.eye(4, dtype=float)}
    cam_to_world = {}
    source = {}
    for cam_id in range(1, 101):
        t_cam_anchor = by_camera_pose.get((cam_id, ANCHOR_POSE_ID))
        if t_cam_anchor is not None:
            cam_to_world[cam_id] = np.linalg.inv(t_cam_anchor)
            source[cam_id] = f"direct_anchor_pose_{ANCHOR_POSE_ID:02d}"

    # Derive all other checkerboard poses from the directly anchored cameras.
    for _ in range(20):
        changed = False
        board_estimates = {}
        for (cam_id, pose_id), t_cam_board in by_camera_pose.items():
            if cam_id in cam_to_world:
                board_estimates.setdefault(pose_id, []).append(cam_to_world[cam_id] @ t_cam_board)
        for pose_id, estimates in board_estimates.items():
            if not estimates:
                continue
            new_t = np.eye(4, dtype=float) if pose_id == ANCHOR_POSE_ID else average_transforms(estimates)
            if pose_id not in board_to_world:
                changed = True
            board_to_world[pose_id] = new_t

        # Fill cameras missing the anchor pose through shared labelled board poses.
        for cam_id in range(1, 101):
            if cam_id in cam_to_world:
                continue
            estimates = []
            used_poses = []
            for (obs_cam_id, pose_id), t_cam_board in by_camera_pose.items():
                if obs_cam_id != cam_id or pose_id not in board_to_world:
                    continue
                estimates.append(board_to_world[pose_id] @ np.linalg.inv(t_cam_board))
                used_poses.append(pose_id)
            if estimates:
                cam_to_world[cam_id] = average_transforms(estimates)
                source[cam_id] = "inferred_from_shared_poses_" + ",".join(f"{p:02d}" for p in sorted(used_poses))
                changed = True

        if not changed:
            break

    missing = [i for i in range(1, 101) if i not in cam_to_world]
    if missing:
        raise RuntimeError(f"Could not solve all camera poses, missing {missing}")
    return cam_to_world, board_to_world, source, pose_counts


def opencv_to_opengl_c2w(c2w: np.ndarray) -> np.ndarray:
    convert = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w @ convert


def main() -> None:
    global ROOT, CALIB_DIR, OUT_DIR, IMAGE_DIR, GRID_W, GRID_H, ANCHOR_POSE_ID
    parser = argparse.ArgumentParser(description="Build calibrated transforms for the desk light-field capture.")
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--grid_w", default=GRID_W, type=int)
    parser.add_argument("--grid_h", default=GRID_H, type=int)
    parser.add_argument("--anchor_pose_id", default=ANCHOR_POSE_ID, type=int)
    args = parser.parse_args()
    ROOT = Path(args.root).resolve()
    CALIB_DIR = ROOT / "Calibration_mat"
    OUT_DIR = ROOT / "3dgs_undistorted"
    IMAGE_DIR = OUT_DIR / "images"
    GRID_W = args.grid_w
    GRID_H = args.grid_h
    ANCHOR_POSE_ID = args.anchor_pose_id
    observations, intrinsics = build_observations()
    cam_to_world, board_to_world, source, pose_counts = solve_camera_rig(observations)

    centers = np.stack([cam_to_world[i][:3, 3] for i in range(1, 101)], axis=0)
    center_mean = centers.mean(axis=0)
    # Keep real millimeter relative scale but center the rig for numerical convenience.
    centered_cam_to_world = {}
    for i, t in cam_to_world.items():
        tc = t.copy()
        tc[:3, 3] -= center_mean
        centered_cam_to_world[i] = tc

    frames = []
    for view_id in range(1, 101):
        c2w_cv = centered_cam_to_world[view_id]
        c2w_gl = opencv_to_opengl_c2w(c2w_cv)
        intr = intrinsics[view_id]
        row = (view_id - 1) // GRID_W
        col = (view_id - 1) % GRID_W
        frames.append(
            {
                "file_path": f"images/{view_id}.jpg",
                "view_id": view_id,
                "grid_row": row,
                "grid_col": col,
                "w": 1920,
                "h": 1056,
                "fl_x": intr["fl_x"],
                "fl_y": intr["fl_y"],
                "cx": intr["cx"],
                "cy": intr["cy"],
                "k1": intr["k1"],
                "k2": intr["k2"],
                "p1": intr["p1"],
                "p2": intr["p2"],
                "camera_angle_x": 2.0 * math.atan(1920 / (2.0 * intr["fl_x"])),
                "camera_angle_y": 2.0 * math.atan(1056 / (2.0 * intr["fl_y"])),
                "distortion_source": {
                    "k1": intr["k1"],
                    "k2": intr["k2"],
                    "p1": intr["p1"],
                    "p2": intr["p2"],
                },
                "image_processing": "original_frame_no_warp",
                "extrinsics_source": source[view_id],
                "extrinsics_anchor_board_pose": f"_{ANCHOR_POSE_ID:02d}.png",
                "transform_matrix": c2w_gl.tolist(),
                "transform_matrix_opencv": c2w_cv.tolist(),
            }
        )

    fl_x = float(np.median([f["fl_x"] for f in frames]))
    fl_y = float(np.median([f["fl_y"] for f in frames]))
    cx = float(np.median([f["cx"] for f in frames]))
    cy = float(np.median([f["cy"] for f in frames]))
    data = {
        "camera_model": "OPENCV",
        "w": 1920,
        "h": 1056,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "k1": float(np.median([f["k1"] for f in frames])),
        "k2": float(np.median([f["k2"] for f in frames])),
        "p1": float(np.median([f["p1"] for f in frames])),
        "p2": float(np.median([f["p2"] for f in frames])),
        "camera_angle_x": 2.0 * math.atan(1920 / (2.0 * fl_x)),
        "camera_angle_y": 2.0 * math.atan(1056 / (2.0 * fl_y)),
        "coordinate_note": "camera_model is OPENCV and each frame includes its own k1/k2/p1/p2 from the MATLAB calibration. transform_matrix is NeRF/OpenGL convention. transform_matrix_opencv keeps MATLAB/OpenCV camera convention. Extrinsics are label-aligned by BoardLabels: checkerboard pose _01 is the world frame, cameras observing _01 use inv(board_to_camera) directly, and cameras missing _01 are inferred only through shared labelled checkerboard poses. Units remain millimeters. Images are original extracted frames with no extra undistortion warp; the 3DGS loader must support OPENCV distortion for k1/k2/p1/p2 to take effect.",
        "extrinsics_anchor_board_pose": f"_{ANCHOR_POSE_ID:02d}.png",
        "board_pose_observation_counts": {f"{k:02d}": int(v) for k, v in sorted(pose_counts.items())},
        "source_observations": len(observations),
        "num_board_poses": len(board_to_world),
        "frames": frames,
    }
    (OUT_DIR / "transforms.json").write_text(json.dumps(data, indent=2), encoding="ascii")

    with (OUT_DIR / "camera_centers_mm.txt").open("w", encoding="ascii") as f:
        f.write("# view_id x_mm y_mm z_mm grid_row grid_col extrinsics_source\n")
        for view_id in range(1, 101):
            c = centered_cam_to_world[view_id][:3, 3]
            row = (view_id - 1) // GRID_W
            col = (view_id - 1) % GRID_W
            f.write(f"{view_id} {c[0]:.12g} {c[1]:.12g} {c[2]:.12g} {row} {col} {source[view_id]}\n")

    print("wrote", OUT_DIR / "transforms.json")
    print("anchor board pose", f"_{ANCHOR_POSE_ID:02d}", "direct cameras", pose_counts.get(ANCHOR_POSE_ID, 0))
    print("observations", len(observations), "board poses", len(board_to_world))
    print("camera center min", centers.min(axis=0))
    print("camera center max", centers.max(axis=0))
    print("centered span", (centers.max(axis=0) - centers.min(axis=0)))
    print("median intrinsics", fl_x, fl_y, cx, cy)


if __name__ == "__main__":
    main()
