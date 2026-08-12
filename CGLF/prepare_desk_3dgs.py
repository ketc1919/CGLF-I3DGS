from __future__ import annotations

import argparse
from pathlib import Path
import json
import math

import cv2
import matio
import numpy as np


ROOT = Path(".")
FRAME_DIR = ROOT / "frames"
CALIB_DIR = ROOT / "Calibration_mat"
OUT_DIR = ROOT / "3dgs_undistorted"
OUT_IMAGE_DIR = OUT_DIR / "images"
GRID_W = 10
GRID_H = 10
APPLY_UNDISTORT = False


def calibration_path(view_id: int) -> Path:
    p = CALIB_DIR / f"calibration{view_id:02d}.mat"
    if p.exists():
        return p
    p = CALIB_DIR / f"calibration{view_id}.mat"
    if p.exists():
        return p
    raise FileNotFoundError(f"Missing calibration file for view {view_id}")


def read_calibration(view_id: int) -> dict:
    data = matio.load_from_mat(calibration_path(view_id))
    session = data["calibrationSession"]
    camera_params = session.properties["any"][0, 0]["cameraParams"].properties["any"][0, 0]

    matlab_k = camera_params["IntrinsicMatrix"].astype(float)
    # MATLAB cameraParameters stores K transposed relative to the OpenCV convention:
    # [fx 0 0; 0 fy 0; cx cy 1].
    fx = float(matlab_k[0, 0])
    fy = float(matlab_k[1, 1])
    cx = float(matlab_k[2, 0])
    cy = float(matlab_k[2, 1])
    k1, k2 = [float(v) for v in camera_params["RadialDistortion"].reshape(-1)]
    p1, p2 = [float(v) for v in camera_params["TangentialDistortion"].reshape(-1)]
    h, w = [int(v) for v in camera_params["ImageSize"].reshape(-1)]

    return {
        "view_id": view_id,
        "K": np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64),
        "dist": np.array([k1, k2, p1, p2, 0.0], dtype=np.float64),
        "width": w,
        "height": h,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
    }


def scale_calibration_to_image(calib: dict, width: int, height: int) -> dict:
    if (calib["width"], calib["height"]) == (width, height):
        return calib
    sx = width / float(calib["width"])
    sy = height / float(calib["height"])
    calib = dict(calib)
    k = calib["K"].copy()
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] *= sx
    k[1, 2] *= sy
    calib["K"] = k
    calib["fx"] *= sx
    calib["fy"] *= sy
    calib["cx"] *= sx
    calib["cy"] *= sy
    calib["width"] = width
    calib["height"] = height
    calib["resolution_scaled"] = True
    return calib


def view_to_grid(view_id: int) -> tuple[int, int]:
    idx = view_id - 1
    return idx // GRID_W, idx % GRID_W


def c2w_for_view(view_id: int) -> np.ndarray:
    row, col = view_to_grid(view_id)
    # Approximate 10x10 light-field camera array. The calibration sessions contain
    # per-camera intrinsics and calibration-board poses, but not a shared rig pose.
    # Use a centered unit-baseline planar rig, with all optical axes parallel.
    x = col - (GRID_W - 1) / 2.0
    y = row - (GRID_H - 1) / 2.0
    z = 0.0
    mat = np.eye(4, dtype=float)
    mat[:3, 3] = [x, y, z]
    return mat


def main() -> None:
    global ROOT, FRAME_DIR, CALIB_DIR, OUT_DIR, OUT_IMAGE_DIR, GRID_W, GRID_H, APPLY_UNDISTORT
    parser = argparse.ArgumentParser(description="Prepare desk light-field frames and transforms for 3DGS-style training.")
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--grid_w", default=GRID_W, type=int)
    parser.add_argument("--grid_h", default=GRID_H, type=int)
    parser.add_argument("--apply_undistort", action="store_true")
    args = parser.parse_args()
    ROOT = Path(args.root).resolve()
    FRAME_DIR = ROOT / "frames"
    CALIB_DIR = ROOT / "Calibration_mat"
    OUT_DIR = ROOT / "3dgs_undistorted"
    OUT_IMAGE_DIR = OUT_DIR / "images"
    GRID_W = args.grid_w
    GRID_H = args.grid_h
    APPLY_UNDISTORT = bool(args.apply_undistort)
    OUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    calibrations: list[dict] = []
    frames: list[dict] = []

    for view_id in range(1, GRID_W * GRID_H + 1):
        calib = read_calibration(view_id)
        calibrations.append(calib)

        src = FRAME_DIR / f"{view_id}.jpg"
        if not src.exists():
            raise FileNotFoundError(src)
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read {src}")
        h, w = img.shape[:2]
        calib = scale_calibration_to_image(calib, w, h)

        # The extracted video frames are already in a visually corrected image
        # space. Applying the MATLAB radial coefficients again bends straight
        # scene lines in views such as 41, 42, and 75, so keep the pixels as-is.
        new_k = calib["K"]
        if APPLY_UNDISTORT:
            processed = cv2.undistort(img, calib["K"], calib["dist"], None, new_k)
            image_processing = "opencv_undistort_same_k"
        else:
            processed = img
            image_processing = "original_frame_no_warp"
        out_name = f"{view_id}.jpg"
        out_path = OUT_IMAGE_DIR / out_name
        ok = cv2.imwrite(str(out_path), processed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"Failed to write {out_path}")

        fl_x = float(new_k[0, 0])
        fl_y = float(new_k[1, 1])
        cx = float(new_k[0, 2])
        cy = float(new_k[1, 2])
        frames.append(
            {
                "file_path": f"images/{out_name}",
                "view_id": view_id,
                "grid_row": view_to_grid(view_id)[0],
                "grid_col": view_to_grid(view_id)[1],
                "w": w,
                "h": h,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy,
                "camera_angle_x": 2.0 * math.atan(w / (2.0 * fl_x)),
                "camera_angle_y": 2.0 * math.atan(h / (2.0 * fl_y)),
                "distortion_source": {
                    "k1": calib["k1"],
                    "k2": calib["k2"],
                    "p1": calib["p1"],
                    "p2": calib["p2"],
                },
                "image_processing": image_processing,
                "transform_matrix": c2w_for_view(view_id).tolist(),
            }
        )

    median_fl_x = float(np.median([f["fl_x"] for f in frames]))
    median_fl_y = float(np.median([f["fl_y"] for f in frames]))
    median_cx = float(np.median([f["cx"] for f in frames]))
    median_cy = float(np.median([f["cy"] for f in frames]))
    w = frames[0]["w"]
    h = frames[0]["h"]

    transforms = {
        "camera_model": "PINHOLE",
        "w": w,
        "h": h,
        "fl_x": median_fl_x,
        "fl_y": median_fl_y,
        "cx": median_cx,
        "cy": median_cy,
        "camera_angle_x": 2.0 * math.atan(w / (2.0 * median_fl_x)),
        "camera_angle_y": 2.0 * math.atan(h / (2.0 * median_fl_y)),
        "coordinate_note": "Approximate 10x10 planar light-field rig: row-major view ids 1..100, unit baseline, parallel optical axes. Intrinsics are from calibration mats scaled to the extracted frames. Images are original frames with no additional undistortion warp because the videos are already visually corrected.",
        "frames": frames,
    }
    (OUT_DIR / "transforms_grid_approx.json").write_text(json.dumps(transforms, indent=2), encoding="ascii")

    with (OUT_DIR / "intrinsics_per_view.txt").open("w", encoding="ascii") as f:
        f.write("# view_id original_fx original_fy original_cx original_cy k1 k2 p1 p2 undistorted_fx undistorted_fy undistorted_cx undistorted_cy\n")
        for calib, frame in zip(calibrations, frames):
            f.write(
                f"{calib['view_id']} {calib['fx']:.12g} {calib['fy']:.12g} {calib['cx']:.12g} {calib['cy']:.12g} "
                f"{calib['k1']:.12g} {calib['k2']:.12g} {calib['p1']:.12g} {calib['p2']:.12g} "
                f"{frame['fl_x']:.12g} {frame['fl_y']:.12g} {frame['cx']:.12g} {frame['cy']:.12g}\n"
            )

    print("output", OUT_DIR)
    print("images", len(frames))
    print("median intrinsics", median_fl_x, median_fl_y, median_cx, median_cy)
    print("first frame", frames[0])


if __name__ == "__main__":
    main()
