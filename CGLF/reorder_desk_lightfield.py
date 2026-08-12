from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import shutil

import numpy as np


SRC_DIR = Path(".")
DST_DIR = Path(".")
SRC_IMAGE_DIR = SRC_DIR / "images"
DST_IMAGE_DIR = DST_DIR / "images"
GRID_W = 10
GRID_H = 10


def camera_center(frame: dict) -> np.ndarray:
    if "transform_matrix_opencv" in frame:
        m = np.array(frame["transform_matrix_opencv"], dtype=float)
    else:
        m = np.array(frame["transform_matrix"], dtype=float)
    return m[:3, 3]


def main() -> None:
    global SRC_DIR, DST_DIR, SRC_IMAGE_DIR, DST_IMAGE_DIR, GRID_W, GRID_H
    parser = argparse.ArgumentParser(description="Reorder a light-field image set into row_col order by camera centers.")
    parser.add_argument("--src_dir", required=True, type=str)
    parser.add_argument("--dst_dir", required=True, type=str)
    parser.add_argument("--grid_w", default=GRID_W, type=int)
    parser.add_argument("--grid_h", default=GRID_H, type=int)
    args = parser.parse_args()
    SRC_DIR = Path(args.src_dir).resolve()
    DST_DIR = Path(args.dst_dir).resolve()
    SRC_IMAGE_DIR = SRC_DIR / "images"
    DST_IMAGE_DIR = DST_DIR / "images"
    GRID_W = args.grid_w
    GRID_H = args.grid_h
    with (SRC_DIR / "transforms.json").open("r", encoding="ascii") as f:
        transforms = json.load(f)

    frames = transforms["frames"]
    if len(frames) != GRID_W * GRID_H:
        raise RuntimeError(f"Expected {GRID_W * GRID_H} frames, got {len(frames)}")

    indexed = [(frame, camera_center(frame)) for frame in frames]
    # Current transforms use a centered OpenCV camera coordinate system where
    # larger Y is physically higher in the array and smaller X is left.
    rows_by_y = sorted(indexed, key=lambda item: -item[1][1])

    ordered_frames = []
    mapping_lines = [
        "# new_name new_row new_col old_file original_view_id center_x center_y center_z extrinsics_source"
    ]
    DST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for row in range(GRID_H):
        row_items = rows_by_y[row * GRID_W : (row + 1) * GRID_W]
        row_items = sorted(row_items, key=lambda item: item[1][0])
        for col, (frame, center) in enumerate(row_items):
            new_name = f"{row:02d}_{col:02d}.jpg"
            old_file = frame["file_path"]
            src = SRC_DIR / old_file
            dst = DST_IMAGE_DIR / new_name
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, dst)

            new_frame = dict(frame)
            new_frame["original_file_path"] = old_file
            new_frame["original_view_id"] = frame.get("original_view_id", frame.get("view_id"))
            new_frame["view_id"] = row * GRID_W + col + 1
            new_frame["grid_row"] = row
            new_frame["grid_col"] = col
            new_frame["file_path"] = f"images/{new_name}"
            new_frame["lightfield_order"] = "left_to_right_top_to_bottom_by_transform_camera_center"
            ordered_frames.append(new_frame)

            mapping_lines.append(
                f"{new_name} {row} {col} {old_file} {new_frame['original_view_id']} "
                f"{center[0]:.12g} {center[1]:.12g} {center[2]:.12g} "
                f"{frame.get('extrinsics_source', '')}"
            )

    transforms["frames"] = ordered_frames
    transforms["ordering_note"] = (
        "Images are reordered by transform_matrix_opencv camera centers: rows are sorted "
        "by descending camera-center Y, and columns by ascending camera-center X. Filenames "
        "use row_col format from 00_00 to 09_09, left-to-right and top-to-bottom."
    )
    transforms["camera_angle_x"] = 2.0 * math.atan(float(transforms["w"]) / (2.0 * float(transforms["fl_x"])))
    transforms["camera_angle_y"] = 2.0 * math.atan(float(transforms["h"]) / (2.0 * float(transforms["fl_y"])))

    (DST_DIR / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="ascii")
    (DST_DIR / "lightfield_order_mapping.txt").write_text("\n".join(mapping_lines) + "\n", encoding="ascii")

    for aux_name in ["camera_centers_mm.txt", "intrinsics_per_view.txt", "camera_centers_diagnostic.png"]:
        src = SRC_DIR / aux_name
        if src.exists():
            shutil.copy2(src, DST_DIR / aux_name)

    print("wrote", DST_DIR)
    print("images", len(ordered_frames))
    print("first row original ids", [f["original_view_id"] for f in ordered_frames[:GRID_W]])
    print("last row original ids", [f["original_view_id"] for f in ordered_frames[-GRID_W:]])


if __name__ == "__main__":
    main()
