from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import shutil

import cv2


SRC_DIR = Path(".")
DST_DIR = Path(".")
SRC_IMAGE_DIR = SRC_DIR / "images"
DST_IMAGE_DIR = DST_DIR / "images"

CROP_X = 120
CROP_Y = 120
CROP_W = 1500
CROP_H = 800


def update_intrinsics(frame: dict) -> None:
    frame["w"] = CROP_W
    frame["h"] = CROP_H
    frame["cx"] = float(frame["cx"]) - CROP_X
    frame["cy"] = float(frame["cy"]) - CROP_Y
    frame["camera_angle_x"] = 2.0 * math.atan(CROP_W / (2.0 * float(frame["fl_x"])))
    frame["camera_angle_y"] = 2.0 * math.atan(CROP_H / (2.0 * float(frame["fl_y"])))
    frame["crop"] = {
        "source_w": 1920,
        "source_h": 1056,
        "x": CROP_X,
        "y": CROP_Y,
        "w": CROP_W,
        "h": CROP_H,
        "note": "Principal point shifted by subtracting crop x/y. Distortion coefficients are unchanged.",
    }


def main() -> None:
    global SRC_DIR, DST_DIR, SRC_IMAGE_DIR, DST_IMAGE_DIR, CROP_X, CROP_Y, CROP_W, CROP_H
    parser = argparse.ArgumentParser(description="Crop a desk light-field image set and update transforms.")
    parser.add_argument("--src_dir", required=True, type=str)
    parser.add_argument("--dst_dir", required=True, type=str)
    parser.add_argument("--crop_x", default=CROP_X, type=int)
    parser.add_argument("--crop_y", default=CROP_Y, type=int)
    parser.add_argument("--crop_w", default=CROP_W, type=int)
    parser.add_argument("--crop_h", default=CROP_H, type=int)
    args = parser.parse_args()
    SRC_DIR = Path(args.src_dir).resolve()
    DST_DIR = Path(args.dst_dir).resolve()
    SRC_IMAGE_DIR = SRC_DIR / "images"
    DST_IMAGE_DIR = DST_DIR / "images"
    CROP_X = args.crop_x
    CROP_Y = args.crop_y
    CROP_W = args.crop_w
    CROP_H = args.crop_h
    DST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    with (SRC_DIR / "transforms.json").open("r", encoding="ascii") as f:
        transforms = json.load(f)

    for frame in transforms["frames"]:
        src = SRC_DIR / frame["file_path"]
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read {src}")
        h, w = image.shape[:2]
        if CROP_X < 0 or CROP_Y < 0 or CROP_X + CROP_W > w or CROP_Y + CROP_H > h:
            raise RuntimeError(f"Crop outside image for {src}: image={w}x{h}")
        cropped = image[CROP_Y : CROP_Y + CROP_H, CROP_X : CROP_X + CROP_W]
        dst = DST_DIR / frame["file_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(dst), cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"Failed to write {dst}")
        update_intrinsics(frame)

    transforms["w"] = CROP_W
    transforms["h"] = CROP_H
    transforms["cx"] = float(transforms["cx"]) - CROP_X
    transforms["cy"] = float(transforms["cy"]) - CROP_Y
    transforms["camera_angle_x"] = 2.0 * math.atan(CROP_W / (2.0 * float(transforms["fl_x"])))
    transforms["camera_angle_y"] = 2.0 * math.atan(CROP_H / (2.0 * float(transforms["fl_y"])))
    transforms["crop"] = {
        "source_w": 1920,
        "source_h": 1056,
        "x": CROP_X,
        "y": CROP_Y,
        "w": CROP_W,
        "h": CROP_H,
        "placement": "middle-left",
        "purpose": "remove top-left camera text and bottom-right timestamp text",
    }
    transforms["coordinate_note"] += (
        f" Images are cropped to {CROP_W}x{CROP_H} from x={CROP_X}, y={CROP_Y}; "
        "cx/cy are shifted to the cropped image coordinate system."
    )

    (DST_DIR / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="ascii")

    for aux_name in ["camera_centers_mm.txt", "intrinsics_per_view.txt", "camera_centers_diagnostic.png"]:
        src = SRC_DIR / aux_name
        if src.exists():
            shutil.copy2(src, DST_DIR / aux_name)

    print("wrote", DST_DIR)
    print("crop", CROP_X, CROP_Y, CROP_W, CROP_H)
    print("images", len(transforms["frames"]))
    print("transforms", DST_DIR / "transforms.json")


if __name__ == "__main__":
    main()
