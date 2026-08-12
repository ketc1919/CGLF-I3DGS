from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pycolmap


ROOT = Path(".")
IMAGE_DIR = ROOT
POSITION_FILE = ROOT / "camera_positions.txt"
PROJECT_DIR = ROOT / "colmap_fixed_pose"
DB_PATH = PROJECT_DIR / "database.db"
PAIR_LIST = PROJECT_DIR / "grid_pairs.txt"
INPUT_MODEL = PROJECT_DIR / "input_model"
SPARSE_MODEL = PROJECT_DIR / "sparse"
IMAGE_SIZE = (3888, 2592)
GRID_SIZE = 17


def parse_positions() -> list[dict]:
    rows = []
    for line in POSITION_FILE.read_text(encoding="ascii").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Bad position row: {line}")
        name = parts[0]
        row, col = map(int, Path(name).stem.split("_"))
        rows.append(
            {
                "name": name,
                "row": row,
                "col": col,
                "x": float(parts[1]),
                "y": float(parts[2]),
            }
        )
    rows.sort(key=lambda r: (r["row"], r["col"]))
    if len(rows) != GRID_SIZE * GRID_SIZE:
        raise ValueError(f"Expected 289 positions, got {len(rows)}")
    for item in rows:
        if not (IMAGE_DIR / item["name"]).exists():
            raise FileNotFoundError(IMAGE_DIR / item["name"])
    return rows


def write_known_pose_model(rows: list[dict]) -> None:
    INPUT_MODEL.mkdir(parents=True, exist_ok=True)
    width, height = IMAGE_SIZE
    cx = width / 2.0
    cy = height / 2.0
    # SIMPLE_PINHOLE: f, cx, cy. Principal point is fixed by BA options.
    initial_focal = float(max(width, height))

    (INPUT_MODEL / "cameras.txt").write_text(
        "\n".join(
            [
                "# Camera list with one line of data per camera:",
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
                "# Number of cameras: 1",
                f"1 SIMPLE_PINHOLE {width} {height} {initial_focal:.12g} {cx:.12g} {cy:.12g}",
                "",
            ]
        ),
        encoding="ascii",
    )

    image_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(rows)}, mean observations per image: 0",
    ]
    for image_id, item in enumerate(rows, start=1):
        # Identity world-to-camera rotation, camera center C=(x,y,0).
        # COLMAP stores t = -R*C, so t=(-x,-y,0).
        tx = -item["x"]
        ty = -item["y"]
        tz = 0.0
        image_lines.append(
            f"{image_id} 1 0 0 0 {tx:.12g} {ty:.12g} {tz:.12g} 1 {item['name']}"
        )
        image_lines.append("")
    (INPUT_MODEL / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="ascii")

    (INPUT_MODEL / "points3D.txt").write_text(
        "\n".join(
            [
                "# 3D point list with one line of data per point:",
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
                "# Number of points: 0, mean track length: 0",
                "",
            ]
        ),
        encoding="ascii",
    )


def write_grid_pairs(rows: list[dict], radius: int = 2) -> None:
    by_rc = {(r["row"], r["col"]): r["name"] for r in rows}
    pairs: set[tuple[str, str]] = set()
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            a = by_rc[(r, c)]
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
                        b = by_rc[(rr, cc)]
                        pairs.add(tuple(sorted((a, b))))
    PAIR_LIST.write_text("\n".join(f"{a} {b}" for a, b in sorted(pairs)) + "\n", encoding="ascii")


def main() -> None:
    global ROOT, IMAGE_DIR, POSITION_FILE, PROJECT_DIR, DB_PATH, PAIR_LIST, INPUT_MODEL, SPARSE_MODEL, IMAGE_SIZE, GRID_SIZE
    parser = argparse.ArgumentParser(description="Run pycolmap with fixed light-field poses.")
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--position_file", default="", type=str)
    parser.add_argument("--project_dir", default="", type=str)
    parser.add_argument("--image_width", default=IMAGE_SIZE[0], type=int)
    parser.add_argument("--image_height", default=IMAGE_SIZE[1], type=int)
    parser.add_argument("--grid_size", default=GRID_SIZE, type=int)
    args = parser.parse_args()
    ROOT = Path(args.root).resolve()
    IMAGE_DIR = ROOT
    POSITION_FILE = Path(args.position_file).resolve() if args.position_file else ROOT / "camera_positions.txt"
    PROJECT_DIR = Path(args.project_dir).resolve() if args.project_dir else ROOT / "colmap_fixed_pose"
    DB_PATH = PROJECT_DIR / "database.db"
    PAIR_LIST = PROJECT_DIR / "grid_pairs.txt"
    INPUT_MODEL = PROJECT_DIR / "input_model"
    SPARSE_MODEL = PROJECT_DIR / "sparse"
    IMAGE_SIZE = (args.image_width, args.image_height)
    GRID_SIZE = args.grid_size
    rows = parse_positions()

    if PROJECT_DIR.exists():
        shutil.rmtree(PROJECT_DIR)
    PROJECT_DIR.mkdir(parents=True)

    write_known_pose_model(rows)
    write_grid_pairs(rows, radius=2)

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "SIMPLE_PINHOLE"
    reader_options.camera_params = f"{max(IMAGE_SIZE):.12g},{IMAGE_SIZE[0] / 2:.12g},{IMAGE_SIZE[1] / 2:.12g}"

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.max_image_size = 2400
    extraction_options.num_threads = 4
    extraction_options.use_gpu = False
    extraction_options.sift.max_num_features = 12000

    image_names = [r["name"] for r in rows]
    print("Extracting features...")
    pycolmap.extract_features(
        DB_PATH,
        IMAGE_DIR,
        image_names=image_names,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
    )

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.num_threads = 4
    matching_options.use_gpu = False
    matching_options.sift.max_ratio = 0.85

    pairing_options = pycolmap.ImportedPairingOptions()
    pairing_options.match_list_path = PAIR_LIST

    print("Matching grid-neighbor image pairs...")
    pycolmap.match_image_pairs(
        DB_PATH,
        matching_options=matching_options,
        pairing_options=pairing_options,
    )

    pipeline_options = pycolmap.IncrementalPipelineOptions()
    pipeline_options.fix_existing_frames = True
    pipeline_options.mapper.fix_existing_frames = True
    pipeline_options.ba_refine_focal_length = True
    pipeline_options.ba_refine_principal_point = False
    pipeline_options.ba_refine_extra_params = False
    pipeline_options.triangulation.ignore_two_view_tracks = False
    pipeline_options.triangulation.min_angle = 0.01
    pipeline_options.extract_colors = True

    print("Triangulating points with fixed known poses and refining focal length...")
    reconstruction = pycolmap.triangulate_points(
        pycolmap.Reconstruction(INPUT_MODEL),
        DB_PATH,
        IMAGE_DIR,
        SPARSE_MODEL,
        clear_points=True,
        options=pipeline_options,
        refine_intrinsics=True,
    )

    print(reconstruction.summary())
    for camera in reconstruction.cameras.values():
        print("Estimated camera:", camera)


if __name__ == "__main__":
    main()
